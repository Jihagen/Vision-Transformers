"""
Runner: hidden states of the transfer-task images, clean and perturbed.

For every sample of a frozen Tier IV.2 manifest and every perturbation
condition (clean, the trained UAPs, and any extra sets passed with
--uap_roots), run one forward pass up to --layer_key under the task's own
evaluator prompt and save the hidden state at two positions (both read from
the same pass; attention is causal, so the first is unaffected by the anchor):

    prompt_end   last token of the chat prompt (where generation starts)
    label_slot   last token after teacher-forcing the start of the answer,
                 '{\\n  "label":' (the position that predicts the label; the
                 analogue of the rating-token position the directions use)

No text is generated. Perturbation mechanics are those of
attack/IV2_generalisation/perturbation.py (delta added to pixel_values,
clamped to [-1, 1]).

Output (in --output_dir/<task>/):
    hidden_<layer_key>_<position>.npy   float32, (n_samples, n_conditions, D)
    index.json                          sample_ids, conditions, layer, positions

Usage
-----
    python attack/runners/run_transfer_activations.py --tasks moral_evaluation \\
        --uap_roots noise_seed0=results/universal_perturbation_controls/noise_seed0 \\
        --extra_epsilons 0.5 1.0
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from utils.paths import LOCAL_MODEL_REPO as _DEFAULT_MODEL_PATH
from utils.paths import resolve_data_path

_ALL_TASKS = ["shopping_relevance", "damage_severity", "moral_evaluation"]
LABEL_ANCHOR = '{\n  "label":'
POSITIONS = ["prompt_end", "label_slot"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hidden states of transfer-task images, clean and perturbed.")
    p.add_argument("--tasks", nargs="+", choices=_ALL_TASKS, default=["moral_evaluation"])
    p.add_argument("--layer_key", default="language_29_D5120")
    p.add_argument("--output_dir", default="results/transfer_activations")
    p.add_argument("--uap_roots", nargs="+", default=[], metavar="NAME=DIR",
                   help="Extra perturbation sets, evaluated at --extra_epsilons.")
    p.add_argument("--extra_epsilons", nargs="+", type=float, default=[0.5, 1.0])
    p.add_argument("--max_side", type=int, default=336,
                   help="Must match max_side used during UAP training (default 336).")
    p.add_argument("--model_path", default=_DEFAULT_MODEL_PATH)
    p.add_argument("--offload_suffix", default="transfer_act")
    p.add_argument("--limit", type=int, default=None, help="Cap samples per task.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    from attack.IV1_gradient_matching.perturb import _EarlyStop
    from attack.IV2_generalisation import datasets as ds
    from attack.IV2_generalisation.perturbation import (
        build_model_inputs, discover_uap_conditions, load_delta,
    )
    from attack.IV2_generalisation.tasks import TASKS
    from utils.hooks import _inspect_model_structure
    from utils.image_utils import preprocess_image

    found = discover_uap_conditions()
    if args.uap_roots:
        found += discover_uap_conditions(dict(r.split("=", 1) for r in args.uap_roots), args.extra_epsilons)
    conditions = [("clean", None, None)] + [(a, e, load_delta(p)) for a, e, p in found]
    names = ["clean"] + [f"{a}_eps{e:.2f}" for a, e, _ in found]
    logger.info(f"{len(names)} conditions: {names}")

    manifests = {t: ds.load_manifest(t) for t in args.tasks}

    from utils.model_loader import load_model_and_processor
    from utils.hpc import get_offload_dir, setup_cuda_env

    setup_cuda_env()
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "38GiB" for i in range(n_gpus)} if n_gpus >= 6 else None
    offload = get_offload_dir(base=Path(args.model_path).parent / "offload_dir", suffix=args.offload_suffix)
    model, processor = load_model_and_processor(
        args.model_path, offload_dir=str(offload) if offload else None, max_memory=max_memory,
    )
    model.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device
    anchor = torch.tensor([processor.tokenizer.encode(LABEL_ANCHOR, add_special_tokens=False)], device=device)

    # Capture the target layer's output and stop the forward pass right after it.
    llm_layers, _, _ = _inspect_model_structure(model)
    layer_idx = int(args.layer_key.split("_")[1])
    captured: dict = {}

    def _capture(module, _, output):
        captured["h"] = (output[0] if isinstance(output, (tuple, list)) else output)[0]

    def _stop(module, inp):
        raise _EarlyStop()

    llm_layers[layer_idx].register_forward_hook(_capture)
    if layer_idx + 1 < len(llm_layers):
        llm_layers[layer_idx + 1].register_forward_pre_hook(_stop)

    def hidden_states(ids, mask, pixels):
        """(prompt_end, label_slot) hidden states, each (D,) float32."""
        forced_ids = torch.cat([ids, anchor], dim=1)
        forced_mask = torch.cat([mask, torch.ones_like(anchor)], dim=1)
        captured.clear()
        with torch.no_grad():
            try:
                model(input_ids=forced_ids, attention_mask=forced_mask,
                      pixel_values=pixels.to(model.dtype), use_cache=False)
            except _EarlyStop:
                pass
        h = captured["h"].float().cpu().numpy()
        return h[ids.shape[1] - 1], h[-1]

    for task_name, manifest in manifests.items():
        task = TASKS[task_name]
        rows = manifest.head(args.limit) if args.limit else manifest
        out_dir = Path(args.output_dir) / task_name
        out_dir.mkdir(parents=True, exist_ok=True)
        hidden = {pos: [] for pos in POSITIONS}
        sample_ids, cache = [], {}

        for i, (_, row) in enumerate(rows.iterrows()):
            try:
                image = preprocess_image(resolve_data_path(row["img_path"]), max_side=args.max_side)
            except Exception as e:
                logger.warning(f"  [{i+1}/{len(rows)}] skip {row['sample_id']}: image load failed — {e}")
                continue
            prompt = task.prompt_builder(row)
            per_pos = {pos: [] for pos in POSITIONS}
            for _, _, delta_np in conditions:
                ids, mask, pixels = build_model_inputs(processor, image, prompt, device, delta_np, cache)
                for pos, h in zip(POSITIONS, hidden_states(ids, mask, pixels)):
                    per_pos[pos].append(h)
            for pos in POSITIONS:
                hidden[pos].append(np.stack(per_pos[pos]))
            sample_ids.append(row["sample_id"])

            if (i + 1) % 25 == 0 or (i + 1) == len(rows):
                logger.info(f"  {task_name} [{i+1}/{len(rows)}]")
                for pos in POSITIONS:
                    np.save(out_dir / f"hidden_{args.layer_key}_{pos}.npy",
                            np.stack(hidden[pos]).astype(np.float32))
                (out_dir / "index.json").write_text(json.dumps({
                    "task": task_name, "layer_key": args.layer_key, "positions": POSITIONS,
                    "label_anchor": LABEL_ANCHOR, "conditions": names, "sample_ids": sample_ids,
                    "max_side": args.max_side}, indent=1))

    print(f"\n✅ Hidden states -> {args.output_dir}")


if __name__ == "__main__":
    main()
