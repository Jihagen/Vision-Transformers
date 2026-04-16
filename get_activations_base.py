#!/usr/bin/env python
"""
get_activations_base.py
=======================
Shared engine for all persona activation experiments.

Provides:
  - Model loading & hook registration
  - build_persona_prompt(persona_dict)  — constructs the full rating prompt
  - call_model_with_retries(...)        — inference with OOM / JSON fallbacks
  - run_experiment(...)                 — main loop over a list of persona dicts

Experiment scripts (run_synonym_test.py, run_gender_emotion.py, …) import
this module, define their persona list, and call run_experiment().
"""
from __future__ import annotations
import os, sys, logging, json, re, gc
from typing import Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─── Local HF/dev path (fallback) ────────────────────────────────────────────
DEVROOT = os.environ.get("DEVROOT", "/anvme/workspace/iwi5268h-vision-transformers/dev")
HF_DEPS = os.path.join(DEVROOT, "hf_deps")
TF_SRC  = os.path.join(DEVROOT, "transformers", "src")

if os.path.isdir(HF_DEPS):
    sys.path.insert(0, HF_DEPS)
if os.path.isdir(TF_SRC):
    sys.path.insert(0, TF_SRC)

try:
    import huggingface_hub.utils._validators as _hf_val
    _hf_val.validate_repo_id = lambda *a, **k: None
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    import transformers.utils.hub as _tfh
    _tfh.validate_repo_id = lambda *a, **k: None
except Exception:
    pass

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Mutable hook state (module-level so closures share it) ──────────────────
layer_embeddings: dict = {}
hook_handles:     list = []
hook_call_count:  dict = {}
CAPTURE_PHASE:    str  = "idle"   # "vision_once" | "rating_step" | "idle"


# ─── General utilities ───────────────────────────────────────────────────────
def print_cuda_memory(prefix: str = "") -> None:
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved()  / 2**30
        logger.debug(f"{prefix} CUDA mem → Allocated: {a:.2f} GiB, Reserved: {r:.2f} GiB")


def emergency_cleanup(preserve_embeddings: bool = False) -> None:
    global layer_embeddings
    if not preserve_embeddings:
        layer_embeddings.clear()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"⚠️ empty_cache failed: {e}")
        try:
            torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning(f"⚠️ ipc_collect failed: {e}")
    gc.collect()


def load_manifest(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing manifest: {path}")
    df = pd.read_pickle(path)
    required = {"filename", "img_path"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must have {required}, got {set(df.columns)}")
    return df.copy()


def get_missing_rows(manifest_df: pd.DataFrame, results_list: list[dict]) -> pd.DataFrame:
    done = {r.get("filename") for r in results_list if isinstance(r, dict) and "filename" in r}
    return manifest_df[~manifest_df["filename"].isin(done)].reset_index(drop=True)


def _load_existing_results(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        obj = np.load(path, allow_pickle=True).item()
        return obj.get("results", []) if isinstance(obj, dict) else []
    except Exception as e:
        logger.warning(f"⚠️ Could not load existing results from {path}: {e}")
        return []


def _save_checkpoint(
    ckpt_path: str,
    ckpt_temp: str,
    results_list: list,
    manifest_df: pd.DataFrame,
    persona_key: str,
) -> None:
    payload = {
        "version": "experiment_v1",
        "persona_key": persona_key,
        "selected_images": manifest_df[["filename", "img_path"]].to_dict(orient="records"),
        "results": results_list,
    }
    np.save(ckpt_temp, payload)
    os.replace(ckpt_temp, ckpt_path)
    logger.info(f"  Checkpoint saved → {ckpt_path}")


def remove_batch_dimension(t: torch.Tensor) -> np.ndarray:
    try:
        cpu = t.to(torch.float32).cpu()
        if cpu.dim() > 0:
            unbatched = cpu.squeeze(0) if cpu.shape[0] == 1 else cpu[0]
        else:
            unbatched = cpu
        return unbatched.numpy()
    except Exception as e:
        logger.warning(f"remove_batch_dimension error: {e}")
        return t.to(torch.float32).cpu().numpy()


def preprocess_image(image_path: str, max_size: int = 800) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image from {w}x{h} to {int(w*ratio)}x{int(h*ratio)}")
    return image


def extract_last_json_block(text: str) -> dict:
    for raw in reversed(re.findall(r"\{.*?\}", text, flags=re.S)):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON found")


# ─── Prompt building ──────────────────────────────────────────────────────────
_RATING_SUFFIX = (
    'You see this image; rate its interestingness from:\n'
    '"Not Interesting","Slightly Interesting","Moderately Interesting",\n'
    '"Very Interesting","Extremely Interesting"\n\n'
    'Return ONLY valid JSON.\n'
    'Do not use markdown.\n'
    'Do not use code fences.\n'
    'Do not add any text before or after the JSON.\n\n'
    'Keys:\n'
    '- "interestingness": one of the five strings above\n'
    '- "explanation": exactly 1 short sentence'
)


def build_persona_prompt(persona_dict: dict) -> str:
    """
    Build the full interestingness-rating prompt for a given persona.

    Reserved keys (stripped from the persona description):
      persona_key  — checkpoint filename identifier
      _format      — "raw"     : values only, comma-separated (matches original
                                  get_activations.py style; use for synonym test)
                   — "labeled" : "key: value" pairs (default; use for new experiments)

    All other keys are included in persona description order (Python 3.7+ insertion order).
    """
    fmt    = persona_dict.get("_format", "labeled")
    fields = {k: v for k, v in persona_dict.items()
              if k != "persona_key" and not k.startswith("_")}

    if fmt == "raw":
        persona_str = ", ".join(str(v) for v in fields.values())
    else:
        persona_str = ", ".join(f"{k}: {v}" for k, v in fields.items())

    return (
        f"Imagine you are a person with the following details:\n"
        f"{persona_str}\n\n"
        f"{_RATING_SUFFIX}"
    )


# ─── Hook infrastructure ──────────────────────────────────────────────────────
def select_cls(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0, :]


def select_last_token(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1, :]


def get_hook(
    name: str,
    store_dict: dict,
    selector: Optional[callable] = None,
    phase_needed: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
):
    def hook(module, _, output):
        global hook_call_count, CAPTURE_PHASE
        if phase_needed and CAPTURE_PHASE != phase_needed:
            return
        hook_call_count[name] = hook_call_count.get(name, 0) + 1
        try:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if out is None or not hasattr(out, "detach"):
                return
            x = out.detach()
            if selector is not None:
                x = selector(x)
            store_dict[name] = x.to(dtype).cpu()
        except Exception as e:
            logger.warning(f"⚠️ Hook failed on {name}: {e}")
    return hook


def _inspect_model_structure(model):
    logger.info("🔍 Inspecting model structure...")

    llm_layers = None
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        if hasattr(model.language_model.model, "layers"):
            llm_layers = model.language_model.model.layers
            logger.info(f"Found {len(llm_layers)} language layers")

    vision_paths = [
        ("vision_model.model.layers",
         lambda: model.vision_model.model.layers
         if hasattr(model, "vision_model") and hasattr(model.vision_model, "model")
            and hasattr(model.vision_model.model, "layers") else None),
        ("vision_model.encoder.layers",
         lambda: model.vision_model.encoder.layers
         if hasattr(model, "vision_model") and hasattr(model.vision_model, "encoder")
            and hasattr(model.vision_model.encoder, "layers") else None),
        ("vision_model.vision_model.encoder.layers",
         lambda: model.vision_model.vision_model.encoder.layers
         if hasattr(model, "vision_model") and hasattr(model.vision_model, "vision_model")
            and hasattr(model.vision_model.vision_model, "encoder")
            and hasattr(model.vision_model.vision_model.encoder, "layers") else None),
    ]
    vision_layers = None
    vision_path_used = None
    for path_name, path_fn in vision_paths:
        try:
            layers = path_fn()
            if layers is not None:
                logger.info(f"Found {len(layers)} vision layers at {path_name}")
                vision_layers = layers
                vision_path_used = path_name
                break
        except Exception as e:
            logger.debug(f"Path {path_name} failed: {e}")

    if vision_layers is None and hasattr(model, "vision_model"):
        logger.warning("⚠️ Could not find vision layers")

    return llm_layers, vision_layers, vision_path_used


def register_hooks(model) -> None:
    global hook_handles
    llm_layers, vision_layers, vision_path = _inspect_model_structure(model)
    handles = []

    if llm_layers is not None:
        for i, layer in enumerate(llm_layers):
            h = layer.register_forward_hook(get_hook(
                name=f"llm_layer_{i}_rating_token",
                store_dict=layer_embeddings,
                selector=select_last_token,
                phase_needed="rating_step",
                dtype=torch.float16,
            ))
            handles.append(h)
        logger.info(f"Registered {len(llm_layers)} LLM hooks (phase=rating_step)")

    if vision_layers is not None:
        for i, layer in enumerate(vision_layers):
            h = layer.register_forward_hook(get_hook(
                name=f"vision_layer_{i}_cls",
                store_dict=layer_embeddings,
                selector=select_cls,
                phase_needed="vision_once",
                dtype=torch.float16,
            ))
            handles.append(h)
        logger.info(f"Registered {len(vision_layers)} vision hooks via {vision_path}")
    else:
        logger.warning("⚠️ No vision layers found for hooking")

    hook_handles = handles
    logger.info(f"Total hooks registered: {len(handles)}")


def cleanup_hooks() -> None:
    global hook_handles
    for h in hook_handles:
        h.remove()
    hook_handles.clear()
    logger.info("Removed all hooks")


# ─── Model inference ──────────────────────────────────────────────────────────
def model_response(
    prompt: str,
    image_path: str,
    model,
    processor,
    image_max_size: int = 800,
    max_new_tokens: int = 64,
) -> tuple[str, dict]:
    global hook_call_count, CAPTURE_PHASE

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    hook_call_count.clear()

    image = preprocess_image(image_path, max_size=image_max_size)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=chat_input,
        images=image,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        max_length=512,
    )

    target_device = next(model.parameters()).device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(target_device)

    whitelist = {"input_ids", "attention_mask", "pixel_values"}
    model_inputs = {k: inputs[k] for k in whitelist if k in inputs}

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    logger.info(f"🚀 Generation (image_max_size={image_max_size}, max_new_tokens={max_new_tokens})...")
    CAPTURE_PHASE = "vision_once"
    try:
        print_cuda_memory("Before generate")
        with torch.no_grad():
            gen_ids = model.generate(**model_inputs, **generation_kwargs)
        print_cuda_memory("After generate")
    finally:
        CAPTURE_PHASE = "idle"

    decoded = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    try:
        data = extract_last_json_block(decoded)
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}; raw tail: {decoded[-400:]}")
        raise ValueError("Invalid JSON in response")

    # Teacher-force one step right before the label token to capture LLM activations
    full_seq_ids = gen_ids[0].tolist()
    input_len    = model_inputs["input_ids"].shape[1]
    forced_anchor = '{"interestingness":"'
    forced_ids    = processor.tokenizer.encode(forced_anchor, add_special_tokens=False)
    prefix_ids    = full_seq_ids[:input_len] + forced_ids

    local = {
        "input_ids": torch.tensor([prefix_ids], device=model_inputs["input_ids"].device),
        "attention_mask": torch.ones(
            1, len(prefix_ids),
            device=model_inputs["attention_mask"].device,
            dtype=model_inputs["attention_mask"].dtype,
        ),
        "pixel_values": model_inputs["pixel_values"],
    }

    logger.info("🎯 Teacher-forced one-step for LLM layer capture...")
    CAPTURE_PHASE = "rating_step"
    with torch.no_grad():
        _ = model.generate(
            **local,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    CAPTURE_PHASE = "idle"

    del gen_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return decoded, data


def call_model_with_retries(
    prompt: str,
    image_path: str,
    model,
    processor,
) -> tuple[str, dict]:
    """
    Wraps model_response with OOM (image-downscale) and invalid-JSON (token-expand) fallbacks.
    Mirrors the retry logic from the original get_activations.py.
    """
    try:
        return model_response(prompt, image_path, model, processor,
                              image_max_size=800, max_new_tokens=64)

    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        last_exc = e
        for oom_size in [640, 512, 384, 256, 128]:
            logger.warning(f"⚠️ OOM; retrying with image_max_size={oom_size}")
            emergency_cleanup(preserve_embeddings=False)
            try:
                return model_response(prompt, image_path, model, processor,
                                      image_max_size=oom_size, max_new_tokens=64)
            except RuntimeError as e2:
                if "illegal memory access" in str(e2).lower():
                    logger.error("❌ CUDA illegal memory access — aborting.")
                    raise
                if "out of memory" in str(e2).lower():
                    last_exc = e2
                    continue
                raise
        logger.error(f"❌ OOM persists at minimum image size; giving up on this image.")
        raise last_exc

    except ValueError as e:
        if "Invalid JSON" not in str(e):
            raise
        for n_tokens in [96, 128]:
            logger.warning(f"⚠️ Invalid JSON; retrying with max_new_tokens={n_tokens}")
            emergency_cleanup(preserve_embeddings=False)
            try:
                return model_response(prompt, image_path, model, processor,
                                      image_max_size=800, max_new_tokens=n_tokens)
            except ValueError as e2:
                if "Invalid JSON" not in str(e2):
                    raise
        raise


# ─── Main experiment runner ───────────────────────────────────────────────────
def run_experiment(
    model_path: str,
    offload_dir: str,
    output_dir: str,
    personas: list[dict],
    manifest_path: str = "data/selected_uniform_500_manifest.pkl",
    checkpoint_every: int = 2,
    max_retry_rounds: int = 5,
) -> None:
    """
    Run activation extraction for a list of persona dicts.

    Each persona dict must contain:
      "persona_key"  — str, used as the checkpoint filename stem
      <field keys>   — included in the persona description (see build_persona_prompt)
      "_format"      — optional, "raw" or "labeled" (default "labeled")

    Checkpoints:  {output_dir}/results_{persona_key}.npy
    Missing logs: {output_dir}/missing_{persona_key}.pkl
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(offload_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from {model_path} ...")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        offload_folder=offload_dir,
        offload_state_dict=True,
    )
    model.eval()

    if hasattr(model, "hf_device_map"):
        for k, v in model.hf_device_map.items():
            logger.info(f"  {k} → {v}")
    print_cuda_memory("After model load")

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    register_hooks(model)

    manifest_df = load_manifest(manifest_path)
    logger.info(f"Loaded manifest with {len(manifest_df)} target images")

    try:
        for persona_index, persona_dict in enumerate(personas):
            persona_key = persona_dict["persona_key"]
            ckpt_path   = os.path.join(output_dir, f"results_{persona_key}.npy")
            ckpt_temp   = os.path.join(output_dir, f"results_{persona_key}_temp.npy")

            prompt = build_persona_prompt(persona_dict)
            persona_line = prompt.splitlines()[1]   # the description line

            logger.info(f"\n{'='*60}")
            logger.info(f"[{persona_index + 1}/{len(personas)}] persona_key={persona_key}")
            logger.info(f"  Persona: {persona_line}")

            existing     = _load_existing_results(ckpt_path)
            results_list = list(existing)
            logger.info(f"  Existing results: {len(results_list)}")

            total_new = 0

            for retry_round in range(max_retry_rounds + 1):
                missing_df = get_missing_rows(manifest_df, results_list)
                if len(missing_df) == 0:
                    logger.info(f"  ✅ All images processed for {persona_key}")
                    break

                logger.info(
                    f"  Round {retry_round}/{max_retry_rounds} | missing: {len(missing_df)}"
                )
                processed_this_round = 0

                for idx, row in enumerate(missing_df.itertuples(index=False), start=1):
                    img_path_str = str(row.img_path)
                    try:
                        logger.info(
                            f"  [{retry_round}] {idx}/{len(missing_df)}: {img_path_str}"
                        )
                        layer_embeddings.clear()

                        raw, data = call_model_with_retries(prompt, img_path_str, model, processor)

                        logger.info(f"  📊 Captured {len(layer_embeddings)} embedding tensors")
                        embeds = {k: remove_batch_dimension(v) for k, v in layer_embeddings.items()}
                        layer_embeddings.clear()

                        rec = {
                            "persona_key":    persona_key,
                            "filename":       os.path.basename(img_path_str),
                            "img_path":       img_path_str,
                            "interestingness": data["interestingness"],
                            "explanation":    data["explanation"],
                            "embeddings":     embeds,
                        }
                        results_list.append(rec)
                        total_new += 1
                        processed_this_round += 1

                        if len(results_list) % checkpoint_every == 0:
                            _save_checkpoint(ckpt_path, ckpt_temp, results_list, manifest_df, persona_key)

                    except Exception as e:
                        if "illegal memory access" in str(e).lower():
                            logger.error(
                                f"  ❌ CUDA illegal memory access on {img_path_str} — aborting run."
                            )
                            raise
                        logger.warning(f"  ❌ Error on {img_path_str}: {e}; skipping")
                        emergency_cleanup(preserve_embeddings=False)
                        continue

                _save_checkpoint(ckpt_path, ckpt_temp, results_list, manifest_df, persona_key)
                logger.info(f"  Round {retry_round} done; processed this round: {processed_this_round}")

                if processed_this_round == 0:
                    logger.warning("  No progress this round; stopping retries early.")
                    break

            final_missing = get_missing_rows(manifest_df, results_list)
            logger.info(
                f"  Total new for {persona_key}: {total_new} | "
                f"remaining missing: {len(final_missing)}"
            )

            if len(final_missing) > 0:
                missing_path = os.path.join(output_dir, f"missing_{persona_key}.pkl")
                final_missing.to_pickle(missing_path)
                logger.warning(f"  Saved missing rows → {missing_path}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user; last checkpoint is on disk.")
        raise
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        cleanup_hooks()
