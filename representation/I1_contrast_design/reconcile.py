"""
Cross-condition image size reconciliation.

Problem: the OOM downscale retry is non-deterministic — whether a given image
triggers OOM depends on GPU memory fragmentation at the exact moment it is
processed, which varies between collection runs.  If female_anger processes
image X at 800px but male_anger processed it at 640px (after hitting OOM that
time), the vision encoder produces different CLS-token activations for the same
image, corrupting the analysis.

Fix (conservative): after all conditions are collected, detect any images where
the recorded `image_max_size` differs across conditions, then re-run those
images in every affected condition at a single agreed-upon size.

Agreed size strategy:
  1. Try the full size (800px) — OOM at collection time is often transient; a
     retry may well succeed.
  2. If OOM still occurs, step down the DOWNSCALE_LADDER until it works.
  3. Record the agreed size in every condition's result file.

This guarantees all conditions use identical pixel inputs for every image, so
vision activations are comparable.  The number of images per condition remains
unchanged (no images dropped).

Usage
-----
Standalone analysis (no model needed):

    from representation.I1_contrast_design.reconcile import analyze_size_mismatches
    report = analyze_size_mismatches({'female_anger': Path(...), 'male_anger': Path(...)})

Full reconciliation (model must be passed in):

    from representation.I1_contrast_design.reconcile import reconcile_conditions
    counts = reconcile_conditions(
        result_paths={'female_anger': Path(...), ...},
        persona_dicts={'female_anger': {...}, ...},
        model=model,
        processor=processor,
    )
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

import numpy as np
import torch

from utils.image_utils import DOWNSCALE_LADDER, remove_batch_dimension
from utils.hooks import HookState, register_extract_hooks
from utils.hpc import emergency_cleanup

logger = logging.getLogger(__name__)

_DEFAULT_SIZE = DOWNSCALE_LADDER[0]   # 800


# ── Vision-activation comparison (works on old records without image_max_size) ─

def build_vision_table(
    result_paths: dict[str, Path],
    layer_key: str = "vision_0_D1408",
) -> dict[str, dict[str, np.ndarray]]:
    """
    Return {condition: {filename: vision_vec}} for all conditions.
    Uses vision layer activations directly — no model needed.
    """
    from representation.I1_contrast_design.load import (
        build_layer_matrix, get_results_list,
    )
    table: dict[str, dict[str, np.ndarray]] = {}
    for pk, path in result_paths.items():
        results = get_results_list(np.load(Path(path), allow_pickle=True).item())
        if not results:
            logger.warning(f"{pk}: 0 results — skipping in vision table")
            continue
        mat  = build_layer_matrix(results)
        if layer_key not in mat:
            logger.warning(f"{pk}: layer {layer_key!r} not found — skipping")
            continue
        X   = mat[layer_key]["X"]
        fns = [r["filename"] for r in results]
        table[pk] = {fn: X[i] for i, fn in enumerate(fns)}
    return table


def analyze_vision_mismatches(
    result_paths: dict[str, Path],
    layer_key: str = "vision_0_D1408",
    tol: float = 0.01,
) -> dict[str, dict[str, str]]:
    """
    Detect images whose vision activations differ across conditions.

    This works on old-format records that lack `image_max_size`.  Vision
    activations for the same image should be bit-for-bit identical across all
    conditions (the vision encoder is prompt-independent).  Any difference above
    `tol` indicates a resolution inconsistency caused by the OOM retry ladder.

    Args:
        result_paths: {persona_key: path_to_npy}
        layer_key:    which vision layer to compare (default: first layer)
        tol:          max-abs tolerance for declaring two vectors identical

    Returns:
        {filename: {persona_key: "majority" | "minority"}}
        Only images with at least one minority condition are included.
    """
    table = build_vision_table(result_paths, layer_key=layer_key)
    all_conds = list(table.keys())
    all_fns   = sorted({fn for vecs in table.values() for fn in vecs})

    mismatches: dict[str, dict[str, str]] = {}
    for fn in all_fns:
        vecs = {c: table[c][fn] for c in all_conds if fn in table[c]}
        if len(vecs) < 2:
            continue

        # Cluster by tolerance
        cond_list = list(vecs.keys())
        clusters: list[list[str]] = []
        assigned: set[str] = set()
        for c in cond_list:
            if c in assigned:
                continue
            group = [c]
            for c2 in cond_list:
                if c2 not in assigned and c2 != c:
                    if np.abs(vecs[c] - vecs[c2]).max() <= tol:
                        group.append(c2)
            clusters.append(group)
            assigned.update(group)

        if len(clusters) == 1:
            continue   # all identical

        majority = max(clusters, key=len)
        maj_set  = set(majority)
        labels = {c: ("majority" if c in maj_set else "minority") for c in cond_list}
        mismatches[fn] = labels

    return mismatches


def print_vision_mismatch_report(
    mismatches: dict[str, dict[str, str]],
) -> None:
    if not mismatches:
        print("No vision-activation mismatches — all conditions consistent.")
        return
    minority_conditions: dict[str, list[str]] = {}
    for fn, labels in mismatches.items():
        for cond, role in labels.items():
            if role == "minority":
                minority_conditions.setdefault(cond, []).append(fn)

    print(f"Found {len(mismatches)} images with cross-condition vision discrepancies.")
    print()
    print("Conditions that need targeted re-runs:")
    total_reruns = 0
    for cond, fns in sorted(minority_conditions.items()):
        print(f"  {cond}: {len(fns)} image(s) — {sorted(fns)}")
        total_reruns += len(fns)
    print(f"\nTotal targeted re-runs required: {total_reruns}")


# ── Analysis (model-free) ─────────────────────────────────────────────────────

def _load_results(path: Path) -> list[dict]:
    obj = np.load(path, allow_pickle=True).item()
    return obj.get("results", [])


def analyze_size_mismatches(
    result_paths: dict[str, Path],
) -> dict[str, dict[str, int]]:
    """
    Detect images where `image_max_size` differs across conditions.

    Args:
        result_paths: {persona_key: path_to_npy}

    Returns:
        {filename: {persona_key: size_used}}  — only images with discrepancies.
        Conditions that lack the `image_max_size` field are treated as 800
        (the default used before tracking was added).
    """
    # Build {filename: {persona_key: size}} table
    size_table: dict[str, dict[str, int]] = {}
    for pk, path in result_paths.items():
        for rec in _load_results(Path(path)):
            fn   = rec.get("filename")
            size = rec.get("image_max_size", _DEFAULT_SIZE)
            if fn is None:
                continue
            size_table.setdefault(fn, {})[pk] = size

    # Keep only images present in ALL conditions with at least one size mismatch
    mismatches: dict[str, dict[str, int]] = {}
    n_conditions = len(result_paths)
    for fn, sizes in size_table.items():
        if len(sizes) < n_conditions:
            continue   # not yet processed in all conditions
        if len(set(sizes.values())) > 1:
            mismatches[fn] = sizes

    return mismatches


def print_mismatch_report(mismatches: dict[str, dict[str, int]]) -> None:
    if not mismatches:
        print("No size mismatches — all conditions used the same image size.")
        return
    print(f"Found {len(mismatches)} images with size discrepancies:")
    for fn, sizes in sorted(mismatches.items()):
        row = "  ".join(f"{pk}={sz}px" for pk, sz in sorted(sizes.items()))
        print(f"  {fn}: {row}")


# ── Reconciliation (requires model) ──────────────────────────────────────────

def _run_single_image(
    prompt: str,
    img_path: str,
    model,
    processor,
    state: HookState,
    target_size: int,
) -> tuple[dict, int] | None:
    """
    Re-run one image at target_size, stepping down if still OOM.

    Returns:
        (result_record_fields, size_actually_used) or None on failure.
    """
    import gc
    from utils.prompt_builder import extract_last_json_block, RATING_ANCHOR

    # Import here to avoid circular at module level
    from representation.I1_contrast_design.collect import model_response

    ladder = [s for s in DOWNSCALE_LADDER if s <= target_size]
    if not ladder:
        ladder = DOWNSCALE_LADDER

    for size in ladder:
        state.reset_embeddings()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        try:
            _, data = model_response(
                prompt, img_path, model, processor, state,
                image_max_size=size, max_new_tokens=64,
            )
            embeds = {k: remove_batch_dimension(v) for k, v in state.embeddings.items()}
            state.reset_embeddings()
            return {"interestingness": data["interestingness"],
                    "explanation":     data["explanation"],
                    "image_max_size":  size,
                    "embeddings":      embeds}, size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at {size}px for {img_path}; trying smaller")
                emergency_cleanup(embeddings_dict=state.embeddings)
                state.reset_embeddings()
                continue
            logger.error(f"Runtime error on {img_path} at {size}px: {e}")
            return None
        except Exception as e:
            logger.error(f"Error on {img_path} at {size}px: {e}")
            return None

    logger.error(f"All sizes failed for {img_path}")
    return None


def _save_results_inplace(path: Path, updated_results: list[dict], original_obj: dict) -> None:
    """Atomically overwrite a result .npy file with an updated results list."""
    tmp = path.with_suffix(".npy_tmp")
    payload = {**original_obj, "results": updated_results}
    np.save(tmp, payload)
    os.replace(tmp, path)


def reconcile_by_vision(
    result_paths: dict[str, Path],
    persona_dicts: dict[str, dict | None],
    model,
    processor,
    layer_key: str = "vision_0_D1408",
    tol: float = 0.01,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Detect and fix cross-condition vision mismatches in-place.

    Unlike reconcile_conditions (which relies on stored image_max_size),
    this function detects mismatches by comparing the actual vision activations
    across conditions — it therefore works on old-format result files that
    predate the image_max_size tracking field.

    For each image where a minority of conditions produced a different vision
    activation (due to OOM-triggered downscaling), re-run that image in the
    minority conditions using the majority representation's resolution.

    Because the majority group's size is unknown from old records, this function
    tries the full size (800px) first and steps down only on OOM.  This matches
    the majority representation in the vast majority of cases (OOM at collection
    time is usually transient).

    Args:
        result_paths:  {persona_key: path_to_npy}
        persona_dicts: {persona_key: persona_dict_or_None}
        model:         loaded HuggingFace model
        processor:     loaded processor
        layer_key:     vision layer used for mismatch detection
        tol:           max-abs tolerance for declaring two vectors identical
        dry_run:       report only, do not re-run

    Returns:
        {persona_key: n_records_updated}
    """
    from utils.prompt_builder import build_persona_prompt

    mismatches = analyze_vision_mismatches(result_paths, layer_key=layer_key, tol=tol)
    print_vision_mismatch_report(mismatches)
    if not mismatches or dry_run:
        return {}

    # Build targeted re-run list: {persona_key: [filenames]}
    reruns: dict[str, list[str]] = {}
    for fn, labels in mismatches.items():
        for cond, role in labels.items():
            if role == "minority":
                reruns.setdefault(cond, []).append(fn)

    # Load result files into memory
    result_objs: dict[str, dict]         = {}
    results_by_pk: dict[str, list[dict]] = {}
    idx_maps: dict[str, dict[str, int]]  = {}
    for pk, path in result_paths.items():
        obj = np.load(Path(path), allow_pickle=True).item()
        result_objs[pk]   = obj
        recs = list(obj.get("results", []))
        results_by_pk[pk] = recs
        idx_maps[pk]      = {r["filename"]: i for i, r in enumerate(recs) if "filename" in r}

    state = HookState()
    register_extract_hooks(model, state)

    update_counts: dict[str, int] = {pk: 0 for pk in result_paths}

    try:
        for pk, fns in sorted(reruns.items()):
            prompt = build_persona_prompt(persona_dicts.get(pk))
            logger.info(f"Re-running {len(fns)} images for {pk}…")
            for fn in sorted(fns):
                rec_idx = idx_maps[pk].get(fn)
                if rec_idx is None:
                    logger.warning(f"  {fn} not found in {pk} index — skipping")
                    continue
                rec      = results_by_pk[pk][rec_idx]
                img_path = rec.get("img_path", "")
                logger.info(f"  {fn}: re-running for {pk}")
                result = _run_single_image(
                    prompt, img_path, model, processor, state,
                    target_size=_DEFAULT_SIZE,
                )
                if result is None:
                    logger.error(f"  Re-run failed for {fn}/{pk} — keeping original")
                    continue
                fields, actual_size = result
                results_by_pk[pk][rec_idx] = {**rec, **fields, "image_max_size": actual_size}
                update_counts[pk] += 1
                logger.info(f"  Updated {fn} for {pk} at {actual_size}px")
    finally:
        state.remove_hooks()

    for pk, path in result_paths.items():
        if update_counts[pk] > 0:
            _save_results_inplace(Path(path), results_by_pk[pk], result_objs[pk])
            logger.info(f"Saved {pk}: {update_counts[pk]} records updated → {path}")

    total = sum(update_counts.values())
    logger.info(f"Vision reconciliation complete. Total records updated: {total}")
    return update_counts


def reconcile_conditions(
    result_paths: dict[str, Path],
    persona_dicts: dict[str, dict | None],
    model,
    processor,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Detect and fix cross-condition image-size mismatches in-place.

    For each image where sizes differ across conditions:
      1. Determine the agreed size: start from the full size (800px) and step
         down only if OOM occurs (transient OOM at collection time may not
         recur on retry).
      2. Re-run the image in every condition that used a different size.
      3. Update the result file atomically.

    Args:
        result_paths:  {persona_key: path_to_npy}
        persona_dicts: {persona_key: persona_dict_or_None}
        model:         loaded HuggingFace model (already on device)
        processor:     loaded processor
        dry_run:       if True, only report mismatches without re-running

    Returns:
        {persona_key: n_records_updated}
    """
    from utils.prompt_builder import build_persona_prompt

    mismatches = analyze_size_mismatches(result_paths)
    if not mismatches:
        logger.info("No size mismatches found — nothing to reconcile.")
        return {}

    print_mismatch_report(mismatches)
    if dry_run:
        logger.info("dry_run=True — stopping before re-running images.")
        return {}

    # Load all result files into memory
    result_objs: dict[str, dict]        = {}
    results_by_pk: dict[str, list[dict]] = {}
    for pk, path in result_paths.items():
        obj = np.load(Path(path), allow_pickle=True).item()
        result_objs[pk]   = obj
        results_by_pk[pk] = list(obj.get("results", []))

    # Build {pk: {filename: list_index}} for fast lookup
    idx_maps: dict[str, dict[str, int]] = {}
    for pk, recs in results_by_pk.items():
        idx_maps[pk] = {r["filename"]: i for i, r in enumerate(recs) if "filename" in r}

    # Register hooks once (they are stateless)
    state = HookState()
    register_extract_hooks(model, state)

    update_counts: dict[str, int] = {pk: 0 for pk in result_paths}

    try:
        for fn, sizes in sorted(mismatches.items()):
            logger.info(f"Reconciling {fn}: sizes={sizes}")

            # Determine agreed size: try full size first (OOM may be transient)
            agreed_size = _DEFAULT_SIZE

            # Find which condition needs re-running first to establish agreed_size
            # Use the condition with the smallest current size as the "worst case" test
            test_pk = min(sizes, key=lambda pk: sizes[pk])
            test_idx = idx_maps[test_pk].get(fn)
            if test_idx is None:
                logger.warning(f"  {fn} not found in {test_pk} results — skipping")
                continue

            test_rec  = results_by_pk[test_pk][test_idx]
            test_path = test_rec.get("img_path", "")
            test_prompt = build_persona_prompt(persona_dicts.get(test_pk))

            # Probe at agreed_size (full resolution) to find what actually works
            probe = _run_single_image(
                test_prompt, test_path, model, processor, state, target_size=agreed_size
            )
            if probe is None:
                logger.error(f"  Could not re-run {fn} even at min size — skipping")
                continue

            _, agreed_size = probe
            logger.info(f"  Agreed size for {fn}: {agreed_size}px")

            # Now re-run every condition that doesn't match the agreed size
            for pk, current_size in sizes.items():
                if current_size == agreed_size:
                    continue   # already correct

                rec_idx = idx_maps[pk].get(fn)
                if rec_idx is None:
                    logger.warning(f"  {fn} not found in {pk} — skipping")
                    continue

                rec      = results_by_pk[pk][rec_idx]
                img_path = rec.get("img_path", "")
                prompt   = build_persona_prompt(persona_dicts.get(pk))

                logger.info(f"  Re-running {fn} for {pk}: {current_size}px → {agreed_size}px")
                result = _run_single_image(
                    prompt, img_path, model, processor, state, target_size=agreed_size
                )
                if result is None:
                    logger.error(f"  Re-run failed for {fn}/{pk} — keeping original")
                    continue

                fields, actual_size = result
                # Patch the record in-place
                results_by_pk[pk][rec_idx] = {
                    **rec,
                    **fields,
                    "image_max_size": actual_size,
                }
                update_counts[pk] += 1
                logger.info(f"  Updated {fn} for {pk} at {actual_size}px")

    finally:
        state.remove_hooks()

    # Write updated result files
    for pk, path in result_paths.items():
        if update_counts[pk] > 0:
            _save_results_inplace(Path(path), results_by_pk[pk], result_objs[pk])
            logger.info(f"Saved {pk}: {update_counts[pk]} records updated → {path}")

    total = sum(update_counts.values())
    logger.info(f"Reconciliation complete. Total records updated: {total}")
    return update_counts
