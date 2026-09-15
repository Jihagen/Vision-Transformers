"""
Example: Stages IV.1 + IV.2 -- train a universal pixel-space perturbation
against a validated direction, then test whether it transfers to a
completely different judgment task and dataset.

Illustrative, not meant to be run as-is: fill in MODEL_PATH, `vector`/
`layer_key` (typically the output of 01_discover_and_validate.py), and your
own train/eval image paths + new-task manifest. Requires a GPU.
"""
import numpy as np
import pandas as pd

from repuap import JudgmentTask, load_vlm, train_uap, evaluate_generalisation

MODEL_PATH = "your-model-repo-or-local-path"

# The task the UAP is trained against.
source_task = JudgmentTask(
    valid_labels=[1, 2, 3, 4, 5], scale_width=4.0, json_key="label",
    prompt_builder=lambda ctx: (
        "Rate this image's interestingness from 1 to 5.\n\n"
        'Return only valid JSON: {"label": <1-5>, "explanation": "<one sentence>"}'
    ),
)

# From 01_discover_and_validate.py's discover_direction() output.
layer_key = "language_29_D5120"
vector = np.zeros(5120, dtype=np.float32)  # placeholder -- use a real discovered vector

train_image_paths: list[str] = []   # fill in
eval_image_paths: list[str] = []    # fill in

# A genuinely different downstream judgment, on a genuinely different dataset --
# this is the generalisation test. manifest needs at least sample_id, image_id, img_path.
target_task = JudgmentTask(
    valid_labels=[0, 1, 2], scale_width=2.0, json_key="label",
    prompt_builder=lambda row: (
        f"Assess the severity of visible damage for item {row['image_id']}.\n\n"
        'Return only valid JSON: {"label": <0-2>, "explanation": "<one sentence>"}'
    ),
)
manifest = pd.DataFrame(columns=["sample_id", "image_id", "img_path"])


def main():
    model, processor = load_vlm(MODEL_PATH, max_memory=None)

    deltas = train_uap(
        model, processor, source_task,
        prompt=source_task.prompt_builder(None),
        direction_vector=vector, layer_key=layer_key,
        train_images=train_image_paths, eval_images=eval_image_paths,
        epsilons=[0.1, 0.5, 1.0, 2.0], n_epochs=20, max_side=336,
        out_dir="uap_deltas",
    )
    for eps, result in deltas.items():
        print(f"eps={eps}: final train cos={result['cos_history'][-1]:.4f}")

    uap_conditions = [(f"eps{eps:.2f}", result["delta"]) for eps, result in deltas.items()]
    generalisation = evaluate_generalisation(
        model, processor, target_task, manifest=manifest, uap_conditions=uap_conditions,
    )
    print("\nGeneralisation summary:")
    print(generalisation["summary"])


if __name__ == "__main__":
    main()
