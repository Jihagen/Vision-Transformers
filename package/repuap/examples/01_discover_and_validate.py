"""
Example: Stages I + III -- discover a candidate direction for a contrast of
your choosing, then validate it's causally load-bearing.

This is illustrative, not a script meant to be run as-is: fill in
MODEL_PATH and the image lists for your own model/dataset. Requires a GPU
for any model of meaningful size.
"""
from repuap import JudgmentTask, load_vlm, discover_direction, validate_causally

MODEL_PATH = "your-model-repo-or-local-path"

# A judgment task with a 1-5 numeric label under the JSON key "label".
task = JudgmentTask(
    valid_labels=[1, 2, 3, 4, 5],
    scale_width=4.0,
    json_key="label",
    prompt_builder=lambda ctx: (
        "Rate this image's interestingness from 1 to 5.\n\n"
        'Return only valid JSON: {"label": <1-5>, "explanation": "<one sentence>"}'
    ),
)

# Two prompt conditions defining the contrast (here: two personas over the
# SAME task/images). You could instead hold the prompt fixed and vary the
# image sets -- discover_direction accepts either.
positive_prompt = "Imagine you are an easily-excited, curious person.\n\n" + task.prompt_builder(None)
negative_prompt = "Imagine you are a bored, unimpressed person.\n\n" + task.prompt_builder(None)

# [{"filename": ..., "img_path": ...}, ...] -- fill in your own image manifest.
positive_images = []
negative_images = []


def main():
    model, processor = load_vlm(MODEL_PATH, max_memory=None)

    discovery = discover_direction(
        model, processor, task,
        positive_prompt=positive_prompt, negative_prompt=negative_prompt,
        positive_images=positive_images, negative_images=negative_images,
    )
    print(f"Layers evaluated: {len(discovery['reports'])}")
    print(f"Recommended layer: {discovery['best_layer']}")
    for layer_key, report in sorted(discovery["reports"].items(), key=lambda kv: -kv[1].projection_auc)[:5]:
        print(f"  {layer_key}: AUC={report.projection_auc:.3f} "
              f"probe_acc={report.probe_accuracy_cv:.3f} keep={report.keep}")

    layer_key = discovery["best_layer"]
    if layer_key is None:
        raise RuntimeError("No layer passed the AUC/probe thresholds -- try more images, "
                            "a stronger contrast, or lower discover_direction's thresholds.")
    vector = discovery["vectors"][layer_key]

    dose_response = validate_causally(
        model, processor, task, prompt=task.prompt_builder(None),
        vector=vector, layer_key=layer_key,
        images=positive_images + negative_images,
        alphas=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    )
    print("\nDose-response summary (mean score should move monotonically with alpha):")
    print(dose_response["summary"])


if __name__ == "__main__":
    main()
