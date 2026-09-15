# repuap

Discover a linear representation direction in a vision-language model,
validate it's causally load-bearing, compress it into a pixel-space
universal adversarial perturbation (UAP), and test whether that
perturbation generalises to judgment tasks it was never trained on.

This is a general-purpose extraction of the 4-stage protocol developed in
the parent repo's study of `Llama-4-Scout-17B-16E-Instruct` as an evaluator
— nothing here is specific to that model, those personas, or those tasks.
It depends only on `torch`, `transformers`, `numpy`, `pandas`, `scipy`,
`scikit-learn`, and `Pillow`.

## The four stages

1. **Discover** (`repuap.discover`) — run a judgment task under two
   conditions (e.g. two personas, or two image sets) and compute a
   candidate direction per layer: a mean-difference vector, a linear-probe
   (CAV) direction, and an AUROC/alignment report to decide which layers'
   directions are worth keeping.
2. **Validate causally** (`repuap.causal`) — inject `alpha * direction` into
   the residual stream at a chosen layer and sweep `alpha` to confirm the
   direction actually moves the model's judgment (a dose-response curve),
   not just that it's decodable.
3. **Train a UAP** (`repuap.uap`) — gradient-match a single pixel-space
   delta (per L-infinity budget) that pushes any image's hidden state at
   that layer toward the validated direction, without touching activations
   or prompts at inference time.
4. **Test generalisation** (`repuap.generalise`) — apply the trained
   delta(s) to a *different* dataset and a *different* judgment task, paired
   per-sample against a clean baseline, and compute standard shift metrics
   (`condition_metrics`) and a UAP-budget-response (`budget_response`).

All four stages share one abstraction: `JudgmentTask` (`repuap.tasks`) — a
prompt builder, a JSON schema (which key holds the label), a valid-label
set, a score mapping, and a strict-JSON-first/regex-fallback parser. Define
your task once, use it everywhere.

## Model assumption

`repuap.hooks` auto-discovers `model.language_model.model.layers` and
`model.vision_model.{model,encoder,vision_model.encoder}.layers` — this
matches `AutoModelForImageTextToText` checkpoints with a similar structure
to Llama-4 / Llava / Qwen2-VL. If your model doesn't expose one of these
attribute paths, `repuap.hooks._inspect_model_structure` is the one place
that needs a small addition.

## Quickstart

```python
from repuap import JudgmentTask, load_vlm, discover_direction, validate_causally, train_uap, evaluate_generalisation

model, processor = load_vlm("your-model-repo-or-local-path")

# Define the judgment your positive/negative conditions differ on.
task = JudgmentTask(
    valid_labels=[1, 2, 3, 4, 5],
    scale_width=4.0,
    json_key="label",
    prompt_builder=lambda ctx: (
        "Rate this image's interestingness from 1 (not interesting) to 5 "
        '(extremely interesting).\n\nReturn only valid JSON: '
        '{"label": <1-5>, "explanation": "<one sentence>"}'
    ),
)

positive_images = [{"filename": "a.jpg", "img_path": "/path/a.jpg"}, ...]
negative_images = [{"filename": "b.jpg", "img_path": "/path/b.jpg"}, ...]

discovery = discover_direction(
    model, processor, task,
    positive_prompt="Imagine you are an easily-excited person. " + task.prompt_builder(None),
    negative_prompt="Imagine you are a bored, unimpressed person. " + task.prompt_builder(None),
    positive_images=positive_images, negative_images=negative_images,
)
layer_key = discovery["best_layer"]
vector = discovery["vectors"][layer_key]

dose_response = validate_causally(
    model, processor, task, prompt=task.prompt_builder(None),
    vector=vector, layer_key=layer_key, images=positive_images + negative_images,
)
print(dose_response["summary"])  # confirm mean score moves monotonically with alpha

deltas = train_uap(
    model, processor, task, prompt=task.prompt_builder(None),
    direction_vector=vector, layer_key=layer_key,
    train_images=[img["img_path"] for img in positive_images + negative_images],
    eval_images=[...],
    epsilons=[0.1, 0.5, 1.0, 2.0],
)

# A genuinely different downstream task, on a genuinely different dataset:
new_task = JudgmentTask(valid_labels=[0, 1, 2], scale_width=2.0, json_key="label",
                         prompt_builder=lambda row: f"...judge {row['image_id']}...")
import pandas as pd
manifest = pd.DataFrame([{"sample_id": "s0", "image_id": "i0", "img_path": "/path/new.jpg"}])

result = evaluate_generalisation(
    model, processor, new_task, manifest=manifest,
    uap_conditions=[(f"eps{eps}", deltas[eps]["delta"]) for eps in deltas],
)
print(result["summary"])
```

## What's intentionally NOT in this package

- Cluster/SLURM job scripts, multi-GPU memory-cap tuning, offline-cache
  environment variables — `load_vlm(..., max_memory=...)` exposes the one
  knob that matters; everything else is your deployment's concern.
- Any specific persona, contrast, or dataset definitions — those are
  research inputs, not pipeline code. Bring your own image lists / prompts.
- A non-HuggingFace model adapter layer — not built until there's a concrete
  need for one.

## Tests

```
pip install -e .[test]
pytest tests/
```

All tests run on CPU; no GPU or model download required. One test
(`test_reproduces_known_vector.py`) additionally validates against real data
from the parent study repo if it's present alongside this package (skipped
otherwise).
