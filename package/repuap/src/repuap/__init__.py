"""
repuap: discover a linear representation direction in a vision-language
model, validate it's causally load-bearing, compress it into a pixel-space
universal adversarial perturbation, and test whether it generalises to
unseen tasks/datasets.

Four stages, one shared JudgmentTask abstraction:

    model, processor = load_vlm("your-model-repo-or-path")

    task = JudgmentTask(valid_labels=[1,2,3,4,5], scale_width=4.0,
                         prompt_builder=lambda ctx: "...", json_key="label")

    discovery = discover_direction(model, processor, task,
        positive_prompt=..., negative_prompt=...,
        positive_images=[...], negative_images=[...])

    dose_response = validate_causally(model, processor, task,
        prompt=..., vector=discovery["vectors"][layer_key], layer_key=layer_key,
        images=[...])

    deltas = train_uap(model, processor, task,
        prompt=..., direction_vector=discovery["vectors"][layer_key], layer_key=layer_key,
        train_images=[...], eval_images=[...])

    new_task = JudgmentTask(...)  # a genuinely different downstream judgment
    generalisation = evaluate_generalisation(model, processor, new_task,
        manifest=my_dataframe,
        uap_conditions=[("eps1.0", deltas[1.0]["delta"])])

See README.md for the full walkthrough and the constraints each stage
assumes (HF `AutoModelForImageTextToText`-style model structure; see
hooks.py's module docstring).
"""

from .model_io import load_vlm, preprocess_image
from .tasks import JudgmentTask, ParsedResponse, parse_response, extract_last_json_block
from .hooks import HookState, register_extract_hooks, register_inject_hooks, register_projection_removal_hooks
from . import discover, causal, uap, generalise
from .pipeline import discover_direction, validate_causally, train_uap, evaluate_generalisation

__all__ = [
    "load_vlm", "preprocess_image",
    "JudgmentTask", "ParsedResponse", "parse_response", "extract_last_json_block",
    "HookState", "register_extract_hooks", "register_inject_hooks", "register_projection_removal_hooks",
    "discover", "causal", "uap", "generalise",
    "discover_direction", "validate_causally", "train_uap", "evaluate_generalisation",
]

__version__ = "0.1.0"
