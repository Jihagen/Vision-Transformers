#!/usr/bin/env python
"""
run_synonym_test.py  —  Synonym / Confounder Test
==================================================
Hypothesis
----------
The word "extreme" appears in both the persona's mental_workload field
("extreme") and in the rating scale label ("Extremely Interesting").  The
unexpected result — extreme-workload personas rating images as "Extremely
Interesting" rather than showing disengagement — may partly be driven by
token-level co-activation rather than genuine reasoning.

Test
----
Re-run personas 7319 (extreme_positive) and 5504 (extreme_negative) with
"overwhelming" substituted for "extreme" in the mental_workload description.
Everything else is kept byte-for-byte identical to the original runs so that
the synonym is the only variable.

Why "overwhelming"
------------------
- Semantically equivalent to "extreme" in the cognitive-load literature
  (overwhelmed = cannot process any more input).
- Has no lexical overlap with "Extremely Interesting"; the two concepts are
  semantically distant (negative burden vs. positive engagement).
- A single common adjective — the model has strong priors for it.

Alternatives considered: "severe" (medical connotations), "maximal"
(too technical), "very high" (phrase, not adjective → changes structure).

Results
-------
  data/experiments/synonym_test/results_7319_synonym.npy
  data/experiments/synonym_test/results_5504_synonym.npy

Usage
-----
  OFFLOAD_DIR=./offload_dir_synonym \\
  python run_synonym_test.py --model_path /path/to/model
"""
import os, argparse
from get_activations_base import run_experiment

# ---------------------------------------------------------------------------
# Persona definitions
# Each dict mirrors the original persona row from df_generated-personas-sample.pkl
# with a single field changed: mental_workload "extreme" → "overwhelming".
# _format="raw" keeps the original comma-separated style (no key labels) so
# that the only textual difference from the original run is the one word.
# ---------------------------------------------------------------------------
EXPERIMENT_PERSONAS = [
    # Persona 7319 — originally "extreme_positive" (extreme workload + excitement)
    {
        "persona_key":     "7319_synonym",
        "age":             "15-19",
        "gender":          "male",
        "country":         "Greece",
        "continent":       "Europe",
        "job_branch":      "Retail and Customer Service",
        "mental_workload": "overwhelming",   # original: "extreme"
        "emotion":         "excitement",
        "_format":         "raw",
    },
    # Persona 5504 — originally "extreme_negative" (extreme workload + anger)
    {
        "persona_key":     "5504_synonym",
        "age":             "25-29",
        "gender":          "male",
        "country":         "Hungary",
        "continent":       "Europe",
        "job_branch":      "Law and Public Service",
        "mental_workload": "overwhelming",   # original: "extreme"
        "emotion":         "anger",
        "_format":         "raw",
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synonym confounder test: re-run extreme personas with 'overwhelming'"
    )
    parser.add_argument("--model_path", required=True,
                        help="Local path to the model directory")
    args = parser.parse_args()

    run_experiment(
        model_path    = args.model_path,
        offload_dir   = os.environ.get("OFFLOAD_DIR", "./offload_dir_synonym"),
        output_dir    = "data/experiments/synonym_test",
        personas      = EXPERIMENT_PERSONAS,
        manifest_path = "data/selected_uniform_500_manifest.pkl",
    )
