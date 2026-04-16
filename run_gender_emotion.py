#!/usr/bin/env python
"""
run_gender_emotion.py  —  Gender × Emotion Experiment
======================================================
Goal
----
Isolate the impact of (a) gender and (b) emotion on interestingness ratings,
using a deliberately under-specified "base" persona that minimises confounds.

Base persona design
-------------------
Field           Value                   Rationale
-----------     --------------------    -----------------------------------------
age range       30-45                   Prototypical employed adult; avoids
                                        youth (15-19) and older-worker effects.
gender          female / male           The variable under test.
employment      full-time employed      Single cross-demographic anchor; no
                                        industry given to avoid sector priors.
mental workload moderate                True midpoint of [low, moderate, high,
                                        extreme] in the original persona data.
emotion         anger / fear / …        Primary variable under study.

Country and continent are omitted entirely — no "neutral" country exists, and
any named country introduces cultural priors.  Both genders receive the same
omission, so it does not confound the gender comparison.

Known caveats
-------------
* Model's implicit cultural default is likely Western/American when origin is
  unspecified.  This is consistent across conditions and does not confound
  the within-experiment comparisons.
* "Full-time employed" implicitly excludes students, caregivers, retirees.
  Intentional controlled choice — document in paper.
* The rating scale still contains "Extremely Interesting"; this cannot be
  changed without breaking cross-experiment comparability, but it equally
  affects all conditions here.

Experimental design & run order
---------------------------------
Step 1 (this script, --personas all):
  Female × all 8 emotions  +  Male × anger
  → Compare female_anger vs male_anger.

Step 2 (decision):
  If female_anger == male_anger → use female-only results for all emotions.
  If female_anger != male_anger → run remaining 7 male emotions separately.

Results
-------
  data/experiments/gender_emotion/results_female_{emotion}.npy
  data/experiments/gender_emotion/results_male_anger.npy

Running in parallel with synonym test
--------------------------------------
  # Terminal / SLURM job A
  OFFLOAD_DIR=./offload_dir_synonym  python run_synonym_test.py   --model_path /path/to/model

  # Terminal / SLURM job B
  OFFLOAD_DIR=./offload_dir_gender   python run_gender_emotion.py --model_path /path/to/model

Each process loads the model independently; results land in separate files so
there is no write conflict.  Ensure OFFLOAD_DIR differs between jobs.

Usage
-----
  # Run everything (female all + male anger):
  OFFLOAD_DIR=./offload_dir_gender python run_gender_emotion.py --model_path /path/to/model

  # Run only the female personas (e.g. to start earlier on a single GPU):
  python run_gender_emotion.py --model_path /path/to/model --personas female

  # Run only male anger (on a second GPU in parallel):
  OFFLOAD_DIR=./offload_dir_gender_male python run_gender_emotion.py \\
      --model_path /path/to/model --personas male
"""
import os, argparse
from get_activations_base import run_experiment

# Emotion list matches the values present in df_generated-personas-sample.pkl

EMOTIONS = [
    "anger",        # first — needed for the male comparison; run this first
    "fear",
    "disgust",
    "sad",
    "amusement",
    "awe",
    "contentment",
    "excitement",
]

# ---------------------------------------------------------------------------
# Base persona template — fields shared across all conditions
# ---------------------------------------------------------------------------
_BASE = {
    "age range":        "30-45",
    "employment":       "full-time employed",
    "mental workload":  "moderate",
    # gender and emotion are filled per-condition below
}

FEMALE_PERSONAS = [
    {
        "persona_key": f"female_{emotion}",
        **_BASE,
        "gender":  "female",
        "emotion": emotion,
    }
    for emotion in EMOTIONS
]

MALE_PERSONAS = [
    {
        "persona_key": "male_anger",
        **_BASE,
        "gender":  "male",
        "emotion": "anger",
    },
]

ALL_PERSONAS = FEMALE_PERSONAS + MALE_PERSONAS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gender × emotion experiment with base persona"
    )
    parser.add_argument("--model_path", required=True,
                        help="Local path to the model directory")
    parser.add_argument(
        "--personas",
        choices=["all", "female", "male"],
        default="all",
        help=(
            "Subset to run.  "
            "'all' = female × 8 emotions + male × anger (default).  "
            "'female' = female conditions only.  "
            "'male' = male anger only (use a separate OFFLOAD_DIR when running "
            "in parallel with 'female')."
        ),
    )
    args = parser.parse_args()

    personas_to_run = {
        "all":    ALL_PERSONAS,
        "female": FEMALE_PERSONAS,
        "male":   MALE_PERSONAS,
    }[args.personas]

    run_experiment(
        model_path    = args.model_path,
        offload_dir   = os.environ.get("OFFLOAD_DIR", "./offload_dir_gender_emotion"),
        output_dir    = "data/experiments/gender_emotion",
        personas      = personas_to_run,
        manifest_path = "data/selected_uniform_500_manifest.pkl",
    )
