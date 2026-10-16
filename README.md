# Task-Agnostic Adversarial Perturbations Through Internal Psychological-State Representations

MSc thesis project, **in progress**. This repository holds the research code and
the data exports behind the project website. It is not yet a finished or
end-to-end reproducible research release; `STATUS.md` states exactly what exists.

## Central question

Human judgement is systematically influenced by states such as emotion, arousal
and mental workload. A vision-language model does not need to experience such
states for representations associated with the same concepts to influence its
behaviour. The thesis asks:

> Can we identify an internal state associated with a psychological concept,
> establish that it causally influences evaluation, reach that state through the
> visual input alone, and thereby produce an attack that generalises beyond the
> task on which it was trained?

The eventual attack is a **universal adversarial perturbation (UAP)**: one fixed
image-space perturbation added to arbitrary images. It is trained to move the
model's internal representation along a previously identified and causally tested
direction, not to change a particular output label.

## Model and evaluation setting

- **Model:** `meta-llama/Llama-4-Scout-17B-16E-Instruct` (34 vision layers, 48 language layers), used as an evaluator.
- **Main task:** rate an image on a five-level interestingness scale (*Not Interesting* … *Extremely Interesting*), with a short explanation. A yes/no relevance task is used as a second in-distribution check for the UAPs.
- **Conditions:** the same images are judged under different persona prompts (gender × emotion, optionally extended with a country, or with a mental-workload level) and under a blank prompt without a persona.
- **Images:** a fixed 500-image subset of 1,000 Open Images V7 photographs. Transfer experiments use samples from Marqo-GS-10M, MEDIC and SMID. See `DATA.md`; no images are redistributed.
- **Activations:** residual-stream hidden states at every vision and language layer.

## Study structure

Each level tests a prerequisite for the next. A decodable representation is not
assumed to be a useful attack target.

| Level | Question | Code | Status |
|---|---|---|---|
| **1. Representation discovery** | Do persona and affect contrasts correspond to consistent linear directions, and at which layers? | `representation/` | available, preliminary |
| **2. Representation geometry / analysis** | How do those directions relate to each other? Are they shared, distinct, compositional? | `analytics/` | in progress |
| **3. Causal intervention** | Does adding a direction to the residual stream change the model's judgement? | `control/III1_vector_control/` | available, preliminary |
| **4. Universal adversarial perturbation and transfer** | Can a pixel-space perturbation target such a direction, and does its effect transfer to unseen datasets and tasks? | `attack/IV1_gradient_matching/`, `attack/IV2_generalisation/` | in progress |

## What is available and what is ongoing

**Available now** (saved result tables, exported to `web_export/`):

- layer-wise separability for 81 contrasts across all 82 layers
- condition similarity matrices at three layers, a nearest-neighbour grouping sweep, and interestingness–concept alignment by layer
- blank-prompt dose-response sweeps for five direction/layer pairs, plus single-dose and ablation checks
- UAP alignment, behavioural effect on the original tasks (two of three targets) and transfer to three unseen tasks, with bootstrap intervals

**Ongoing or missing:**

- several geometry results exist only as notebook output and are not yet saved as tables
- the UAP work is not finished: the workload UAP has no saved evaluation on the original tasks, budgets are large, and there are no random-perturbation baselines yet
- all experiments are single runs without seeds, repeats or held-out estimates of separability
- the model revision was not logged and the image ID list is not published

`STATUS.md` has the full table, including what each level still needs.

## Results so far

These are interim results. Numbers are taken from the exports in `web_export/data/`
unless marked as notebook output.

### Level 1: representation discovery

Persona- and affect-related contrasts are linearly separable in the language
stream. Layer ranges give the language layers at which the family-mean projection
AUC is at least 0.99.

| Contrast family | Contrasts | Peak mean AUC | Layers with mean AUC ≥ 0.99 | Peak cos(mean-diff, probe) |
|---|---:|---:|---|---:|
| Gender (emotion-matched pairs, 3 persona sets) | 24 | 1.00 | 23–24, 27–32 | 0.98 at layer 29 |
| Country (Germany or Nigeria vs base) | 32 | 1.00 | 6–7, 23–24, 26–28 | 0.97 at layer 27 |
| Emotion, one vs rest | 8 | 0.98 | none | 0.85 at layer 23 |
| Emotion vs contentment anchor | 7 | 1.00 | 22–33 | 0.98 at layer 23 |
| Mental workload (overwhelming vs minimal) | 1 | 1.00 | 23–36 | 0.98 at layer 24 |
| Interestingness (high- vs low-rated images) | 9 | 1.00 | 26–47 | 0.88 at layer 44 |

- **Persona concepts appear in the language stream, not the vision tower.** Vision-layer AUC stays at about 0.50 for gender, country, emotion-vs-contentment and workload. This is expected: those conditions are introduced through text.
- **Interestingness is partly visible in the vision tower** (family-mean AUC 0.73–0.86), consistent with image content carrying information about the judgement.
- **The contrast construction matters.** One-vs-rest emotion directions are noisier than pairwise contrasts against a contentment anchor.

Caveat: projection AUC is measured on the same activations the direction was
estimated from, in a 5,120-dimensional space, with between 27 and 1,998 samples
per class. Values at 1.00 are optimistic and are not held-out estimates.

### Level 2: geometry and analysis

From saved tables:

- **Decodability does not mean a concept organises the layer.** In a UMAP embedding of 48 persona-condition means, the nearest neighbour shares the country variant for 88% of conditions at `language_23`, a layer where emotion is strongly decodable. At `language_29` this drops to 25%, and 73% of nearest neighbours share emotion and gender.
- **Interestingness becomes more entangled with persona concepts at depth.** The mean absolute cosine between persona-specific interestingness directions and the matching emotion direction rises from 0.15 at `language_20` to 0.50 at `language_29` and above 0.8 from `language_39`.

From analysis notebooks only (`results/EX1_T1_*.ipynb`, `results/EX1_T2_additivity.ipynb`),
not yet saved as tables and not shown on the website:

- a gender direction from the base personas transfers to the country-extended sets (cosine about 0.95–0.99)
- the Germany and Nigeria country directions are distinct (cosine 0.675)
- summing separately measured gender, emotion and country vectors reconstructs the compound-persona activation in direction (cosine about 0.97) while overshooting its magnitude by about 14%
- the pooled interestingness direction is anti-aligned with the workload direction (cosine −0.36), although intervention at `language_24` moves ratings the other way (below)

### Level 3: causal intervention

Directions are added to the residual stream as `α · unit direction` under a blank
prompt, so the original persona manipulation is not a confound. Mean
interestingness rating (1–5) over 93 images:

| Direction | Layer | α = −8 | α = 0 | α = +8 | Pattern |
|---|---|---:|---:|---:|---|
| Interestingness | `language_29` | 1.98 | 2.51 | 3.10 | monotonic across all 9 strengths |
| Excited vs angry | `language_29` | 2.26 | 2.51 | 2.73 | monotonic |
| Workload | `language_24` | 2.30 | 2.51 | 2.95 | monotonic |
| Workload | `language_29` | 2.55 | 2.51 | 2.41 | weaker, opposite sign |
| Country (Nigeria) | `language_23` | 2.47 | 2.51 | 2.63 | small, concentrated at large α |

Gender at `language_29` was tested at α = ±2 and with ablation only; the means
stay between 2.48 and 2.55.

The interim reading is a **dissociation between decodability and causal
relevance**: gender and country are among the most cleanly separable directions
at level 1 but move the rating little, whereas interestingness, excited-vs-angry
and workload produce graded dose-response curves. The workload effect depends on
the layer.

Caveats: single runs, one behavioural task, standard error of each mean about
0.05, and no random-direction control yet.

### Level 4: universal adversarial perturbation and transfer

UAPs were trained for three targets (`interest`, `excited_vs_angry`, `workload`)
at `language_29`, plus a `workload` variant at `language_24`, on 400 images and
checked on 100 held-out images, for ε ∈ {0.1, 0.5, 1.0, 2.0}.

**Read the budgets with care.** ε is an L∞ bound in processor-normalised pixel
units, whose range is [−1, 1]. ε = 1.0 is half of the pixel range and ε = 2.0 is
all of it. These are not imperceptible perturbations.

- **Internal alignment** rises with ε for all four UAPs (mean cosine shift on held-out images between +0.05 and +0.18 at ε = 2.0); at ε = 0.1 it is about zero.
- **Original tasks.** For `interest` and `excited_vs_angry`, mean interestingness changes by −0.13 to +0.03 at ε ≤ 0.5, by about +0.5 at ε = 1.0, and every image is rated *Extremely Interesting* at ε = 2.0. The `workload` UAP moves the rating the other way: −0.11, −0.37, −0.63 and −0.36 at ε = 0.1, 0.5, 1.0 and 2.0 (100 held-out images). Part of that is the model reacting to visible corruption: 37% of its explanations at ε = 0.5 and 81% at ε = 1.0 describe the image as distorted or unclear. Without those images the shift is −0.21 at ε = 0.5 (n = 63) and +0.05 at ε = 1.0 (n = 19). The relevance task has not been run for `workload`.
- **Transfer** to shopping relevance (Marqo-GS-10M, n = 300), moral evaluation (SMID, n = 274) and damage severity (MEDIC, n = 300), without retraining: shifts are close to zero at ε = 0.1 and grow with the budget. The workload UAP gives the most consistently ordered response. At ε = 2.0 the outputs are dominated by attack-specific content and no longer describe the image, so this budget is a different regime and not a stronger version of the graded effect.
- An exploratory analysis on SMID finds an arousal × ε interaction for all three UAPs, which does not survive a control for image valence (see below).

#### What the transfer result supports

Three statements of different strength. Only the first is established.

- **A. Measured.** UAPs trained only against internal state directions shift judgements on three unseen datasets and tasks without retraining. At ε = 0.5 the workload UAP shifts all three tasks, the interest UAP one, and the excited-vs-angry UAP none; at ε = 1.0 all three UAPs shift all three tasks.
- **B. Interpretation under test.** The UAPs act on a shared affective state representation and not on task labels. Whether that state is closer to valence or to arousal is open; a valence probe is the main comparison in the planned test. The evidence so far is mixed (tables below).
- **C. Not claimed.** That this representation is upstream in the processing chain: nothing here measures processing order. Nor that affective directions transfer better than a label direction in general: `excited_vs_angry` is affective and shows no advantage over `interest`.

**Target directions** (`web_export/data/geometry/uap_target_direction_cosines.csv`).
Cosine similarity at `language_29_D5120`. The interest UAP targets the
blank-prompt interestingness direction.

| | Interest target | Excited-vs-angry target | Workload target |
|---|---|---|---|
| Interest target (blank prompt) | 1 | +0.402 | −0.050 |
| Excited-vs-angry target | +0.402 | 1 | −0.500 |
| Workload target | −0.050 | −0.500 | 1 |
| *Baseline:* pooled interestingness (not a UAP target) | +0.739 | +0.834 | −0.360 |
| *Baseline:* gender | +0.016 | +0.040 | +0.036 |
| *Baseline:* country, Germany | −0.069 | −0.172 | +0.123 |
| *Baseline:* country, Nigeria | −0.008 | −0.236 | +0.324 |
| *Baseline:* random direction, 95% range | ±0.027 | ±0.027 | ±0.027 |

The three targets are not one direction. Interest and excited-vs-angry are
moderately aligned, workload is unrelated to interest (within the range of the
gender baseline) and opposed to excited-vs-angry. A common axis that all three
load on is not excluded by this table and is what the probe test addresses.

**Transfer at ε = 0.5** (`web_export/data/attack/transfer_shift_statistics.csv`).
Mean signed label shift as a fraction of the task's scale, with a 95% bootstrap
interval over images. "Direction" is the share of changed labels that moved the
dominant way; 50% means changes go both ways equally.

| Task | UAP | Signed shift [95% CI] | Labels changed | Direction |
|---|---|---|---|---|
| Shopping relevance (n = 300) | interest | −0.023 [−0.050, +0.004] | 31% | 55% down |
| | excited-vs-angry | −0.011 [−0.037, +0.015] | 32% | 53% down |
| | workload | **−0.064 [−0.092, −0.037]** | 34% | 69% down |
| Moral evaluation (n = 274) | interest | **+0.043 [+0.009, +0.077]** | 40% | 61% up |
| | excited-vs-angry | +0.013 [−0.021, +0.047] | 39% | 51% up |
| | workload | **−0.081 [−0.115, −0.048]** | 43% | 72% down |
| Damage severity (n = 300) | interest | +0.012 [−0.027, +0.048] | 24% | 54% up |
| | excited-vs-angry | +0.040 [+0.000, +0.082] | 27% | 60% up |
| | workload | **+0.185 [+0.138, +0.235]** | 42% | 78% up |

Bold: interval excludes zero. At ε = 1.0 the interval excludes zero for every
UAP on every task (same file). Caveats that belong with this table:

- **No noise baseline on the transfer tasks yet.** Part of any shift may be generic image corruption. Magnitude-matched noise runs are queued.
- **Damage severity is partly an artefact for the workload UAP.** In 33% of its ε = 0.5 explanations the model describes the image itself as degraded, distorted or overlaid with text. Without those images the shift is +0.010 [−0.028, +0.050]. On shopping relevance (0.3% such explanations) and moral evaluation (8%) the workload shift is unchanged when they are removed (−0.064 and −0.081).
- **Reachability.** At ε = 0.5 the workload UAP moved hidden states further toward its target than the others (cosine gain 0.035, against 0.019 for excited-vs-angry and 0.006 for interest; `uap_alignment.csv`). Its earlier onset may reflect that.
- **One budget step.** There is no budget between 0.1 and 0.5 or between 0.5 and 1.0.
- **One training run per UAP**, no seeds.
- **Per image, the workload UAP's larger absolute shift over the interest UAP is clear on damage severity only** (+0.137 [+0.093, +0.182]); on shopping relevance (+0.014 [−0.010, +0.038]) and moral evaluation (+0.013 [−0.018, +0.045]) the interval includes zero (`transfer_paired_contrasts.csv`).

**Arousal or valence on SMID** (`web_export/data/attack/moral_arousal_valence_regression.csv`).
Regression of the morality shift on ε and the clean label, with human arousal or
valence ratings of the image and their interaction with ε added. Intervals
resample whole images, so the four budgets of one image stay together.

| UAP | R² from ε and clean label | Added by arousal terms | Added by valence terms | Arousal × ε alone [95% CI] | Arousal × ε with valence in the model [95% CI] |
|---|---|---|---|---|---|
| interest | 0.328 | +0.019 | +0.074 | +0.261 [+0.148, +0.372] | −0.034 [−0.135, +0.064] |
| excited-vs-angry | 0.314 | +0.027 | +0.088 | +0.310 [+0.207, +0.408] | +0.038 [−0.052, +0.126] |
| workload | 0.582 | +0.010 | +0.064 | +0.228 [+0.117, +0.330] | −0.070 [−0.165, +0.017] |

Most of the explained variance comes from ε and the clean label. The arousal × ε
interaction is present on its own and disappears once valence is included, while
the valence × ε coefficient stays clearly negative (−0.44 to −0.48). Arousal and
valence correlate at −0.47 in this sample. The analysis is post hoc, has no
noise baseline, and the model has no clean-label × ε term, so a scale-ceiling
effect is not excluded. It does not single out arousal.

**Planned tests of B** (jobs submitted; results will be added here):

| Test | What it measures | Status |
|---|---|---|
| Probe alignment | Linear probes for human arousal and valence on `language_29` activations of the clean SMID images, cross-validated; cosine of each probe direction with the three UAP targets against a random-direction null | activations queued |
| Movement along the probe axes | Shift of perturbed images along the arousal and valence probe directions per UAP and ε, and whether it predicts the label shift | activations queued |
| Noise baseline | Magnitude-matched noise (3 seeds, ε = 0.5 and 1.0) through the full transfer evaluation, and the SMID regression under noise | queued |
| Workload UAP on the original tasks | Whether the stress direction moves interestingness opposite to the interest UAP | interestingness done (see "Original tasks" above): opposite sign, confounded with perceived image corruption; relevance queued |

Not yet done: any behavioural evaluation of the `language_24` variant, budgets
between the four used here, repeated training runs, and the second transfer
level (amplitude scaling).

## Website data

The website does not read experiment folders and contains no hand-entered
numbers. Its figures are drawn from versioned exports:

```
results/  (local, not in git)  →  scripts/export_web_data.py  →  web_export/data/*.csv|json
                                                               →  web_export/release_manifest.json
```

- Each export carries its source files, model and configuration, feature and task labels, units, aggregation method, a `preliminary`/`final` flag and caveats.
- `web_export/release_manifest.json` records the generation date, code commit, model configuration and a SHA-256 for every source file.
- Regenerate with `python scripts/export_web_data.py` on a machine that holds `results/`.

Details are in `web_export/README.md`.

## Repository structure

- `representation/`: activation extraction, mean-difference vectors, linear probes, vector evaluation
- `analytics/`: geometric and compositional analyses
- `control/III1_vector_control/`: residual-stream interventions and logit-lens checks
- `attack/IV1_gradient_matching/`: UAP optimisation against internal directions
- `attack/IV2_generalisation/`: dataset preparation and evaluation on unseen tasks
- `runners/`, `attack/runners/`: experiment entry points
- `package/repuap/`: an early extraction of the four-stage protocol into an installable package (lightly tested)
- `scripts/export_web_data.py`, `web_export/`: website data pipeline
- `utils/`: model loading, hooks, prompts, paths

The code reflects how the project developed. Early components were written for
exploratory experiments and still assume the original data layout and a multi-GPU
SLURM environment.

## Setup

```bash
conda env create -f env.yml
conda activate vision-transformers
```

Paths resolve relative to the repository root. The model is expected under
`hpc_infrastructure/hf_cache/`; set `VT_MODEL_PATH` (checkpoint directory) or
`VT_HF_CACHE` to use another location. Runners are started from the repository
root, for example `python runners/run_interest_dose_response.py --help`.

Running the experiments needs gated model access, the datasets in `DATA.md` and
several large GPUs. There is no tested end-to-end reproduction workflow yet.

## References

- Turner et al., **ActAdd: Steering Language Models Without Optimization**, arXiv:2308.10248
- Kim et al., **TCAV: Interpretability Beyond Feature Attribution**, arXiv:1711.11279
- Zou et al., **Representation Engineering: A Top-Down Approach to AI Transparency**, arXiv:2310.01405
- Tigges et al., **Language Models Linearly Represent Sentiment**, BlackboxNLP @ ACL 2024
- Park et al., **The Linear Representation Hypothesis and the Geometry of Large Language Models**, arXiv:2311.03658
- Rimsky et al., **Steering Llama 2 via Contrastive Activation Addition**, arXiv:2312.06681
- Chen et al. (Anthropic), **Persona Vectors: Monitoring and Controlling Character Traits in Language Models**, arXiv:2507.21509
- nostalgebraist, **interpreting GPT: the logit lens**, LessWrong 2020
- **Activation Scaling for Steering and Interpreting Language Models**, EMNLP 2024 Findings
- Kriegeskorte et al., **Representational Similarity Analysis**, Frontiers in Systems Neuroscience 2008
- Kornblith et al., **Similarity of Neural Network Representations Revisited (CKA)**, arXiv:1905.00414
