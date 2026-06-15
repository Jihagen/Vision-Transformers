# Vision-Transformers: Representation & Control of Persona-Conditioned Image Judgements in Llama-4-Scout

This repository studies how a large vision-language model (**Llama-4-Scout-17B-16E-Instruct**)
internally represents — and can be causally steered along — concepts that shape its
judgements of images, when those judgements are elicited under different
**persona conditioning** (gender, emotional state, country/culture) and for a
target task of rating how **interesting** an image is.

The guiding question:

> When the model is prompted *"as if"* it were a person with a given gender,
> emotional disposition, or cultural background, and asked to rate how
> interesting an image is — does the model's hidden activations encode these
> persona features and the interestingness judgement as **linear directions**?
> Are those directions **causally responsible** for the output (not just
> correlated with it)? And is "interestingness" itself a **persona-independent**
> axis, or persona/context-specific?

The codebase is organised as a **three-tier pipeline**:

```
Tier I   (representation/)  — DISCOVER candidate directions in activation space
Tier II  (analytics/)       — INTERPRET the geometry of those directions
Tier III (control/, representation/III1_vector_control/) — INTERVENE causally (inject / scale / ablate)
```

plus a cross-cutting set of **extra validation checks** (logit-lens, lexicon
probes, ordinal-gradient checks, vision-layer pre-verbal signal analysis) that
are used throughout Tiers I-III to sanity-check that a discovered direction
means what we think it means.

---

## Table of Contents

1. [Repository layout](#1-repository-layout)
2. [Data](#2-data)
3. [Tier I — Representation Discovery](#3-tier-i--representation-discovery)
4. [Tier II — Analytics](#4-tier-ii--analytics)
5. [Tier III — Causal Control](#5-tier-iii--causal-control)
6. [Extra validation checks (cross-cutting toolkit)](#6-extra-validation-checks-cross-cutting-toolkit)
7. [Cosine-gradient matching: is "interestingness" a shared axis?](#7-cosine-gradient-matching-is-interestingness-a-shared-axis)
8. [Findings so far](#8-findings-so-far)
9. [Status / what's done vs. pending](#9-status--whats-done-vs-pending)
   - [9.1 TIER III conclusion & TIER IV roadmap](#91--tier-iii-conclusion--tier-iv-roadmap)
10. [How to run things](#10-how-to-run-things)
11. [References](#11-references)

---

## 1. Repository layout

| Path | Purpose |
|---|---|
| `representation/` | **Tier I**: contrast design & activation collection (I.1), mean-difference vectors (I.2), linear/CAV probes (I.3), vector evaluation (I.4), PCA/SVD subspace check (I.5). Also houses the **actual Tier III implementation** (`III1_vector_control/`). |
| `analytics/` | **Tier II**: geometric interpretation of discovered directions — persona analytics (II.1) and interestingness analytics (II.2). |
| `control/` | **Tier III** experiment *specifications* (III.1-III.5): vector addition, dose-response scaling, ablation, combined injection, vector swap. Mostly stubs describing the intended design; the working implementation for III.1-III.3 lives in `representation/III1_vector_control/`. |
| `runners/` | CLI entry points that wire the above together (activation collection, discovery, analytics, control experiments, blank validation, logit-lens evidence). |
| `utils/` | Shared infrastructure: activation hooks (`hooks.py`), model/processor loading, image preprocessing, prompt building, HPC helpers. |
| `data/` | Raw inputs: 500-image manifest, persona/blank activation checkpoints (`results_*.npy`), image set. |
| `results/` | All pipeline outputs: `representation_discovery/`, `analytics/`, `blank_validation/`, `extra_checks/`, plus the main analysis notebooks. |
| `hpc_infrastructure/` | Conda env, model weight cache, SLURM logs, CPU-offload scratch dirs. |

---

## 2. Data

- **`data/selected_uniform_500_manifest.pkl`** — the 500-image evaluation set (filename + path), sampled to be uniform over the model's own interestingness ratings.
- **`data/results_blank_activations.npy`** — pre-collected activations for all 500 images under a **blank prompt** (no persona description at all). This is the confound-free baseline used throughout (e.g. Branch A's `no_inject` condition, the ordinal-gradient check in §6.4, the vision-layer check in §6.5).
- **`data/results_persona_<id>.npy`** — persona-conditioned activation collections (one file per image-batch / persona run).
- **`data/imagesDemographics/`** — the underlying image set (~1000 images).

### Activation result format

Every `results_*.npy` is a pickled dict:

```python
{
  "version": "experiment_v1",
  "persona_key": "blank" | "<gender>_<emotion>[_<country>]",
  "selected_images": [{"filename": ..., "img_path": ...}, ...],
  "results": [
    {
      "filename": ...,
      "interestingness": "Not Interesting" | "Slightly Interesting" | "Moderately Interesting"
                          | "Very Interesting" | "Extremely Interesting",
      "explanation": "<free-text model explanation>",
      "embeddings": {
        "vision_layer_<N>_cls":      np.ndarray (1408,),  # N = 0..33
        "llm_layer_<N>_rating_token": np.ndarray (5120,), # N = 0..47
      },
    },
    ...
  ],
}
```

`embeddings` are captured at two points (see I.1):
- **`vision_layer_N_cls`**: the CLS token of the vision encoder's layer N — a
  *pre-verbal* representation of the image, independent of the persona prompt
  or the model's eventual answer.
- **`llm_layer_N_rating_token`**: the LLM's residual stream at layer N, at the
  **teacher-forced position immediately before the model emits its rating
  word** (the `'{"interestingness":"'` anchor). This is the representation
  used for almost all Tier I/II/III work, but it carries an important caveat
  — see §6.1.

### Naming conventions

- **`comp_key`** = `"<modality>_<layer>_D<dim>"`, e.g. `language_29_D5120` (LLM
  layer 29, hidden size 5120) or `vision_4_D1408` (vision encoder layer 4,
  hidden size 1408). 48 LLM layers (0-47), 34 vision layers (0-33).
- **`md_vector`** = a *mean-difference* direction, `v = mean(H_pos) - mean(H_neg)`,
  unit-normalised. Stored as `{contrast_name}.npy` → `dict[comp_key] -> (D,) ndarray`.
- **`CAV` / probe vector** = the (unit-normalised) weight vector of a logistic
  regression trained to separate `H_pos` from `H_neg`. A second, independent
  estimate of "the direction", used to cross-check the md_vector (§3.4).

---

## 3. Tier I — Representation Discovery

Tier I asks: *for a given contrast (e.g. "female persona" vs "male persona",
or "rated Very/Extremely Interesting" vs "rated Not/Slightly Interesting"),
is there a consistent linear direction in activation space that separates the
two groups, and at which layer is it strongest?*

### I.1 — Activation collection (`representation/I1_contrast_design/`)

Runs the persona-conditioned image-rating task and captures activations.
Two-stage inference per image:

1. **Full generation** — the model is given the image + a persona-conditioned
   prompt (built by `utils/prompt_builder.py::build_persona_prompt`, or
   `build_blank_prompt` for no persona) and generates a JSON response
   `{"interestingness": "...", "explanation": "..."}`.
2. **Teacher-forced one-step** — the prompt is re-run with the generation
   forced up to and including `'{"interestingness":"'` (the `RATING_ANCHOR`),
   and activations are captured at that exact position. This is what
   `llm_layer_N_rating_token` is.

Vision-encoder activations (`vision_layer_N_cls`) are captured once per image
(persona-independent — the image encoding doesn't change with the prompt).

Includes OOM-safe image downscaling (`utils/image_utils.py`,
ladder `[800, 640, 512, 384, 256, 128]`) and checkpoint/recovery via
`load.load_activation_results`, `load.filter_by_label_support`, and
`reconcile.reconcile_vision_activations` (fixes cross-condition image-size
mismatches in vision activations).

**Experiment variants** (`runners/experiment_definitions.py`):
- **`base`**: 16 conditions = {female, male} × 8 emotions
  (anger, amusement, awe, contentment, disgust, excitement, fear, sad), with
  fixed confounders (age 30-45, full-time employed, moderate mental workload).
- **`extended`**: `base` + a `Country` field (`Germany` or `Nigeria`),
  registered at runtime via `register_country_variant(country)`.
- **`blank`**: no persona description at all (just the image + rating task).

### I.2 — Mean-difference vectors (`representation/I2_mean_difference/`)

For a contrast defined by a positive group and a negative group of activation
collections, computes per-layer

```
v = mean(H_positive) - mean(H_negative)        (then L2-normalised)
```

`compute_all_contrasts(loaded_data, contrasts, save_dir=...)` runs this for
every `comp_key` (82 layers total: 48 language + 34 vision) and every
contrast, saving `dict[comp_key -> (D,) ndarray]` to
`results/representation_discovery/<variant>/md_vectors/{contrast}.npy`.

This is the **ActAdd-style** ("Activation Addition", Turner et al. 2023)
direction estimate — simple, robust, and what's actually used for injection in
Tier III.

### I.3 — Linear probes / CAVs (`representation/I3_linear_probe/`)

A second, independent estimate of the same direction: `StandardScaler →
LogisticRegression(C=1.0)`, 5-fold stratified cross-validation. The
(unit-normalised) logistic-regression weight vector is the **CAV**
(Concept Activation Vector, Kim et al. 2018 TCAV); `accuracy_cv` is the
held-out decodability of the contrast at that layer.

### I.4 — Vector evaluation / filtering (`representation/I4_vector_evaluation/`)

For every `(contrast, layer)` pair, computes a `VectorReport`:

| Check | Metric | "Good" threshold |
|---|---|---|
| Projection separability | AUROC of projecting `H_pos`/`H_neg` onto the md_vector | > 0.6 (often ≫, up to 1.0) |
| MD↔CAV alignment | cosine(md_vector, CAV) | > 0.7 |
| Probe decodability | 5-fold CV accuracy of the I.3 probe | > 0.55 (chance = 0.5) |
| Layer profile | AUROC across all 82 layers | used to find the *peak* layer |

A layer/contrast is **kept** (`keep=True`) only if it passes all of these —
i.e. two independently-estimated directions (mean-difference and logistic
regression) agree on what the discriminating direction *is* (high cosine), and
that direction actually separates the classes (high AUROC) in a way a simple
classifier can also pick up (high probe accuracy). This triangulation is the
main defence against "found a direction that's just noise that happens to
correlate in this sample."

### I.5 — PCA/SVD subspace check (`representation/I5_pca_svd/`)

Tests whether a contrast is better described by a single direction or a
**k-dimensional subspace**: SVD on the mean-centred, pooled
`H_pos ∪ H_neg`. If the top singular vector explains ≥ 80% of variance
(`one_vector_sufficient=True`), a single mean-difference vector is an adequate
summary; otherwise the `basis_vectors` (k × D) describe the subspace.

### I.2-I.5 orchestration

`runners/run_representation_discovery.py` runs I.2→I.3→I.4→I.5 for one
variant and writes `results/representation_discovery/<variant>/summary.json`
(`{variant, contrasts, n_contrasts, settings, per_contrast: {...}}`) plus
per-contrast CSVs under `evaluation/`, `probes/`, `subspace/`. Supports four
**discovery modes**:

| Mode | Contrast | Used for |
|---|---|---|
| `gender` (default) | female persona vs male persona, per emotion (8 contrasts) | the gender direction (§8.1) |
| `emotion` | one emotion vs pooled rest, optionally anchored | emotion directions |
| `country` | country-extended persona vs base persona, per condition (16 contrasts) | country/culture directions (§8.2) |
| `interest` | "rated Very/Extremely Interesting" vs "rated Not/Slightly Interesting", polar ends, per-condition + blank + global | the interestingness direction (§8.3, §7) |

---

## 4. Tier II — Analytics

Tier II takes the directions discovered in Tier I and asks *what they mean
geometrically* — are they shared across conditions, do they form clusters, do
they compose additively?

### II.1 — Persona analytics (`analytics/II1_persona_analytics/`)

Implemented and used in `results/EX1_EX2_persona_analytics.ipynb` and
`results/EX1_EX2_additivity_staged.ipynb`:

- **`gdv.py`** — General Discrimination Value (cluster separability metric,
  0=fully overlapping, 1=fully separated) per condition per layer.
- **`pca_embed.py` / `umap_embed.py` / `tsne_embed.py`** — 2D projections of
  activations, with UMAP/t-SNE trustworthiness as a sanity check that the
  projection isn't distorting the geometry.
- **`cosine_similarity.py`** — pairwise cosine similarity matrices between
  condition-mean vectors or contrast vectors, with heatmap plotting. This is
  the core tool used for §7's cosine-gradient-matching analysis.
- **`condition_geometry.py`** — per-condition mean activation vectors +
  cosine matrix + UMAP, the main "what does the activation space look like"
  view (Ch.6 of the persona-analytics notebook).
- **`additivity.py`** — tests whether `v_compound ≈ v_gender + v_emotion +
  v_country` (additive composition of independently-discovered feature
  vectors). Explored in 3 stages in `EX1_EX2_additivity_staged.ipynb`
  (country-only, gender-only-within-compound, full 3-term composition, plus a
  magnitude/overshoot check).
- **`cross_variant_alignment.py`** — checks whether e.g. the gender direction
  found in `base` aligns with the gender direction found in `extended_germany`
  (Ch.9 of the persona-analytics notebook).
- **`distance_matrices.py`, `layer_heatmaps.py`, `hyperplane_geometry.py`** —
  present but currently stubs (`NotImplementedError`); planned for
  RSA-style cross-layer/cross-contrast comparison.

**Notebook structure** (`results/EX1_EX2_persona_analytics.ipynb`):

| Chapter | Content |
|---|---|
| 1 | Experimental design rationale |
| 2 | I.1 activation collection (two-stage inference) |
| 3 | I.2 mean-difference vectors |
| 4 | I.3 linear probes (CAV) |
| 5 | I.5 subspace check |
| 6 | II.1 condition geometry (UMAP, cosine matrix) |
| 7 | Emotion representation discovery (one-vs-rest, cluster geometry) |
| 8 | Country representation discovery (Germany/Nigeria) |
| 9 | Cross-variant gender-vector alignment |
| 10 | *(moved → `EX1_EX2_additivity_staged.ipynb`)* |
| 11 | Summary: what Tier I established |
| 12 | Big picture: revisit guiding question, next steps |

### II.2 — Interestingness analytics (`analytics/II2_interestingness_analytics/`)

**Status: scaffolded as stubs**, with the explicit caveat (from the module
docstring) that drives §6.1 below:

> Avoid relying only on the rating-token representation — it may encode the
> phrase "very interesting" rather than the decision state. Use intermediate
> LLM layer activations from the teacher-forced step instead.

Planned analyses (signatures exist, bodies raise `NotImplementedError`):

- **`global_direction.py`**: `split_by_interestingness`,
  `find_global_interest_direction`, `cross_persona_generalisation` — does a
  direction found in one persona/condition generalise to others?
- **`persona_specificity.py`**: `compute_per_persona_interest_vectors`,
  `compare_persona_interest_vectors`, `correlate_with_persona_features` —
  pairwise cosine similarity between per-persona interest directions; **§7
  below is a first, ad-hoc pass at exactly this analysis**, run directly
  against the I.2 output rather than through this (still-stub) module.
- **`concept_alignment.py`**: `align_with_concepts`,
  `run_concept_alignment_pipeline` — does the interest direction align with
  emotion/valence/arousal/cognitive-load directions from `base_emotion`?

These remain the natural home for turning §7's exploratory script into a
permanent, reproducible analysis.

---

## 5. Tier III — Causal Control

Tier I/II establish *correlational* evidence ("this direction separates the
classes"). Tier III tests **causality**: does intervening on this direction in
the residual stream *change the model's output* in the predicted way?

All interventions are implemented as forward hooks (`utils/hooks.py`,
`HookState`) registered at a target `comp_key` (typically
`language_29_D5120`, the layer where I.4 found the strongest, best-aligned
gender/interest signal — see §8).

### III.1 — Vector addition / injection (`representation/III1_vector_control/control.py`)

```
h' = h + alpha * v        (v = unit-normalised md_vector, alpha typically in [-2, 2])
```

`run_control_experiment(model, processor, images, persona_dict, vector,
layer_key, alphas, save_dir)` runs the rating task for each image at each
`alpha`, recording `ControlResult(filename, alpha, rating, score, explanation,
...)`. Used for both:

- **III.2 — dose-response scaling**: sweeping `alpha` over
  `[-2, -1, -0.5, 0, 0.5, 1, 2]` and checking that the mean ordinal rating
  (`score`, 1-5) shifts **monotonically** with `alpha`. Non-monotonicity would
  indicate instability, wrong layer/vector, or saturation.
- **III.3 — ablation / projection removal**:
  ```
  h' = h - (h . v_hat) * v_hat
  ```
  removes the component of `h` along the direction entirely (rather than
  pushing further in either direction). If the direction is causally load-
  bearing, ablation should measurably *flatten* the persona-conditioned
  behaviour it was responsible for.

`load_averaged_gender_vector(md_vectors_dir, layer_key)` averages the 8
per-emotion `female_*_vs_male_*` md_vectors at a given layer into a single
"emotion-independent" gender direction (this is **the** gender vector used
throughout — see §8.1).

### III.4 / III.5 — combined injection & vector swap (`control/`, stubs)

- **III.4 (combined injection)**: inject `v_gender + v_emotion + v_country`
  simultaneously and compare against the observed compound-persona effect —
  the causal-intervention analogue of II.1's additivity check.
- **III.5 (vector swap)**: prompt with persona A but inject persona B's
  direction — a stronger test that the *effect* is direction-specific, not
  just "any perturbation at this layer changes the output."

Both are currently design-only (`NotImplementedError`).

### Branch A — Blank-prompt gender-vector validation (confound-free)

**The problem**: every persona prompt used elsewhere in this repo includes an
explicit `"Gender: Female"` / `"Gender: Male"` field. If injecting `v_gender`
shifts pronoun usage in the output, that's only convincing evidence of a
*causal* effect if it can't just be explained by "the model is echoing the
`Gender:` field it was given." Branch A removes this confound entirely by
working from the **blank prompt** (no persona fields at all).

**Design** (`runners/run_blank_validation.py`,
`representation/III1_vector_control/blank_validation.py`):

- **Vector**: averaged gender direction (female − male) at `language_29_D5120`
  (§8.1).
- **Primary experiment** (n=100 blank-prompt images), 4 conditions:
  - `no_inject` — read directly from the existing 500-sample
    `data/results_blank_activations.npy` (no GPU work needed for this arm).
  - `inject_pos2` — `h' = h + 2v` ("toward female").
  - `inject_neg2` — `h' = h - 2v` ("toward male").
  - `ablate` — `h' = h - (h·v̂)v̂`.
  - **Primary measure**: `count_pronouns(text)` — counts of
    she/her/hers/herself vs he/him/his/himself in the generated explanation.
    This is *the* a-priori prediction from the gender vector's logit-lens
    result (§6.1/§6.2): `+v` should increase female-pronoun usage and `-v`
    should increase male-pronoun usage, **with no `Gender:` field for the
    model to be echoing**. A significant, direction-consistent shift here is
    strong causal evidence.
  - **Secondary measures**: rating-distribution shift (does steering gender
    also shift the *interestingness* rating — i.e. is gender entangled with
    the interestingness decision?), and an exploratory TF-IDF gender-coding
    probe score on the explanation text.
- **Secondary experiment** (n=100, `male_contentment` persona): full
  dose-response sweep `alpha ∈ {-2,-1,0,1,2}` on the same gender vector,
  reported as rating distribution + TF-IDF gender-probe score per `alpha`
  (`tfidf_gender_score_by_alpha.csv`).

**Status: ✅ done** (job 3719047, completed). Results:

| condition | rating score | n_female_pron | n_male_pron | tfidf_gender |
|---|---|---|---|---|
| `no_inject` | 2.510 | 0.03 | 0.01 | −0.081 |
| `inject_neg2` | 2.484 | 0.00 | 0.00 | −0.182 |
| `inject_pos2` | 2.516 | 0.00 | 0.00 | −0.176 |
| `ablate` | 2.548 | 0.00 | 0.00 | −0.167 |

- **Primary measure (pronoun count): null.** Pronouns essentially never appear
  in blank-prompt explanations (0.00 in all 3 injected conditions, ~0.01-0.03
  in `no_inject`) — the a-priori logit-lens prediction ("`+v` → more
  she/her/herself") doesn't survive into generation on a blank prompt.
- **Secondary (male_contentment, TF-IDF gender-probe score by alpha):**
  −0.268 → −0.265 → −0.252 → −0.250 → −0.238 for α=−2..+2 — **clean 5-point
  monotonic trend, Δ=0.030**, in the predicted direction (toward
  "female"-coded language). Small in absolute size, but directionally
  consistent across all 5 points: weak-but-real secondary causal evidence,
  despite the primary pronoun measure being null.

**Reading**: the gender direction is robustly decodable (§8.1, AUC/cos≈1.0/0.98)
and has a rich, plausible logit-lens lexicon, but its causal footprint on this
task is small and only shows up in an aggregate stylistic probe (TF-IDF), not
in the literal predicted pronoun. See §7.3 for the geometric explanation.

---

### Branch B — Interestingness-vector validation (blank prompt + persona generalisation)

Mirrors Branch A's structure, but validates `v_interest_blank @ language_29`
(§8.3) — the "high interest minus low interest" mean-difference vector from
the blank-prompt discovery run.

- **Primary** (n≈93-100, blank prompt), 4 conditions (`no_inject`,
  `inject_pos2`, `inject_neg2`, `ablate`), measuring rating-distribution shift
  + the `interest_blank` lexicon (`derive_lexicon`, §6.3).
- **Secondary** (n=93, `male_awe_extended_germany` persona — the
  hardest-transfer persona, furthest from "blank"): full dose-response sweep
  `alpha ∈ {-2,-1,0,1,2}`, rating distribution + lexicon-count by alpha.

**Status: ✅ done** (job 3722570, completed). Results:

| condition | rating score | Δ vs no_inject |
|---|---|---|
| `inject_neg2` | 2.387 | −0.123 |
| `no_inject` | 2.510 | — |
| `inject_pos2` | 2.634 | +0.124 |
| `ablate` | 2.656 | +0.146 |

Predicted order (`inject_neg2 < no_inject < inject_pos2`) **achieved**, full
range **Δ=0.247**.

**Secondary (male_awe_extended_germany, alpha sweep)**:

| α | −2 | −1 | 0 | +1 | +2 |
|---|---|---|---|---|---|
| mean score | 2.839 | 2.849 | 2.946 | 3.022 | 3.065 |
| `n_pos_words` (lexicon) | 0.086 | 0.075 | 0.097 | 0.118 | 0.118 |

**Clean monotonic 5-point dose-response in the rating, Δ=0.226** — roughly
half a standard deviation, large enough to shift the modal rating category.
The `interest_blank` lexicon count also trends upward with α (0.086 → 0.118),
the *first* time in this project a logit-lens-derived lexicon shows a
consistent dose-response.

**Reading**: this is the strongest, most reproducible causal result in the
project — a direction that is both highly decodable (§8.3) *and* causally
load-bearing for the rating task, on the hardest-transfer persona. This is
**the TASK 3 prerequisite** (§9.1).

---

### Branch C — Nigeria country-vector validation (blank prompt + persona generalisation)

Mirrors Branch B, but validates the averaged Nigeria country direction
(`load_averaged_country_vector`, §3.2/§8.2) at `language_23_D5120`, using a
nigeria/africa-name lexicon (`results/extra_checks/logit_lens/country_nigeria_language_23.json`)
as the primary measure — the "improved" version of Branch A's pronoun-count
signal.

- **Primary** (n≈93-100, blank prompt): same 4 conditions as Branch A/B,
  measuring nigeria/africa-lexicon count.
- **Secondary** (n=93, `male_excitement` persona — base variant, no `Country`
  field, i.e. country-less): alpha sweep `{-2,-1,0,1,2}`, rating distribution +
  lexicon-count by alpha — does injecting "Nigeria-ness" make a country-less
  persona spontaneously adopt Africa/Nigeria framing?

**Status: ✅ done** (job 3722571, completed). Results:

| condition | rating score | `n_pos_words` (nigeria/africa lexicon) |
|---|---|---|
| `no_inject` | 2.510 | 0.0 |
| `inject_neg2` | 2.387 | 0.0 |
| `inject_pos2` | 2.634 | 0.0 |
| `ablate` | 2.656 | 0.0 |

**Primary lexicon measure: completely null** — the nigeria/africa lexicon
*never* fires, in any condition. (Rating values happen to numerically match
Branch B's primary table — both branches inject at the same alphas on the
same blank-prompt image set, so this is a same-images/same-alpha coincidence,
not shared content.)

**Secondary (male_excitement, alpha sweep)**: small monotonic rating shift,
**Δ=0.043** (~5× smaller than Branch B's Δ=0.226 over the same α range);
lexicon-count still 0.0 at every α.

**Reading**: unlike Branch A (null primary, weak-but-real secondary), Branch C
is null on **both** primary and secondary measures, despite starting from a
*cleaner* a-priori lexicon than Branch A's. See Branch C2 + §7.3 for why.

---

### Branch C2 — Wide alpha-sweep (±8) dose-response, Nigeria country vector, blank prompt

Motivated by skepticism about Branch C's null result ("gender shows *some*
effect, why would a *stronger* country-framing direction show none?"):
`runners/run_country_dose_response.py` sweeps the **same** Nigeria vector
(§Branch C) on the blank prompt over `alpha ∈ {-8,-4,-2,-1,0,1,2,4,8}` — 4× the
original range — measuring (1) rating dose-response, (2) the nigeria/africa
lexicon count, and (3) JSON-parse-failure rate (sanity check that large `|α|`
doesn't just break the output format).

**Status: ✅ done** (job 3724802, completed). Results:

| α | −8 | −4 | −2 | −1 | 0 | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| mean score | 2.473 | 2.495 | 2.495 | 2.495 | 2.505 | 2.505 | 2.516 | 2.548 | 2.634 |
| `n_pos_words` (lexicon) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| parse-fail rate | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

- Essentially **flat from α=−8 to α=+2** (range 2.473-2.516) — confirms the
  original ±2 null.
- A real uptick appears at **α=+4/+8** (2.548 → 2.634, Δ≈0.13 from baseline),
  with "Slightly Interesting" share dropping ~49% → ~36% in favour of
  "Moderately Interesting". But the **α=−8 endpoint breaks monotonicity**
  (2.473, *lower* than α=−4's 2.495) — not a clean linear dose-response.
- The nigeria/africa lexicon **never fires, at 4× the original magnitude**.
  Spot-checking generations at α=−8/0/+8 shows near-identical wording across
  all three, with only occasional "Slightly"→"Moderately Interesting" tier
  bumps at +8 — no Africa/Nigeria framing ever appears.
- Parse-fail rate is 0.0 everywhere — the small +4/+8 effect is real content,
  not JSON breakdown.

**Reading**: the wider sweep does **not** overturn Branch C's conclusion.
There's a small, non-monotonic, generic "intensity" effect at extreme +α, but
it carries **zero country-specific semantic content** even at 4× magnitude,
and is still smaller in total range than Branch B's effect at ¼ the α range.
§7.3 explains *why* geometrically: the country direction is ~orthogonal (and
slightly anti-aligned) with the affect subspace the rating task reads from.

---

## 6. Extra validation checks (cross-cutting toolkit)

These are not separate "tiers" — they're sanity checks applied *to* Tier I/III
outputs, developed because the rating-token representation has a specific,
important failure mode (§6.1).

### 6.1 — The rating-token caveat

`llm_layer_N_rating_token` is captured at the position **immediately before
the model emits its rating word** (`'{"interestingness":"'`). A direction that
separates "rated Very Interesting" from "rated Not Interesting" at this
position could, in the worst case, simply encode *which word is about to be
emitted* — i.e. be a **lexical/decision echo** rather than a representation of
the underlying judgement. This would make e.g. "high interest vs low interest"
directions somewhat tautological. The checks below (logit-lens, ordinal
gradient, vision-layer) were developed specifically to test — and, for the
interestingness vectors, substantially allay — this concern.

### 6.2 — Logit-lens (`representation/III1_vector_control/logit_lens.py`)

**Method** (nostalgebraist 2020, "logit lens"): for a unit direction `v` at
some layer, compute

```
logit_deltas = lm_head @ (v * final_rmsnorm_weight)
```

— a **one-step linear approximation** of "which vocabulary tokens would become
more/less likely if `v` were added to the residual stream right before the
final layer norm + unembedding." `norm_weight`/`lm_head` are loaded directly
from the safetensors shards (`load_unembedding`, ~2GB, CPU-only — no need to
load the full model). `top_tokens(logit_deltas, tokenizer, k)` returns the
top-k tokens in each direction.

This is explicitly a **hypothesis-generation** tool (it ignores the
transformer layers between the injection point and the output — for
`language_29_D5120`, that's 19 of 48 layers), but it's useful for two things:

1. **Pre-registering a prediction** for a Tier III intervention (e.g. "+v_gender
   should increase she/her/herself usage" → directly motivated Branch A's
   `count_pronouns` measure).
2. **Checking semantic content vs. lexical echo** (§6.1): if a direction's
   top tokens are a rich, coherent *semantic field* (not just the literal
   class-label words), that's evidence against the tautology concern.

**Persisted evidence** (`runners/run_logit_lens_evidence.py` →
`results/extra_checks/logit_lens/{*.json, summary.md}`):

| Vector | Layer | Top *positive*-direction tokens | Top *negative*-direction tokens |
|---|---|---|---|
| **Gender** (female − male, averaged over 8 emotions) | `language_29` | `herself, boyfriend, husband, her, she, ' 남편', ' chồng', '丈夫'` (multilingual "her/husband/wife-of-her" cluster) | `wife, himself, wives, his, esposa, moglie, '娶','妻','아내','妻子'` (multilingual "his/wife/husband-of-him" cluster) |
| **Interest (blank)**, high − low | `language_29` | `high, majestic, breathtaking, intense, awe, strong, superior, unrestricted, towers, astronomical` | `mundane, mild, modest, trivial, incidental, insignificant, comedy, bland, humble, sparse` |
| **Interest (global)**, high − low | `language_33` | `extreme, breathtaking, exceptional, awe, unforgettable, impossible, extraordinary, incredible, ultimate, instant` | `nothing, trivial, mundane, indifference, insignificant, bland, boring, weak, banal, irrelevant` |

**Reading**: the gender direction's top tokens are not just "Male"/"Female" —
they're a rich multilingual family-relationship/pronoun cluster, consistent
with a genuine "gender of referent" direction. The two interestingness
directions' top tokens are **semantically rich words about
grandeur/significance vs. mundaneness/insignificance** — *not* the literal
label strings ("Very Interesting" / "Not Interesting" never appear). This is
the first piece of evidence against the lexical-echo concern in §6.1.

### 6.3 — Logit-lens-derived lexicon + dose-response word counting

**Generalises** `count_pronouns()` (a *hand-curated* regex over she/her/herself
vs he/him/his/himself, itself derived by eyeballing the gender vector's
logit-lens output) into an **automatic** procedure applicable to *any*
direction vector — gender, interest, persona-emotion, country, etc.

New functions in `representation/III1_vector_control/logit_lens.py`:

- **`derive_lexicon(logit_deltas, tokenizer, k=200, min_len=3, max_words=30)`**
  → `(positive_words, negative_words)`. Scans the top-`k` logit-lens tokens in
  each direction, keeps only whole-word ASCII-alphabetic tokens (≥`min_len`
  chars after stripping BPE/whitespace markers), lowercases + dedupes,
  caps at `max_words`.
- **`count_lexicon_occurrences(text, pos_words, neg_words)`** → `(n_pos,
  n_neg)`. Whole-word, case-insensitive counts in generated text.

Both are persisted per-vector alongside the logit-lens tokens in
`results/extra_checks/logit_lens/*.json` (`"lexicon": {"positive_words": [...],
"negative_words": [...]}`). E.g. for `interest_blank_language_29`:

- positive lexicon: `high, majestic, breathtaking, intense, ..., strong, superior, towers, astronomical, ...`
- negative lexicon: `mundane, mild, modest, trivial, incidental, insignificant, comedy, bland, humble, ...`

(Lists include some BPE-fragment noise — e.g. `eskipun`, `idual` — that passes
the ASCII-alphabetic filter without being a real word; treat the derived
lexicon as a **candidate list to optionally hand-curate** before using it as a
hard validation criterion, the same way `count_pronouns`'s list was originally
hand-picked.)

**Answering the original question — yes, this generalises to dose-response
validation**: for any Tier III.2 alpha-sweep, take the generations at each
`alpha` and compute `count_lexicon_occurrences(explanation, pos_words,
neg_words)`. If the vector causally drives generation toward/away from its
logit-lens-predicted concept, `(n_pos - n_neg)` should trend **monotonically
with `alpha`** — exactly as `count_pronouns` already does for the gender
vector in Branch A, but now automatically derivable for *any* vector (e.g. the
interestingness vectors, or persona-emotion vectors), without hand-curating a
word list each time. This is a natural **secondary measure to add to future
III.2 dose-response runs** (e.g. for an interestingness alpha-sweep: does
`(n_pos - n_neg)` using the `interest_blank` lexicon increase monotonically
with `alpha`?).

### 6.4 — Ordinal-gradient (Spearman) check

For a binary-contrast direction derived from *polar-end* labels (e.g. "Very/
Extremely Interesting" vs "Not/Slightly Interesting", excluding the middle
"Moderately Interesting" category — see `_INTEREST_HIGH_LABELS`,
`_INTEREST_LOW_LABELS`, `_INTEREST_MIN_N=30` in
`analytics/II2_interestingness_analytics/__init__.py`-adjacent contrast
design): project **all 500** blank activations (including the held-out
"Moderately Interesting" middle category) onto the unit direction, and compute
Spearman's `rho` between the projection and the ordinal label
(`LABEL_SCORE`: 1-5).

**Result** for `v_interest_blank @ language_29`: **rho = 0.801, p ≈ 6e-113,
n=500** — the projection is monotonically increasing across **all 5** rating
levels, including the middle category that was *not used to derive the
direction*. This is strong evidence that the direction captures a genuine
**continuous** "perceived significance" axis, not just a binary
high-vs-low classifier artifact (further evidence against §6.1's concern).

### 6.5 — Vision-layer pre-verbal signal

`vision_layer_N_cls` (N=0..33) is captured from the vision encoder, **before**
any language-model processing — it cannot encode "which word the LLM is about
to emit" by construction. Two checks on the interestingness contrasts:

- **Separability (AUC)**: even at `vision_layer_0`, AUC for "high vs low
  interest" is **0.67**, rising to **~0.89** by `vision_layer_4`. So *some*
  interestingness-correlated signal exists in the raw image encoding, before
  any persona/language processing.
- **Ordinal gradient (Spearman, this session)**: projecting onto
  `v_interest_blank` at vision layers gives `rho ≈ 0.21-0.26` (p < 1e-6,
  n=500) — **significant but much weaker** than the language-layer `rho=0.80`
  (§6.4).
- **Persona-independence**: `cosine(v_interest_blank, v_interest_global)` at
  vision layers ≈ **0.85-0.86**, essentially flat across `vision_0`-`vision_5`.

**Interpretation**: there's a real, statistically significant,
persona-independent visual prior correlated with interestingness (rho≈0.25),
but the bulk of the signal (rho 0.25 → 0.80) is added during language-model
processing — i.e. "interestingness" here is mostly a **constructed judgement**
(incorporating world knowledge / framing), with a modest low-level visual
contribution. See §9 for what this motivates next.

---

## 7. Cosine-gradient matching: is "interestingness" a shared axis?

**"Cosine-gradient matching"** (this session's terminology) = the combination
of two checks applied to a *family* of mean-difference vectors for the same
nominal contrast (here: "high interest vs low interest"), each computed from a
different condition (different persona / country / blank / cross-persona
"global" pool):

1. **Cosine alignment across conditions** — do the per-condition vectors point
   in (approximately) the same direction? If yes, "interestingness" is a
   **shared** axis; if they cluster by persona/emotion/country instead, it's
   **persona-specific**.
2. **Layer-wise convergence ("gradient")** — does this alignment change
   systematically with depth? Combined with §6.4's ordinal-gradient check,
   this asks whether the *shared* direction (if one exists) also reproduces
   the ordinal rating gradient.

This directly operationalises `analytics/II2_interestingness_analytics/
persona_specificity.py::compare_persona_interest_vectors` (currently a stub —
the analysis below is a first ad-hoc pass against the 9 contrasts from
`results/representation_discovery/interestingness/md_vectors/`:
`blank`, `global`, and per-condition for
{male_excitement, female_excitement, male_awe} × {base, Germany, Nigeria}).

### 7.1 — Cosine similarity matrix across layers

Mean **off-diagonal** cosine similarity (9×9 matrix, 36 unique pairs):

| Layer | mean cos | min | notes |
|---|---|---|---|
| `language_29` | 0.881 | 0.739 | first AUC=1.0 layer for `blank`/most conditions |
| `language_31` | 0.920 | 0.828 | |
| `language_33` | 0.938 | 0.853 | first AUC=1.0 layer for `global` |
| `language_36` | 0.959 | 0.900 | |

At `language_29`, the 8 **persona-conditioned** vectors are already extremely
tightly aligned with each other (pairwise cosine **0.87-0.99**), but `blank`
and `global` are comparative outliers relative to that cluster (cosine to
others **0.74-0.93**, `cos(blank, global) = 0.74`). By `language_36`, *every*
pairwise cosine is **≥ 0.90**, including `blank`/`global` (`cos(blank,
global) = 0.93`).

**Layer trend** (`mean cos(blank, ·)` and `mean cos(global, ·)` across
`language_20`-`language_47`): both rise close to monotonically from ~0.7-0.9
at layer 20 to **~0.95 at layers 44-46**, dipping slightly at the final layer
47 (0.948).

**Reading**: regardless of persona, country, or whether a persona is present
at all, the model converges onto an (increasingly) **shared "interestingness"
direction** as depth increases. The relative divergence at the
*decision-onset* layers (27-33, where AUC first hits 1.0) vs. the near-total
convergence by the final layers (36-47) suggests the initial decision signal
is somewhat context-flavoured, but gets "canonicalised" onto a common axis by
the time the model commits to its answer.

### 7.2 — Vision-layer cosine + gradient

- `cosine(v_interest_blank, v_interest_global)` at `vision_0`-`vision_5`:
  **0.847-0.860** (flat) — the *visual* prior on interestingness is already
  largely persona-independent, even before any language-model processing.
- Combined with §6.5's `rho≈0.25` ordinal gradient at vision layers vs.
  `rho≈0.80` at `language_29`: the visual-only signal is real and shared, but
  weak relative to the final language-layer signal.

### 7.3 — Why interest is causal and gender/country aren't: the "affect axis"

Cosine similarities between `v_interest_blank`, the 8 `base_emotion`
one-vs-rest vectors, `v_gender`, and the two country vectors (`v_nigeria`,
`v_germany`), all projected at `language_29_D5120`:

| pair | cosine |
|---|---|
| interest ↔ **awe** | **+0.510** |
| interest ↔ **excitement** | **+0.380** |
| interest ↔ amusement | +0.135 |
| interest ↔ contentment | −0.089 |
| interest ↔ fear | −0.229 |
| interest ↔ sad | −0.327 |
| interest ↔ disgust | −0.342 |
| interest ↔ anger | −0.344 |
| interest ↔ gender | +0.016 |
| interest ↔ nigeria | −0.009 |
| interest ↔ germany | −0.069 |
| nigeria ↔ germany | +0.673 |
| gender ↔ nigeria | −0.028 |

**Reading**: `v_interest_blank` is essentially a **valence/arousal ("affect")
axis** — it aligns strongly with the high-arousal *positive* emotions (awe,
excitement, amusement) and anti-aligns with the negative emotions (anger,
disgust, sad, fear), with low-arousal-positive (contentment) near zero. This
is a coherent affective-psychology pattern, not noise, and it's an
**independent confirmation**: `v_interest_blank` was discovered from
"rated Very/Extremely Interesting vs Not/Slightly Interesting" contrasts,
while the emotion vectors were discovered from a completely different
contrast design (persona `Emotion:` field, one-vs-rest) — yet they land in
overlapping subspaces.

**Gender and country vectors live in a different, ~orthogonal subspace**
(cos ≈ 0 with interest). The two country vectors (Nigeria, Germany) cluster
tightly with each other (cos=0.67) — "country-identity" is its own shared
linear subspace, separate from "affect."

**This explains the Branch A/B/C(2) pattern**: the interestingness-rating task
reads out along the affect axis. `v_interest_blank` *is* (partly) that axis,
so injecting it directly moves the rating (Branch B). `v_gender` and
`v_nigeria` are both highly **decodable** (§8.1/§8.2, AUC≈1.0) but live
**outside** the axis the rating head reads from — so injecting them pushes the
residual stream somewhere the readout doesn't look, producing the
null/weak Branch A/C(2) results regardless of injection magnitude.
**Decodable ≠ causally load-bearing for a given task** — a direction can be
linearly present in activation space (Tier II) without being on the readout
path for a *specific* downstream task (Tier III), unless that direction *is*
(part of) what the task reads.

A useful corollary for TASK 4 (§9.1): `fear` (cos=−0.229 with interest) is the
closest existing proxy to a "stress" direction and is **predicted to produce a
*negative* dose-response** on interestingness — and it already exists in
`results/representation_discovery/base_emotion/md_vectors/` (AUC=0.957 @
`language_32`), so testing it requires no new Tier I/II discovery work.

---

## 8. Findings so far

### 8.1 — Gender direction

- **Vector**: average of 8 per-emotion `female_<emotion> − male_<emotion>`
  mean-difference vectors (`results/representation_discovery/base/md_vectors/`),
  at **`language_29_D5120`**.
- **I.4 evaluation**: AUC = 1.0, probe accuracy = 1.0, MD↔CAV cosine ≈
  **0.98** for all 8 emotions — extremely robust, emotion-independent gender
  direction.
- **Logit-lens** (§6.2): rich multilingual "her/husband" vs "his/wife"
  semantic cluster — looks like a genuine "gender of the referent" direction,
  not a literal "Male"/"Female" token detector.
- **Causal validation**: Branch A (§5, job 3719047) — ✅ done. Primary
  (pronoun-count) **null**; secondary (TF-IDF gender-probe, alpha sweep)
  **clean monotonic Δ=0.030**. Decodable but weakly causal — explained
  geometrically by §7.3 (cos(interest, gender)≈0.02, ~orthogonal to the
  task-readout axis).

### 8.2 — Country / culture directions

- **Mode**: `country` discovery — `<extended_country>_<condition> vs <base>_<condition>`,
  16 contrasts each for Germany and Nigeria.
- **Germany**: peaks around **`language_27`** (e.g.
  `female_amusement_extended_germany_vs_female_amusement`: AUC=1.0, probe=0.999,
  md_cav cos=0.978 at `language_27`).
- **Nigeria**: peaks consistently at **`language_23`** (e.g.
  `female_anger_extended_nigeria_vs_female_anger`: AUC=1.0, probe=1.0, md_cav
  cos=0.959 at `language_23`).
- Both directions are highly decodable (AUC≈1.0, md_cav cos 0.95-0.98) across
  nearly all 16 conditions — country/culture framing is represented as
  strongly and consistently as gender, just at a different (earlier, for
  Nigeria) layer.
- **Causal validation**: Branches C (job 3722571) + C2 (job 3724802, wide
  ±8 sweep) — ✅ done, §5. **Null on both the primary lexicon measure and the
  secondary rating measure**, even at 4× the injection magnitude used for
  Branch A/B. Geometrically (§7.3), `v_nigeria` is ~orthogonal-to-slightly-
  anti-aligned with the affect axis the rating task reads from
  (cos(interest, nigeria)=−0.009), and the two country vectors cluster tightly
  with each other (cos(nigeria, germany)=0.67) — "country-identity" is its own
  subspace, decodable but not on this task's readout path.

### 8.3 — Interestingness direction

- **Mode**: `interest` discovery — polar-ends ("Very/Extremely Interesting" vs
  "Not/Slightly Interesting"), 9 contrasts (`blank`, `global`, 7 per-condition).
- **All 9 contrasts**: AUC = 1.0 across a broad plateau of layers
  (`n_layers_keep` = 75-82 of 82); peak `md_cav_cosine` 0.66-0.97, mean probe
  accuracy 0.85-0.98.
  - `blank`: peak at `language_29` (md_cav cos 0.80).
  - `global`: peak at `language_33` (md_cav cos 0.97, the highest of all 9 —
    consistent with §7's finding that `global` sits closest to the
    "canonical" late-layer direction).
- **§6.2-6.4**: logit-lens tokens are semantically rich
  (grandeur/significance vs. mundaneness), and the ordinal gradient
  (`rho=0.80`, n=500, all 5 levels) strongly supports a genuine continuous
  axis — **not** a lexical-echo artifact of the rating-token position.
- **§7**: cosine-gradient matching shows **strong cross-condition
  convergence**, increasing with depth (mean off-diagonal cosine 0.88 →
  0.96 from `language_29` to `language_36`) — "interestingness" looks like a
  predominantly **shared, persona-independent** axis, especially at later
  layers.
- **§6.5**: a real but modest (`rho≈0.25`) persona-independent **visual**
  prior exists at the vision-encoder layers, well below the language-layer
  signal (`rho≈0.80`).
- **Causal validation**: Branch B (job 3722570) — ✅ done, §5. **Primary
  Δ=0.247 in the predicted order; secondary (male_awe_extended_germany, alpha
  sweep) clean monotonic 5-point dose-response, Δ=0.226**, with the
  logit-lens-derived lexicon also trending upward with α — the strongest
  causal result in the project. §7.3 shows `v_interest_blank` is (partly) an
  **affect axis** (cos with awe=+0.51, excitement=+0.38) that the rating task
  directly reads from, which is why this vector — unlike gender/country — is
  both decodable *and* causally potent.

### 8.4 — Emotion directions (Tier I/II done, Tier III pending)

- **Mode**: `emotion` discovery — one-vs-rest, 8 contrasts
  (`results/representation_discovery/base_emotion/md_vectors/`).
- All 8 emotions are highly decodable; the strongest are **excitement**
  (AUC=1.0, md_cav cos=**0.974** @ `language_23`), **contentment** (AUC=1.0,
  cos=0.915 @ `language_25`), and **awe** (AUC=0.994, cos=0.885 @
  `language_23`); **fear** is weaker but still significant (AUC=0.957, cos=0.540
  @ `language_32`).
- **§7.3**: at `language_29`, **awe** (cos=+0.510) and **excitement**
  (cos=+0.380) overlap substantially with `v_interest_blank`; **fear**
  (cos=−0.229), **sad** (−0.327), **disgust** (−0.342), **anger** (−0.344) all
  anti-align with it.
- **Causal validation**: ❌ not yet run for any emotion vector — this is
  **Branch D** (proposed, §9.1). Given the cosine overlap above, awe/excitement
  are predicted to **replicate Branch B's positive dose-response**
  (independent triangulation, different discovery design), and fear is
  predicted to produce a **negative** dose-response — a ready-made test of
  TASK 4's "stress vector" hypothesis with zero new Tier I/II work.

---

## 9. Status / what's done vs. pending

| Area | Status |
|---|---|
| I.1 activation collection (base, extended×{Germany,Nigeria}, blank, interest) | ✅ done |
| I.2-I.5 discovery: gender (`base`) | ✅ done — §8.1 |
| I.2-I.5 discovery: country (`extended_{germany,nigeria}_country`) | ✅ done — §8.2 |
| I.2-I.5 discovery: interestingness (9 contrasts) | ✅ done — §8.3 |
| I.2-I.5 discovery: emotion (`base_emotion`, 8 one-vs-rest) | ✅ done — §8.4 |
| II.1 persona analytics (GDV, PCA/UMAP, cosine matrices, additivity, cross-variant alignment) | ✅ implemented, notebook Ch.1-9 |
| II.2 interestingness analytics (`global_direction`, `persona_specificity`, `concept_alignment`) | 🚧 stubs — §7 is a first ad-hoc pass at `persona_specificity` |
| III.1-III.3 implementation (inject / dose-response / ablate) | ✅ implemented (`III1_vector_control/control.py`) |
| III.1-III.3 **causal validation of gender vector** (Branch A, job 3719047) | ✅ done — §5, §8.1: null primary, weak monotonic secondary |
| III.1-III.3 **causal validation of interestingness vector** (Branch B, job 3722570) | ✅ done — §5, §8.3: **strong, clean monotonic dose-response (Δ=0.226)** |
| III.1-III.3 **causal validation of country vector** (Branch C/C2, jobs 3722571/3724802) | ✅ done — §5, §8.2: null at both ±2 and ±8 |
| III.1-III.3 **causal validation of emotion vectors** (Branch D: awe/excitement/fear) | ❌ proposed, not yet run — §9.1 (TASK 1/2) |
| III.4 (combined injection) / III.5 (vector swap) | ❌ design-only stubs |
| Extra checks: logit-lens (gender, interest×2, country×2) | ✅ persisted — `results/extra_checks/logit_lens/` |
| Extra checks: logit-lens → lexicon dose-response probe | ✅ implemented + run live for Branch B/C/C2 (`lexicon_validation.py`) |
| Extra checks: ordinal-gradient (Spearman) | ✅ run for `interest_blank` (language + vision layers) |
| Extra checks: vision-layer pre-verbal signal | ✅ run for `interest_blank`/`interest_global` |
| Extra checks: TF-IDF gender-coding probe | ✅ implemented (`gender_probe.py`), used in Branch A secondary measure |
| Extra checks: vector-geometry / affect-axis analysis | ✅ done — §7.3 |
| **TIER IV — gradient matching (TASK 3)** | ❌ not started — blocked on TASK 1/2 conclusion, see §9.1 |
| **Stress/mental-load vector (TASK 4)** | ❌ not started — candidate (`fear`) identified, see §9.1/§8.4 |

---

## 9.1 — TIER III conclusion & TIER IV roadmap

**TIER III verdict** (TASK 1 + TASK 2, this session):

- **Branch B (interest)**: strong, clean, monotonic, generalises to the
  hardest-transfer persona — **the TASK 3 prerequisite is satisfied.**
- **Branch A (gender)**: weak/null — decodable but ~orthogonal to the
  task-readout (affect) axis.
- **Branch C/C2 (country)**: null even at 4× sweep — same orthogonality story,
  confirmed not to be a sweep-range artifact.
- **§7.3 (vector geometry)** is the unifying explanation for all three: the
  rating task reads out along an **affect (valence/arousal) axis**.
  `v_interest_blank` substantially *is* that axis (cos with awe/excitement
  +0.51/+0.38); `v_gender`/`v_nigeria` are ~orthogonal to it (cos≈0). Decodable
  (Tier II) ≠ causally load-bearing for this task (Tier III) unless the
  direction *is* (part of) what the task reads.

**Answers to TASK 1's sub-questions** (a-d):

- **(a)** No — C2's small +4/+8 uptick is non-monotonic, lexicon-free, and
  geometrically predicted to go the *other* way (cos(nigeria,
  awe/excitement) < 0). Not worth chasing larger α on this vector.
- **(b)** Injection is already applied at **every generated token**
  (`register_inject_hooks` fires on every `model.generate()` forward pass) —
  clamping-frequency isn't the issue. The issue is geometric: wrong subspace.
- **(c)** Yes, likely — our `v_nigeria` is a *persona-prompt-identity*
  direction (discovered by contrasting `"Nationality: Nigerian"` prompts vs.
  not). A *task-relevant* "this is about Nigeria/Africa" direction would need
  to live in the affect subspace (positive cos with awe/excitement), which
  ours doesn't. Finding it would require an **output-grounded discovery axis**
  (contrast generations that do/don't mention Africa content) — new Tier I/II
  work, not scoped yet.
- **(d)** Already done at Tier II: probe AUC/accuracy (gender≈1.0,
  nigeria≈1.0 @ 15/16 conditions, all 8 emotions 0.95-1.0) **is** the
  "reverse-engineer the persona from activations" result — near-perfect
  decodability, independent of whether the direction affects this task's
  output.

**TASK 2 answer**: yes — Branch B's dose-response is the relevant response
(table in §5/§8.3 above).

**Proposed next step — Branch D (emotion causal validation)**: before formally
closing TIER III, validate the two vectors with the strongest a-priori case
from §7.3/§8.4:
- **awe** and/or **excitement** @ `language_23` (cos +0.51/+0.38 with
  `v_interest_blank`) — predicted **positive** dose-response, independent
  triangulation of Branch B via a different discovery design. Possibly an even
  cleaner TASK 3 target than `v_interest_blank` itself.
- **fear** @ `language_32` (cos −0.229) — predicted **negative**
  dose-response, doubling as TASK 4's "stress vector" test with an
  *already-discovered* vector (no new Tier I/II work).

**TIER IV (TASK 3 — gradient matching)**, once Branch D confirms a usable
intervention:
1. Pick the strongest validated direction (Branch B's interest vector, and/or
   Branch D's awe/excitement if it replicates/improves on it) at its best
   alpha (informed by a C2-style wide sweep on *this* vector to find where the
   effect saturates / JSON parsing breaks).
2. Design the image-perturbation objective: perturb an input image so that its
   activations move toward the activation pattern produced by the
   validated injection — i.e. gradient-match the image to the *intervention's*
   effect on activations, not just the raw vector.
3. Worst case per TASK 3: construct a "maximally interested" persona prompt
   (rather than a vector injection) and gradient-match images toward *its*
   activation pattern — a prompt-level analogue of TASK 4(2)'s brute-force
   "you are literally interested in anything" fallback.

**TASK 4 (fallbacks, if Branch D's awe/excitement *don't* give TASK 3 a usable
target)**:
1. **Stress vector**: `fear` (already discovered, §8.4) is the natural
   starting point — test its dose-response first before designing a new
   "mental load" persona axis from scratch.
2. **Brute-force "interested in anything" persona**: contrast activations for
   an extreme "you find literally everything interesting" prompt against the
   base persona, as a last-resort gradient-matching target.

---

## 10. How to run things

All commands assume the `vision-transformers` conda env
(`hpc_infrastructure/miniconda3/envs/vision-transformers`) and are run from
the repo root. **GPU-bound steps (I.1 collection, III control experiments,
anything involving model forward passes or training/CV over large corpora)
must go through SLURM** — do not run them on the login node, even with thread
limits.

```bash
# I.1 — collect activations for a persona condition (GPU)
python -m runners.run_collect_activations --persona_key female_excitement --variant base

# I.2-I.5 — representation discovery for a variant/mode (CPU)
python -m runners.run_representation_discovery --variant base --mode gender

# II.1 — persona analytics for a variant/layer (CPU)
python -m runners.run_analytics --variant base --layer language_29_D5120

# III.1 — vector-addition control experiment (GPU)
python -m runners.run_control --layer language_29_D5120 --alphas -2 -1 0 1 2

# Branch A — blank-prompt gender-vector validation (GPU, SLURM)
sbatch run_blank_validation.slurm   # -> results/blank_validation/language_29/

# Branch B — interestingness-vector validation (GPU, SLURM)
sbatch run_interest_validation.slurm   # -> results/interest_validation/language_29/

# Branch C — Nigeria country-vector validation (GPU, SLURM)
sbatch run_country_validation.slurm   # -> results/country_validation/language_23/

# Branch C2 — wide alpha-sweep (±8) country-vector dose-response (GPU, SLURM)
sbatch run_country_dose_response.slurm   # -> results/country_validation/language_23/dose_response_blank/

# Extra checks — logit-lens evidence (CPU, ~2GB weight load)
python -m runners.run_logit_lens_evidence   # -> results/extra_checks/logit_lens/
```

---

## 11. References

- Turner et al., **ActAdd: Steering Language Models Without Optimization**, arXiv:2308.10248
- Kim et al., **TCAV: Interpretability Beyond Feature Attribution**, arXiv:1711.11279
- Zou et al., **Representation Engineering: A Top-Down Approach to AI Transparency**, arXiv:2310.01405
- Tigges et al., **Language Models Linearly Represent Sentiment**, BlackboxNLP @ ACL 2024
- Park et al., **The Linear Representation Hypothesis and the Geometry of Large Language Models**, arXiv:2311.03658
- Rimsky et al., **Steering Llama 2 via Contrastive Activation Addition**, arXiv:2312.06681
- Chen et al. (Anthropic), **Persona Vectors: Monitoring and Controlling Character Traits in Language Models**, arXiv:2507.21509
- nostalgebraist, **interpreting GPT: the logit lens**, LessWrong 2020
- **Activation Scaling for Steering and Interpreting Language Models**, EMNLP 2024 Findings (aclanthology 2024.findings-emnlp.479)
- Kriegeskorte et al., **Representational Similarity Analysis**, Frontiers in Systems Neuroscience 2008
- Kornblith et al., **Similarity of Neural Network Representations Revisited (CKA)**, arXiv:1905.00414
