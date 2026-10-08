# Project status

This is an MSc thesis in progress. The table below states, per study level, what
saved results exist, what the website can show from them today, and what is still
missing before a final research release. It is based on the saved artifacts under
`results/` and on `web_export/release_manifest.json`, not on intended outcomes.

Status values:

- **available**: saved result tables exist and are exported to `web_export/`
- **in progress**: partial saved results; experiments or analysis still open
- **not yet website-ready**: results exist only as notebook output or figures that cannot be published as they are

Every export is flagged `preliminary`. All experiments so far are single runs on
one model, one prompt template and one image sample, without seeds or repeats.

| Level | Status | Saved results that support it | Can be shown on the website now | Still needed for a final release |
|---|---|---|---|---|
| **1. Representation discovery** | available (preliminary) | `results/representation_discovery/<variant>/` for 9 variants, 81 contrasts × 82 layers: `evaluation/vector_evaluation.csv`, `probes/probe_accuracy.csv`, `summary.json` | Layer-wise separability curves per contrast and per contrast family (`web_export/data/discovery/`) | Held-out projection AUC (currently measured on the data the direction was fitted on); cross-validation grouped by image; record the model revision; publish the image ID list |
| **2. Representation geometry / analysis** | in progress | Condition mean vectors at `language_20/23/29` for 3 persona sets (`results/analytics/<set>/`); `results/analytics/early_layer_sweep/early_layer_sweep.json`; `results/representation_discovery/interestingness/analytics/concept_alignment*.csv`; mean-difference directions under `results/representation_discovery/*/md_vectors/` | Condition cosine-similarity heatmaps; nearest-neighbour grouping by layer; interestingness–concept alignment by layer; rotatable 3D view of 15 concept directions with exact cosines, and staged additive composition of the 32 compound personas for every feature order, both at `language_29_D5120` (`web_export/data/geometry/`) | The Stage 1 and Stage 2 additivity checks and cross-persona gender transfer exist only as notebook output (`results/EX1_T1_*.ipynb`, `EX1_T2_additivity.ipynb`) and must be written to tables; held-out additivity (feature vectors are currently averaged over the conditions they predict); activation-scale norms of the mean-difference directions (saved unit-normalised); the same geometry at other layers; nearest-neighbour analysis in activation space instead of a 2-D UMAP embedding |
| **3. Causal intervention** | available (preliminary) | Blank-prompt α sweeps, 9 strengths × 93 images each: interestingness, excited-vs-angry, workload (`language_24` and `language_29`), country/Nigeria (`language_23`) under `results/*_validation/**/dose_response_blank/control_results.csv`; single-dose and ablation summaries for gender, country, interestingness and 8 emotions | Dose-response curves with label distributions; single-dose comparison (`web_export/data/causal/`) | Blank-prompt sweep for gender (only ±2 and ablation exist); random-direction and norm-matched controls; repeats or bootstrap intervals; a second behavioural task besides interestingness |
| **4. Universal adversarial perturbation and transfer** | in progress | UAPs for 3 targets at `language_29` plus workload at `language_24`, 4 budgets each (`results/universal_perturbation_projected/`); original-task evaluation for interest and excited-vs-angry only (`results/attack_eval_projected/`); transfer to 3 unseen tasks for the 3 `language_29` UAPs (`results/generalisation/`) | Hidden-state alignment vs budget; rating shift on the original tasks (2 targets); label shift on the 3 transfer tasks; all marked preliminary and with the budget scale explained (`web_export/data/attack/`) | Original-task evaluation for the workload UAP; any behavioural evaluation of the `language_24` workload UAP; budgets in a conventional imperceptibility range and a random-perturbation baseline; seeds; significance tests for the transfer shifts; transfer "level 2" (amplitude scaling) has not been run; UAP work is ongoing |

## Notes that apply to what is shown

- **Perturbation budgets are large.** ε is an L∞ bound in processor-normalised
  pixel units with range [−1, 1]. ε = 1.0 is half the pixel range and ε = 2.0 is
  the whole range. At ε = 0.1 the saved shifts are close to zero on every
  task (normalised shift below 0.02 on the transfer tasks, and slightly negative
  on the original interestingness task). At ε = 0.5 they are small for the
  interest and excited-vs-angry UAPs and larger for workload. ε = 2.0 should be
  presented as a saturation regime in which outputs no longer describe the image.
- **Transfer is measured; its interpretation is not settled.** At ε = 0.5 only the
  workload UAP shifts all three transfer tasks. Its shift on damage severity
  comes from images the model describes as degraded. The three UAP target
  directions are not one direction (cosines +0.40, −0.05 and −0.50), and the
  arousal × ε interaction on SMID disappears when image valence is controlled.
  A noise baseline, probe-alignment activations and the workload UAP on the
  original tasks are queued; until they are in, the shared-state reading is a
  hypothesis.
- **Budget-response correlations are over four points.** The saved Spearman
  coefficients only say whether four means are ordered. Their p-values are not
  exported.
- **The arousal analysis on the moral-evaluation task is exploratory.** The
  regression pools four budget levels of the same images.
- **The two interactive geometry figures show geometry, not control.** The 3D
  vector field is a display projection that retains 73.9% of the squared norm of
  the 15 unit directions; alignment is to be read from the exported full-space
  cosine matrix. The composition paths reproduce the notebook's mean cosine of
  0.9684 between predicted and observed compound shift, with feature vectors
  averaged over the same 32 conditions they predict. Neither figure involves an
  intervention on the model.
- **Superseded runs are not exported.** `results/universal_perturbation/` and
  `results/attack_eval/` predate the current UAP runs. The early GDV / t-SNE /
  UMAP outputs in `results/metrics/` and `results/experiments/` belong to the
  exploratory phase before the four-level structure.

## Hidden for now

| Item | Why |
|---|---|
| Example figures with images (`results/presentation/07_*`, `09_*` and similar) | They embed source images that cannot be redistributed, including photographs of people |
| Per-image model outputs and explanations | Tied to those images; at high budgets some outputs describe profane content |
| Logit-lens token lists (`results/extra_checks/logit_lens/`) | Saved, but not yet reviewed for publication |
| Notebook-only geometry results (Stage 1 and 2 additivity checks, cross-persona transfer) | Not saved as tables, so the website cannot plot them from data |
| Trained perturbations (`delta*.npy`) | Not released at this stage |

## Open administrative items

- The model revision used for the experiments was not logged.
- The mapping from local image files to Open Images IDs is not published (see `DATA.md`).
- Dependencies in `env.yml` are unpinned.
- End-to-end reproduction from raw data has not been tested outside the original compute environment.
