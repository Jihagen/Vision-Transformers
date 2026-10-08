# Data and model availability

This repository contains code and aggregated result exports. It does not contain
images, model weights, activations or trained perturbations. This file lists what
the experiments use and how to obtain it.

## Model

| | |
|---|---|
| Model | `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| Access | Gated on the Hugging Face Hub under the Llama 4 Community License; request access there |
| Local path | `hpc_infrastructure/hf_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/local-repo`, or set `VT_MODEL_PATH` |
| Revision | Not recorded. The experiments used a local snapshot; the Hub commit was not logged |

## Images used for discovery, intervention and UAP training

The main image set is 1,000 images from the **Open Images V7** training split,
stored locally as `data/imagesDemographics/0001.jpg` … `1000.jpg`. The experiments
use a fixed 500-image subset (`data/selected_uniform_500_manifest.pkl`).

- The images are not redistributed here, with one exception: nine held-out
  images appear in the UAP example gallery (`web_export/assets/uap_gallery/`) as
  336 × 336 model-input renderings, clean and perturbed. Their per-image
  attribution is still to be added. Open Images lists them under
  CC BY 2.0, but the licence of each image is set by its original uploader and
  is not verified by the dataset maintainers.
- They can be downloaded by image ID from the Open Images project
  (https://storage.googleapis.com/openimages/web/index.html).
- **Open item:** the mapping from local file names to Open Images IDs, and the
  origin of the accompanying interestingness annotations
  (`data/df_common_machine_int.pkl`), are not yet published. Both are needed
  before the image set can be reconstructed by someone else.

## Datasets used for the transfer evaluation

| Task | Dataset | Access | Sample |
|---|---|---|---|
| Shopping relevance | `Marqo/marqo-GS-10M` | Hugging Face Hub | 300 query–product pairs |
| Damage severity | `QCRI/MEDIC` | Hugging Face Hub | 300 images, stratified by severity |
| Moral evaluation | `AIML-TUDA/smid` (Socio-Moral Image Database) | Hugging Face Hub, gated: accept the terms and use an approved token | 274 images |

None of these images or their human annotations are redistributed. Check each
dataset card for its licence and citation before reuse. The frozen samples are
built with:

```bash
python attack/IV2_generalisation/prepare_datasets.py --task shopping_relevance --n_samples 300
python attack/IV2_generalisation/prepare_datasets.py --task damage_severity --n_samples 300
HUGGINGFACE_HUB_TOKEN=... python attack/IV2_generalisation/prepare_datasets.py --task moral_evaluation --n_samples 300
```

This needs internet access and writes images to `data/generalisation/` and a
sample manifest to `results/generalisation/<task>/sample_manifest.csv`. The
manifests of the samples actually used are not published yet.

## Not distributed

| Artifact | Location (local only) | Reason |
|---|---|---|
| Hidden-state activations | `data/experiments/` (about 33 GB) | Size |
| Direction vectors, probes, subspace bases | `results/representation_discovery/*/` (about 100 GB) | Size |
| Persona-condition mean activations | `results/analytics/<set>/language_*/mean_vectors_*.npy` | Not released at this stage |
| Trained perturbations | `results/universal_perturbation_projected/*/eps*/delta*.npy` | Not released as files. The gallery's lossless clean/perturbed image pairs allow the `interest` and `excited_vs_angry` perturbations to be recovered where they are not clipped |
| Per-image model outputs | `results/**/control_results.csv`, `results/generalisation/*/predictions.csv` | Tied to non-redistributable images |
| Analysis notebooks and rendered figures | `results/*.ipynb`, `results/**/*.png` | Some figures embed source images |

The aggregated numbers derived from these artifacts are in `web_export/`.

Two exports are derived from the direction vectors and condition means at
`language_29_D5120`: `web_export/data/geometry/vector_field_3d.json` and
`persona_composition_paths.json`. They contain 3D display coordinates, cosine
similarities and norms. The 5120-dimensional vectors and the projection bases
are not included; the source files are identified by SHA-256 in
`web_export/release_manifest.json`. Field definitions are in
`web_export/README.md`.
