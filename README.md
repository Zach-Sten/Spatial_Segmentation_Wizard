# Spatial Segmentation Wizard

<img src="img/segmentation_wizard_icon.png" width="200" align="left" hspace="20">

An interactive pipeline for running and benchmarking multiple spatial transcriptomics segmentation methods on HPC clusters via SLURM. All methods execute inside a single Singularity container.

We're working on adding in a docker container as well and are excited to implement a classifier that builds off of reference data to give quick annotations for downstream QC metrics. Stay tuned for more! ✨🪄💫✨

Suggestions for QC elements or methods to add are always welcome! 

<br clear="left">

## Overview

The pipeline wraps multiple segmentation methods behind a single interactive wizard. Point it at your data, pick your methods, and it generates and submits all SLURM jobs — one per sample per method — with automatic dependency chaining so QC runs after segmentation finishes.

## Comparisons
All new segmentation masks can be viewed in Xenium explorer for quick visual assessments.
<p align="center">
  <img src="img/comparison_1.png" width="1000">
</p>

#### Annotations:
If a refrence dataset is used we can also quickly spot annotations made by the classifier and their confidence. Here we use a rank based gradient boosting classifier to identify cells quickly.
<p align="center">
  <img src="img/annot_comparison_1.png" width="1000">
</p>
<p align="center">
  <img src="img/confidence_comparison_1.png" width="1000">
</p>

## Current Status
**Working:**
- **ProSeg** — probabilistic segmentation, full Explorer export (https://www.nature.com/articles/s41592-025-02697-0)
- **Baysor** — Bayesian transcript-based segmentation, dask-parallelized (https://www.nature.com/articles/s41587-021-01044-w)
- **Xenium baseline** — native Xenium segmentation loaded as reference (https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/algorithms-overview/segmentation)
- **Fastreseg** - R based program that uses a reference dataset to match cell types (https://www.nature.com/articles/s41598-025-08733-5)
- **Cellpose** - neural net model to fit segmentation boundaries by expansion of nuclear mask (https://www.nature.com/articles/s41592-020-01018-x)
- **StarDist** - Deep-learning star convexed polygon detection (https://arxiv.org/abs/1806.03535)
- **Comseg** - Point cloud for transcript detection using KNN (https://www.nature.com/articles/s42003-024-06480-3)
- **QC report** — automated multi-page PDF comparing all methods, emailed on completion
  - Basic metrics: cell count, genes/cell, counts/cell, % transcripts captured
  - Morphological metrics: cell area, elongation, circularity, compactness, eccentricity, solidity, convexity, density, nuclear ratio
  - Segger contamination metrics (MECR)
  - All metrics computed from segmentation boundary geometry in Python (shapely)
  - Guide pages interleaved with data pages for interpretation context
- **Notifications** — email on job start, finish, and error; PDF for QC results attached on finish
- **Interactive wizard** — full config generation, sample discovery, job preview, and SLURM submission

**In progress:**
- Quick classifier from reference for cell type annotations needed for BIDCell, FastReseg integration (scripts present, validation ongoing)

## Supported Methods

| Method | Type | Status |
|--------|------|--------|
| **ProSeg** | Probabilistic (Rust, via SOPA) | ✓ Working |
| **Baysor** | Bayesian transcript-based (Julia, via SOPA) | ✓ Working |
| **Cellpose** | Neural network (Python, via SOPA) | ✓ Working |
| **FastReseg** | Post-hoc refinement (R) | ✓ Working |
| **Comseg** | KNN based point cloud | ✓ Working |
| **Stardist** | Deep learning nuclear segmentation | ✓ Working |
| **BIDCell** | Deep learning (PyTorch) | In progress |
| **Bering** | Graph based learning for transcript localization | In progress |
| **Segger** | Heteogenous graph neural network for link prediction | In progress |


## Quick Start

```bash
pip install pyyaml  # only dependency outside the container

# Interactive wizard — walks you through everything:
python segmentation_wizard.py

# Or use an existing config:
python segmentation_wizard.py --config config/my_config.yaml
```

## Output

Results are written into `{method}_reseg/` folders alongside your raw data. Raw data is never modified. Each completed sample produces:

| File | Description |
|------|-------------|
| `{sample_id}.h5ad` | AnnData count matrix |
| `cells.zarr.zip` | Cell boundaries for Xenium Explorer |
| `cell_feature_matrix.zarr.zip` | Count matrix for Explorer |
| `run_metadata_{method}.json` | Timing, parameters, SLURM job ID |
| `annotations.csv` | Annotated cell types learned from reference dataset |
| `confidence.csv` | Confidence in annotations |

QC output per slide:
- `qc_report.pdf` — 4-page comparative report (emailed automatically)
- `morpho_{method}.csv` — per-cell morphological metrics for all methods
- `cellspa_{method}.csv` — basic count-based QC summary

## Architecture

```
segmentation_wizard.py               ← interactive wizard (the only entry point)
        │
        ├── config/site.yaml         ← cluster settings, edited once per HPC
        ▼
config/pipeline_*.yaml               ← saved per-experiment configuration
        │
        ▼
scripts/slurm/generated/             ← one .sh per sample × method
        │
        ▼  (singularity exec container.sif python ...)
scripts/python/
  run_proseg.py / run_baysor.py / run_cellpose.py / run_bidcell.py / run_qc.py
        │
        ▼
scripts/utils/
  data_io.py        ← shared loading, patching, aggregation, export
  config_loader.py  ← config parsing, sample discovery, site settings
  notify_chain.py   ← chain start/finish email notifications
```

## Container

All segmentation methods run inside a single Singularity container (`container/Singularity_spatial_segmentation_v6`, built from `container/spatial_segmentation_env_v3.yml`). Contains Python 3.10, SOPA, ProSeg, Baysor, Cellpose, BIDCell, FastReseg, CellSPA, Segger, StarDist, Comseg, Bering, scanpy, spatialdata, PyTorch + CUDA, R + spatial packages.

## Running on a different cluster

Everything machine-specific lives in **`config/site.yaml`** — container runtime, GPU resource string, partition and account defaults, bind paths, and any `module load` lines needed before the container runs. It is the only file another site needs to edit; the analysis config stays untouched.

Every key has a built-in default, so a missing or partial `site.yaml` still works. Point at an alternate file with `SEGWIZ_SITE_CONFIG=/path/to/site.yaml`.

Typical port to an Apptainer + module-based cluster:

```yaml
container_cmd: "apptainer"
preamble:
  - "module load apptainer"
gpu_gres: "gpu:a100:1"        # check: sinfo -o "%P %G"
gpu_partition: "gpu-shared"
gpu_account: "your_project"
extra_binds: ["/scratch"]
```

`python_bin` points at the interpreter *inside* the container and must stay an absolute path — see the note in `site.yaml` for why bare `python` breaks.

## Requirements

- **Local:** Python 3.7+ with `pyyaml`
- **HPC:** SLURM, a container runtime (Singularity or Apptainer), the built `.sif`
- **Notifications:** `sendmail` available on the cluster

## Future Plans
- Additional segmentation method integrations
- Expanded QC metrics and spatial analysis
