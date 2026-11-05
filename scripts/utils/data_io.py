"""
data_io.py — Shared utilities for the per-method runner scripts.

Data loading, saving, export, thread setup, and the common CLI contract.
"""

import os
import time
import argparse
import warnings
from pathlib import Path
from functools import wraps

import numpy as np


# ── Runner CLI ──
def runner_parser(description: str) -> argparse.ArgumentParser:
    """Parser with the four arguments every runner receives from its SLURM script.

    Runners needing extra flags call add_argument() on the result.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config",     required=True, help="Path to pipeline config YAML")
    p.add_argument("--sample-dir", required=True, help="Raw sample directory (input)")
    p.add_argument("--output-dir", required=True, help="Directory to write results to")
    p.add_argument("--sample-id",  required=True, help="Sample identifier")
    return p


# ── Timing decorator ──
def timed(step_name: str):
    """Decorator to log timing of pipeline steps."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[INFO] Starting: {step_name}")
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed = (time.time() - t0) / 60
            print(f"[INFO] Completed: {step_name} ({elapsed:.1f} min)")
            return result
        return wrapper
    return decorator


# ── Thread/CPU configuration ──
def configure_threads(cpus: int = None):
    """Set thread counts from SLURM env or explicit value. Call BEFORE heavy imports."""
    if cpus is None:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

    os.environ["OMP_NUM_THREADS"] = str(cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
    os.environ["MKL_NUM_THREADS"] = str(cpus)

    # Suppress noisy-but-harmless warnings in dask worker subprocesses (inherited via env).
    # Append so we don't clobber any pre-existing PYTHONWARNINGS value.
    _extra_filters = (
        "ignore::FutureWarning:dask,"
        "ignore::UserWarning:xarray_schema,"
        "ignore::UserWarning:spatialdata_io,"
        "ignore::UserWarning:pkg_resources"
    )
    existing_pw = os.environ.get("PYTHONWARNINGS", "")
    os.environ["PYTHONWARNINGS"] = f"{existing_pw},{_extra_filters}".lstrip(",")

    print(f"[INFO] Thread count set to {cpus}")
    return cpus


def configure_dask(cpus: int):
    """Configure Dask for sopa/Baysor parallelization.

    sopa's Baysor backend uses dask.distributed (LocalCluster with worker
    processes). Do NOT set a ThreadPool here — pool objects cannot be pickled
    across processes and will cause all distributed workers to fail at startup.
    """
    import dask

    dask.config.set({
        "distributed.worker.nthreads": 1,
        "dataframe.query-planning": True,
        "array.rechunk.method": "tasks",
    })
    print(f"[INFO] Dask configured: distributed backend, {cpus} CPUs available")


# ── Data loading ──
def inspect_morphology(raw_data_path: str) -> dict:
    """List the morphology_focus files of a Xenium run (no heavy imports).

    Returns {"dapi": Path|None, "legacy": [paths], "channels": [(idx, name, path)],
             "protein": [(idx, name, path)]}.
    Standard runs have 1 (no seg kit) or 4 (seg kit) legacy morphology_focus_*.ome.tif.
    Proteomics-kit runs add one ch####_<name>.ome.tif per channel (DAPI, 18S,
    backgrounds, then antibodies); the antibody channels are returned as "protein".
    """
    import re
    morph_dir = Path(raw_data_path) / "morphology_focus"
    legacy = sorted(morph_dir.glob("morphology_focus_*.ome.tif"))
    chans = []
    for f in sorted(morph_dir.glob("ch*_*.ome.tif")):
        m = re.match(r"ch(\d+)_(.+)\.ome\.tif$", f.name)
        if m:
            chans.append((int(m.group(1)), m.group(2), f))
    non_protein = ("dapi", "18s", "background")
    protein = [c for c in chans if not any(t in c[1].lower() for t in non_protein)]
    dapi = next((c[2] for c in chans if c[1].lower() == "dapi"), None)
    return {"dapi": dapi, "legacy": legacy, "channels": chans, "protein": protein}


def extract_protein_requested() -> bool:
    return os.environ.get("SEGWIZ_EXTRACT_PROTEIN", "").lower() in ("1", "true", "yes")


@timed("Load spatial data")
def load_xenium_data(raw_data_path: str):
    """Load Xenium data via spatialdata-io.

    Proteomics-kit outputs store one OME-TIFF per channel (ch0000_dapi.ome.tif, ...)
    alongside (or instead of) the legacy morphology_focus_*.ome.tif, and
    spatialdata-io raises when the file count is not 1 or 4. For those we skip its
    morphology reader and attach the images ourselves: DAPI as "morphology_focus",
    and, if SEGWIZ_EXTRACT_PROTEIN=1 (data.extract_protein in the config), the
    antibody channels as a multi-channel "protein_focus" image.
    """
    from spatialdata_io import xenium
    info = inspect_morphology(raw_data_path)
    dapi = info["dapi"]
    # any per-channel ch####_*.ome.tif files make spatialdata-io count the folder wrong
    new_style = dapi is not None

    sdata = xenium(raw_data_path, cells_as_circles=True, morphology_focus=not new_style)

    if new_style:
        import dask.array as da
        import tifffile
        from spatialdata.models import Image2DModel
        from spatialdata.transformations import Identity

        def _add(key, files, names):
            # key=0 reads only the full-resolution plane, not the multi-file OME series
            planes = [tifffile.imread(f, key=0) for f in files]
            sdata.images[key] = Image2DModel.parse(
                da.from_array(np.stack(planes), chunks=(1, 2048, 2048)),
                dims=("c", "y", "x"), c_coords=names,
                transformations={"global": Identity()},
            )

        _add("morphology_focus", [dapi], ["DAPI"])
        if extract_protein_requested() and info["protein"]:
            _add("protein_focus", [c[2] for c in info["protein"]], [c[1] for c in info["protein"]])
            print(f"[INFO] Extracted {len(info['protein'])} protein channels")
    print(f"[INFO] Loaded: {list(sdata.images.keys())} images, "
          f"{len(sdata.points) if hasattr(sdata, 'points') else '?'} point tables")
    return sdata


def load_platform_data(platform: str, raw_data_path: str):
    """Load data based on platform type."""
    loaders = {
        "xenium": load_xenium_data,
    }
    if platform not in loaders:
        raise ValueError(f"Unsupported platform: {platform}. Supported: {list(loaders.keys())}")
    return loaders[platform](raw_data_path)


# ── Patch preparation ──
@timed("Prepare patches")
def prepare_patches(sdata, image_patch_width=1200, image_patch_overlap=10,
                    transcript_patch_width=None, prior_shapes_key="cell_boundaries"):
    """Create image and transcript patches for segmentation."""
    import sopa

    sopa.segmentation.tissue(sdata)
    sopa.make_image_patches(sdata, patch_width=image_patch_width, patch_overlap=image_patch_overlap)

    kwargs = {"patch_width": transcript_patch_width}
    if prior_shapes_key:
        kwargs["prior_shapes_key"] = prior_shapes_key
    sopa.make_transcript_patches(sdata, **kwargs)

    n_img = len(sdata.shapes.get("image_patches", []))
    n_tx = len(sdata.shapes.get("transcripts_patches", []))
    print(f"[INFO] Patches ready — {n_img} image patches, {n_tx} transcript patches")
    return sdata


# ── Aggregation + export ──
@timed("Aggregate counts")
def aggregate_and_save(sdata, output_dir: Path, sample_id: str,
                       explorer_mode: str = "+cbm", method: str = None):
    """Aggregate counts, save h5ad, and export for Explorer."""
    import sopa

    sopa.aggregate(sdata)
    adata = sdata["table"]

    # Save h5ad
    h5ad_path = output_dir / f"{sample_id}.h5ad"
    adata.write_h5ad(h5ad_path)
    print(f"[INFO] H5AD saved: {h5ad_path}")

    # Export cell boundaries for downstream QC morphological metrics.
    # Prefer the method-specific key (e.g. proseg_boundaries, baysor_boundaries)
    # over the original Xenium cell_boundaries loaded with the raw data.
    all_keys = [k for k in sdata.shapes if "boundaries" in k and "patch" not in k]
    print(f"[INFO] Boundary shape keys available: {all_keys}")
    boundary_key = None
    if method:
        boundary_key = next((k for k in all_keys if method in k), None)
    if boundary_key is None:
        boundary_key = next((k for k in all_keys if k != "cell_boundaries"), None)
    if boundary_key is None:
        boundary_key = next(iter(all_keys), None)
    if boundary_key:
        try:
            boundary_path = output_dir / "cell_boundaries.parquet"
            sdata.shapes[boundary_key].to_parquet(boundary_path)
            print(f"[INFO] Cell boundaries saved ({boundary_key}): {boundary_path}")
        except Exception as e:
            print(f"[WARN] Could not save cell boundaries: {e}")

    # Explorer export
    sopa.io.explorer.write(
        str(output_dir),
        sdata,
        mode=explorer_mode,
        save_h5ad=False,
    )
    print(f"[INFO] Explorer export complete: {output_dir}")

    return adata


@timed("Replace counts with ProSeg expected counts")
def replace_proseg_counts(adata, expected_counts_path: str):
    """Replace aggregated counts with ProSeg's expected_counts.mtx.gz."""
    from scipy.io import mmread

    C = mmread(expected_counts_path)
    if C.shape == (adata.n_vars, adata.n_obs):
        C = C.T
    C = C.tocsr()

    # Sopa may filter cells after aggregation (e.g. low channel intensity),
    # leaving adata with fewer rows than the full ProSeg expected counts matrix.
    # Align using _proseg_idx stored in obs before filtering.
    if C.shape[0] != adata.n_obs:
        if "_proseg_idx" not in adata.obs.columns:
            raise ValueError(
                f"Expected counts shape {C.shape} doesn't match adata "
                f"({adata.n_obs}, {adata.n_vars}) and no _proseg_idx column found in obs. "
                f"Re-run with the latest run_proseg.py to generate this index."
            )
        cell_idx = adata.obs["_proseg_idx"].values
        n_original = C.shape[0]
        C = C[cell_idx]
        print(f"[INFO] Aligned expected counts: {n_original} → {adata.n_obs} cells "
              f"({n_original - adata.n_obs} filtered by sopa)")

    adata.layers["expected_counts"] = C.copy()

    C.data = np.rint(C.data)
    C = C.astype(np.int32)

    assert C.shape == (adata.n_obs, adata.n_vars), \
        f"Shape mismatch: counts {C.shape} vs adata ({adata.n_obs}, {adata.n_vars})"
    assert C.min() >= 0, "Negative counts detected"

    adata.layers["counts"] = C
    adata.X = C

    print(f"[INFO] Replaced counts: {C.shape[0]} cells × {C.shape[1]} genes")
    return adata


# ── Metadata logging ──
def save_run_metadata(output_dir: Path, method: str, config: dict, elapsed_seconds: float):
    """Save a JSON file with run parameters and timing for reproducibility."""
    import json
    from datetime import datetime

    meta = {
        "method": method,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_minutes": round(elapsed_seconds / 60, 1),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK", "unknown"),
        "config": config,
    }

    meta_path = output_dir / f"run_metadata_{method}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[INFO] Run metadata saved: {meta_path}")
