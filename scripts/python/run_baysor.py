"""
run_baysor.py — Baysor segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_baysor.py --config CONFIG --sample-dir /path/to/output-XETG... \
                         --output-dir /path/to/baysor_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import stat
import shutil
import tempfile
import time
import asyncio
import warnings
from pathlib import Path

# Suppress noisy-but-harmless warnings from dependencies (main process)
warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings("ignore", category=UserWarning, module="spatialdata_io")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import (
    runner_parser,
    configure_threads, configure_dask, load_platform_data,
    prepare_patches, aggregate_and_save, save_run_metadata, timed,
)


class TooManyFailedPatches(RuntimeError):
    pass


def _retry_failed_patches(sdata, failed, baysor_kwargs, params):
    """
    Re-run failed patches with a slightly perturbed Baysor scale, then re-resolve.

    Baysor 0.7.1's loom writer crashes when #cells == #genes in a patch
    (inverted length check). Changing the scale changes the cell count, which
    sidesteps the coincidence. Returns the patches that still have no loom.
    """
    import toml
    from functools import partial
    from sopa import settings
    from sopa.segmentation.methods._baysor import BaysorPatch, _get_baysor_command
    from sopa.segmentation.methods import _baysor as _bm
    from sopa.constants import SopaFiles, SopaKeys

    prior_key = SopaKeys.SOPA_PRIOR if SopaKeys.PRIOR_SHAPES_KEY in sdata.shapes[SopaKeys.TRANSCRIPTS_PATCHES] else None
    cmd = _get_baysor_command(prior_key)
    base_scale = baysor_kwargs.get("scale", 15)
    remaining = [Path(d) for d in failed]

    for delta in params.get("retry_scale_deltas", [0.5, -0.5, 1.0]):
        if not remaining:
            break
        print(f"[INFO] Retrying {len(remaining)} failed patch(es) with scale={base_scale + delta}")
        funcs = []
        for d in remaining:
            cfg = toml.load(d / SopaFiles.TOML_CONFIG_FILE)
            cfg.setdefault("segmentation", {})["scale"] = base_scale + delta
            funcs.append(partial(BaysorPatch(cmd, cfg, force=True, capture_output=True), d))
        settings._run_with_backend(funcs)
        remaining = [d for d in remaining if not (d / "segmentation_counts.loom").exists()]

    if len(remaining) < len(failed):
        # Re-resolve all patches now that more have output.
        from sopa.utils import get_transcripts_patches_dirs
        dirs = [Path(d) for d in get_transcripts_patches_dirs(sdata)]
        ok = [d for d in dirs if (d / "segmentation_counts.loom").exists()]
        gene_col = toml.load(ok[0] / SopaFiles.TOML_CONFIG_FILE)["data"]["gene"]
        _bm.resolve(sdata, ok, gene_column=gene_col, min_area=params.get("min_area", 10),
                    key_added=SopaKeys.BAYSOR_BOUNDARIES)
        _bm.set_boundaries_attrs(sdata, SopaKeys.BAYSOR_BOUNDARIES)
    return remaining


def _install_baysor_wrapper(error_log: str):
    """
    Wrap the baysor binary so its stdout/stderr surface in a persistent error log.

    Dask worker stderr is not forwarded to the SLURM .err file, so writing to
    stderr is silent.  Instead, the wrapper appends all failed-patch output to
    a known file (error_log) that we can inspect after the job finishes or fails.
    """
    real_baysor = shutil.which("baysor")
    if not real_baysor:
        print("[WARN] baysor not found in PATH — skipping output wrapper", file=sys.stderr)
        return

    wrapper_dir = tempfile.mkdtemp(prefix="baysor_wrapper_")
    wrapper = os.path.join(wrapper_dir, "baysor")
    with open(wrapper, "w") as f:
        f.write(f"""#!/bin/bash
# Auto-generated wrapper — appends failed Baysor patch output to a persistent log
LOG=$(mktemp /tmp/baysor_XXXXXX.log)
"{real_baysor}" "$@" >"$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "[BAYSOR ERROR] exit $RC  pwd=$(pwd)  args=$@" >> "{error_log}"
    cat "$LOG" >> "{error_log}"
    echo "---" >> "{error_log}"
fi
rm -f "$LOG"
exit $RC
""")
    os.chmod(wrapper, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    os.environ["PATH"] = wrapper_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"[INFO] Baysor output wrapper installed (real binary: {real_baysor})")
    print(f"[INFO] Baysor error log: {error_log}")


def main():
    args = runner_parser("Run Baysor segmentation (single sample)").parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "baysor")
    params = method_cfg["params"]
    platform = cfg["data"]["platform"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = str(output_dir / "baysor_errors.log")
    _install_baysor_wrapper(error_log)

    cpus = configure_threads()
    if params.get("parallelization_backend") == "dask":
        configure_dask(cpus)

    t_start = time.time()

    print("=" * 60)
    print(f"  Baysor — {args.sample_id}")
    print(f"  Input:  {args.sample_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    sdata = load_platform_data(platform, args.sample_dir)

    # Rechunk large labels
    if hasattr(sdata, "labels") and "cell_labels" in sdata.labels:
        try:
            lbl = sdata.labels["cell_labels"]
            sdata.labels["cell_labels"] = lbl.copy(data=lbl.data.rechunk((2048, 2048)))
        except Exception:
            pass

    import sopa
    # sopa caches un-backed sdata patches in ~/.sopa_cache (home quota, small).
    # Redirect to a dir on the data volume. Config: baysor params.cache_dir,
    # default <output_dir>/.sopa_cache.
    import sopa.utils.utils as _sopa_utils
    cache_root = Path(params.get("cache_dir") or output_dir / ".sopa_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    _sopa_utils.HOME_CACHE_DIR = cache_root
    print(f"[INFO] sopa patch cache: {cache_root}")

    if params.get("parallelization_backend") == "dask":
        sopa.settings.parallelization_backend = "dask"

    patch_width         = params.get("patch_width", 500)
    patch_correction    = params.get("patch_error_correction", False)
    patch_correction_max  = params.get("patch_correction_max", 1500)
    patch_correction_step = params.get("patch_correction_step", 100)

    # Image patches only need to be prepared once — Baysor doesn't use them
    prepare_patches(
        sdata,
        image_patch_width=params.get("image_patch_width", 1200),
        image_patch_overlap=params.get("patch_overlap", 10),
        transcript_patch_width=patch_width,
        prior_shapes_key=params.get("prior_shapes_key", None),
    )

    attempt = 0
    while True:
        attempt += 1
        if attempt > 1:
            print(f"[INFO] Retry attempt {attempt} — re-preparing transcript patches at {patch_width}µm")
            sopa.make_transcript_patches(
                sdata,
                patch_width=patch_width,
                **( {"prior_shapes_key": params["prior_shapes_key"]}
                    if params.get("prior_shapes_key") else {} ),
            )

        n_tiles = len(sdata.shapes.get("transcripts_patches", []))
        print(f"[INFO] Running Baysor on {n_tiles} transcript patches (patch_width={patch_width}µm)...")

        @timed("Baysor segmentation")
        def _run():
            try:
                baysor_kwargs = {"min_area": params.get("min_area", 10), "scale": params.get("scale", 15)}
                # force=True: a patch whose Baysor run fails (e.g. the Baysor 0.7.1
                # loom-writer bug) is skipped instead of aborting all patches.
                # Cache is kept so we can count failures, then deleted below.
                sopa.segmentation.baysor(sdata, force=True, delete_cache=False, **baysor_kwargs)

                from sopa.utils import get_transcripts_patches_dirs
                dirs = get_transcripts_patches_dirs(sdata)
                failed = [d for d in dirs if not (Path(d) / "segmentation_counts.loom").exists()]
                if failed:
                    failed = _retry_failed_patches(sdata, failed, baysor_kwargs, params)
                max_frac = params.get("max_failed_patch_fraction", 0.02)
                print(f"[INFO] Baysor patches failed: {len(failed)} / {len(dirs)} (limit {max_frac:.0%})")
                for d in failed[:50]:
                    print(f"[WARN] failed patch: {d}")
                if len(failed) > max_frac * len(dirs):
                    raise TooManyFailedPatches(
                        f"{len(failed)}/{len(dirs)} Baysor patches failed (> {max_frac:.0%}); "
                        "see baysor_errors.log")
                try:
                    from sopa.utils import delete_transcripts_patches_dirs
                    delete_transcripts_patches_dirs(sdata)
                except Exception as ce:
                    print(f"[WARN] Could not delete patch cache: {ce}")
            except TooManyFailedPatches:
                raise
            except Exception as e:
                print(f"[WARN] Baysor dask run failed ({type(e).__name__}: {e})")
                print("[WARN] Attempting to resolve completed patches from disk...")

                try:
                    from sopa.segmentation.transcripts import resolve
                except ImportError:
                    from sopa.segmentation.methods._baysor import resolve

                from sopa.utils import get_transcripts_patches_dirs
                all_dirs = get_transcripts_patches_dirs(sdata)

                def _patch_complete(p):
                    p = Path(p)
                    if not (p / "segmentation_counts.loom").exists():
                        return False
                    return (p / "segmentation_polygons.json").exists() or \
                           (p / "segmentation.json").exists()

                completed = [p for p in all_dirs if _patch_complete(p)]
                print(f"[INFO] Completed patches on disk: {len(completed)} / {len(all_dirs)}")

                if not completed:
                    raise RuntimeError(
                        "No completed Baysor patches found on disk — cannot recover. "
                        "Check baysor logs for the root cause."
                    ) from e

                min_area = params.get("min_area", 10)
                resolve(sdata, completed, min_area=min_area, key_added="baysor_boundaries")
                print(f"[INFO] Resolved from {len(completed)} patch(es) — baysor_boundaries added to sdata")

        try:
            _run()
            break  # success
        except RuntimeError as e:
            if not patch_correction:
                raise
            next_width = patch_width + patch_correction_step
            if next_width > patch_correction_max:
                print(f"[ERROR] Baysor failed at {patch_width}µm and reached max ({patch_correction_max}µm) — giving up.")
                raise
            print(f"[WARN] Baysor failed at {patch_width}µm — retrying at {next_width}µm")
            patch_width = next_width

    aggregate_and_save(
        sdata, output_dir, args.sample_id,
        explorer_mode=params.get("explorer_mode", "+cbm"),
        method="baysor",
    )

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "baysor", method_cfg, elapsed)
    print(f"[DONE] Baysor — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
