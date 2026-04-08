"""
run_stardist.py — StarDist segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_stardist.py --config CONFIG --sample-dir /path/to/output-XETG... \
                           --output-dir /path/to/stardist_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import (
    configure_threads, configure_dask, load_platform_data,
    aggregate_and_save, save_run_metadata, timed,
)


def _patch_stardist_from_pretrained(model_type: str, local_model_dir: Path):
    """Monkeypatch StarDist2D.from_pretrained to load from a local directory.

    Sopa calls StarDist2D.from_pretrained(model_type) inside a dask worker closure.
    We can't intercept that via a kwarg — sopa's stardist() function doesn't expose
    a local-model option. Instead, patch the classmethod itself before sopa builds the
    closure. Fork-based dask workers inherit the patched class.
    """
    from stardist.models import StarDist2D

    _orig = StarDist2D.from_pretrained.__func__

    @classmethod
    def _local_from_pretrained(cls, name_or_alias, **kwargs):
        if str(name_or_alias) == model_type:
            print(f"[INFO] StarDist: loading '{name_or_alias}' from local path {local_model_dir}")
            return cls(None, name=local_model_dir.name, basedir=str(local_model_dir.parent))
        return _orig(cls, name_or_alias, **kwargs)

    StarDist2D.from_pretrained = _local_from_pretrained
    print(f"[INFO] Patched StarDist2D.from_pretrained → {local_model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run StarDist segmentation (single sample)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "stardist")
    params = method_cfg["params"]
    platform = cfg["data"]["platform"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = params.get("model_type", "2D_versatile_fluo")
    channels   = params.get("channels", ["DAPI"])

    # ── Locate local model ─────────────────────────────────────────────────────
    # Try explicit local_model param first, then auto-resolve from seg_models_path.
    # The extracted model lives at {seg_models_path}/models/StarDist2D/{model_type}/
    seg_models_path = cfg.get("data", {}).get("seg_models_path", "")
    local_model_dir: Path | None = None

    if params.get("local_model"):
        local_model_dir = Path(params["local_model"])
        print(f"[INFO] local_model from config: {local_model_dir}")
    elif seg_models_path:
        candidate = Path(seg_models_path) / "models" / "StarDist2D" / model_type
        print(f"[INFO] seg_models_path: {seg_models_path}")
        print(f"[INFO] Checking local model dir: {candidate} — exists={candidate.is_dir()}")
        if candidate.is_dir():
            local_model_dir = candidate

    if local_model_dir is None:
        print(f"[WARN] No local StarDist model found for '{model_type}' — "
              f"will attempt from_pretrained (requires internet, will fail on compute nodes)")
    else:
        # Patch from_pretrained BEFORE importing sopa so the closure it builds
        # already sees the patched version. Workers inherit via fork.
        # Must import stardist here (before sopa) since sopa also imports it.
        _patch_stardist_from_pretrained(model_type, local_model_dir)

    # Also set CSBDEEP_CACHE_DIR as belt-and-suspenders fallback — if the patch
    # somehow doesn't take in a worker, csbdeep at least looks in the right place.
    if seg_models_path:
        os.environ["CSBDEEP_CACHE_DIR"] = seg_models_path

    cpus = configure_threads()
    configure_dask(cpus)
    t_start = time.time()

    print("=" * 60)
    print(f"  StarDist — {args.sample_id}")
    print(f"  Model:  {model_type}")
    print(f"  Input:  {args.sample_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    sdata = load_platform_data(platform, args.sample_dir)

    import sopa
    sopa.segmentation.tissue(sdata)
    sopa.make_image_patches(
        sdata,
        patch_width=params.get("patch_width", 1200),
        patch_overlap=params.get("patch_overlap", 50),
    )
    sopa.settings.parallelization_backend = "dask"

    @timed("StarDist segmentation")
    def _run():
        sopa.segmentation.stardist(
            sdata,
            model_type=model_type,
            channels=channels,
            min_area=params.get("min_area", 0),
            prob_thresh=params.get("prob_thresh", 0.2),
            nms_thresh=params.get("nms_thresh", 0.6),
        )
    _run()

    sopa.make_transcript_patches(
        sdata,
        patch_width=params.get("transcript_patch_width", 500),
        prior_shapes_key="stardist_boundaries",
    )

    aggregate_and_save(
        sdata, output_dir, args.sample_id,
        explorer_mode=params.get("explorer_mode", "+cbm"),
        method="stardist",
    )

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "stardist", method_cfg, elapsed)
    print(f"[DONE] StarDist — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
