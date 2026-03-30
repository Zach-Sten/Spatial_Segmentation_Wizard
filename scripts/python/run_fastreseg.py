"""
run_fastreseg.py — FastReseg refinement for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_fastreseg.py --config CONFIG --sample-dir /path/to/output-XETG... \
                            --output-dir /path/to/fastreseg_reseg/XETG... --sample-id XETG... \
                            --source-dir /path/to/proseg_reseg/XETG...
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import configure_threads, save_run_metadata


FASTRESEG_R_SCRIPT = """
# FastReseg post-hoc refinement — auto-generated
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript run_fastreseg.R <source_dir> <output_dir> [ref_profiles_path]")

source_dir       <- args[1]
output_dir       <- args[2]
ref_profiles_csv <- if (length(args) >= 3 && nzchar(args[3])) args[3] else NULL

cat("FastReseg refinement\\n")
cat("Source:", source_dir, "\\n")
cat("Output:", output_dir, "\\n")

library(FastReseg)

h5ad_files <- list.files(source_dir, pattern = "\\\\.h5ad$", full.names = TRUE)
if (length(h5ad_files) == 0) stop("No .h5ad files found in source directory")
cat("Found:", h5ad_files[1], "\\n")

# Load reference profiles if provided
refProfiles <- NULL
if (!is.null(ref_profiles_csv) && file.exists(ref_profiles_csv)) {
    cat("[INFO] Loading ref profiles from:", ref_profiles_csv, "\\n")
    ref_df      <- read.csv(ref_profiles_csv, row.names = 1, check.names = FALSE)
    refProfiles <- as.matrix(ref_df)
    cat("[INFO] refProfiles:", nrow(refProfiles), "genes x", ncol(refProfiles), "cell types\\n")
}

# ── FastReseg pipeline ──
# result <- fastReseg_full_pipeline(
#   counts       = counts,        # spatial transcript counts matrix
#   clust        = NULL,          # use NULL when providing refProfiles
#   refProfiles  = refProfiles,   # reference expression profiles (genes x cell types)
#   score_baseline = score_baseline,  # from get_baselineCT() on preprocessed spatial data
#   lowerCutoff  = 0.5,
#   higherCutoff = 0.9,
# )

cat("[NOTE] FastReseg R script is a template. Edit for your data.\\n")
"""


def prepare_fastreseg_reference(
    reference_path: str,
    celltype_col: str,
    cache_dir: Path,
) -> Path:
    """Generate ref_profiles.csv (genes × cell types mean expression) for FastReseg.

    Saves the full gene set — FastReseg subsets to spatial panel genes at runtime
    by matching against the spatial counts matrix gene names.

    Returns path to the generated CSV.
    """
    import scanpy as sc
    from scipy import sparse

    profiles_csv = cache_dir / "ref_profiles.csv"

    if profiles_csv.exists():
        print(f"[INFO] FastReseg ref profiles found in cache: {cache_dir}")
        return profiles_csv

    print(f"[INFO] Generating FastReseg ref profiles from: {reference_path}")
    ref = sc.read_h5ad(reference_path)

    if celltype_col not in ref.obs.columns:
        raise ValueError(
            f"Column '{celltype_col}' not found in reference .obs. "
            f"Available: {list(ref.obs.columns)}"
        )

    X = ref.X.toarray() if sparse.issparse(ref.X) else np.array(ref.X)
    cell_types = ref.obs[celltype_col].astype(str)

    means = {}
    for ct in sorted(cell_types.unique()):
        mask = (cell_types == ct).values
        means[ct] = X[mask].mean(axis=0)

    profiles_df = pd.DataFrame(means, index=ref.var_names)
    cache_dir.mkdir(parents=True, exist_ok=True)
    profiles_df.to_csv(profiles_csv)
    print(
        f"[INFO] Saved FastReseg ref profiles: {profiles_csv} "
        f"({profiles_df.shape[0]} genes × {profiles_df.shape[1]} cell types)"
    )
    return profiles_csv


def main():
    parser = argparse.ArgumentParser(description="Run FastReseg (single sample)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-dir", required=True,
                        help="Path to the primary method's results to refine")
    parser.add_argument("--reference-path", default=None,
                        help="Path to reference h5ad for generating FastReseg refProfiles")
    parser.add_argument("--reference-celltype-col", default="cell_type",
                        help="Column in reference .obs with cell type labels")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "fastreseg")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir)

    # ── Generate reference profiles if not yet cached ──
    ref_profiles_path = ""
    if args.reference_path:
        cache_dir = Path(args.reference_path).parent / "reference_cache"
        try:
            ref_profiles_path = str(prepare_fastreseg_reference(
                args.reference_path, args.reference_celltype_col, cache_dir
            ))
        except Exception as e:
            print(f"[WARN] Could not prepare FastReseg reference profiles: {e}")

    configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  FastReseg — {args.sample_id}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    if not source_dir.exists():
        print(f"[ERROR] Source not found: {source_dir}")
        sys.exit(1)

    r_script_path = output_dir / "run_fastreseg.R"
    with open(r_script_path, "w") as f:
        f.write(FASTRESEG_R_SCRIPT)

    result = subprocess.run(
        ["Rscript", str(r_script_path), str(source_dir), str(output_dir), ref_profiles_path],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] R failed:\n{result.stderr}")
        sys.exit(1)

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "fastreseg", method_cfg, elapsed)
    print(f"[DONE] FastReseg — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
