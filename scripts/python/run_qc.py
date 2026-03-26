"""
run_qc.py — CellSPA QC across all completed segmentation results for a single sample.

Auto-discovers completed methods by scanning *_reseg/ dirs for h5ad files.
Runs CellSPA (R) for reference-free metrics + Python plots per method.

Called by the generated SLURM script:
    python run_qc.py --config CONFIG --sample-id XETG... --slide-dir /path/to/slide_folder
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

from utils.config_loader import load_config, get_method_config, get_output_base_override
from utils.data_io import configure_threads, save_run_metadata, timed


# ── CellSPA R script (reference-free: spatial + basic QC metrics) ──────────────
# Reads pre-exported CSV/MTX files from Python — no zellkonverter needed.
CELLSPA_R_SCRIPT = """\
suppressPackageStartupMessages({
    library(CellSPA)
    library(SpatialExperiment)
    library(SingleCellExperiment)
    library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[1]
coords_path <- args[2]
method_name <- args[3]
output_dir  <- args[4]

cat(sprintf("[INFO] CellSPA QC: %s\\n", method_name))

# Load pre-exported counts (genes x cells MTX) and coordinates
counts <- readMM(counts_path)
coords_df <- read.csv(coords_path)
coords <- as.matrix(coords_df[, c("x", "y")])

cat(sprintf("[INFO] Loaded: %d cells x %d genes\\n", ncol(counts), nrow(counts)))

spe <- SpatialExperiment(
    assays        = list(counts = counts),
    spatialCoords = coords
)

# Basic filtering
spe <- tryCatch(processingSPE(spe), error = function(e) {
    cat(sprintf("[WARN] processingSPE: %s\\n", e$message)); spe
})
cat(sprintf("[INFO] After filtering: %d cells\\n", ncol(spe)))

# Spatial diversity metrics (reference-free)
spatial_metrics <- tryCatch(
    calSpatialMetricsDiversity(spe),
    error = function(e) { cat(sprintf("[WARN] Spatial metrics: %s\\n", e$message)); NULL }
)

# Build summary row
summary_df <- data.frame(
    method         = method_name,
    n_cells        = ncol(spe),
    n_genes        = nrow(spe),
    median_counts  = median(colSums(counts(spe))),
    median_genes   = median(colSums(counts(spe) > 0)),
    stringsAsFactors = FALSE
)

if (!is.null(spatial_metrics) && is.data.frame(spatial_metrics)) {
    for (col in colnames(spatial_metrics)) {
        summary_df[[col]] <- mean(spatial_metrics[[col]], na.rm = TRUE)
    }
}

out_path <- file.path(output_dir, sprintf("cellspa_%s.csv", method_name))
write.csv(summary_df, out_path, row.names = FALSE)
cat(sprintf("[INFO] Saved: %s\\n", out_path))
print(summary_df)
"""


def export_for_r(adata, method: str, qc_dir: Path) -> tuple:
    """Export counts (MTX) and spatial coords (CSV) for the R script. Returns (counts_path, coords_path)."""
    from scipy.io import mmwrite
    import scipy.sparse as sp

    counts_path = qc_dir / f"counts_{method}.mtx"
    coords_path = qc_dir / f"coords_{method}.csv"

    # Write genes x cells MTX
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    mmwrite(str(counts_path), X.T)

    # Extract spatial coords — sopa stores them in obsm['spatial']
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"][:, :2]
    else:
        xy_cols = [c for c in adata.obs.columns if c in ("x", "y", "spatial_x", "spatial_y")]
        if len(xy_cols) >= 2:
            coords = adata.obs[xy_cols[:2]].values
        else:
            coords = None

    if coords is not None:
        pd.DataFrame(coords, columns=["x", "y"]).to_csv(coords_path, index=False)
    else:
        print(f"[WARN] No spatial coordinates found for {method} — spatial metrics will be skipped")
        coords_path = None

    return counts_path, coords_path


def run_cellspa(adata, method: str, qc_dir: Path) -> bool:
    """Export data, write and run the CellSPA R script for one method. Returns True on success."""
    counts_path, coords_path = export_for_r(adata, method, qc_dir)
    if coords_path is None:
        return False

    r_script = qc_dir / f"run_cellspa_{method}.R"
    r_script.write_text(CELLSPA_R_SCRIPT)

    result = subprocess.run(
        ["Rscript", str(r_script), str(counts_path), str(coords_path), method, str(qc_dir)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] CellSPA R failed for {method}:\n{result.stderr}")
        return False
    return True


@timed("Generate QC plots")
def generate_qc_plots(adata, method_name: str, output_dir: Path):
    """Generate violin + scatter QC plots."""
    import scanpy as sc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"QC — {method_name}", fontsize=14)
    for ax, metric, label in zip(
        axes,
        ["n_genes_by_counts", "total_counts", "pct_counts_in_top_50_genes"],
        ["Genes per Cell", "Total Counts", "% in Top 50 Genes"],
    ):
        if metric in adata.obs.columns:
            ax.violinplot(adata.obs[metric].values, showmedians=True)
            ax.set_title(label)
            ax.set_ylabel(label)
    plt.tight_layout()
    fig.savefig(output_dir / f"qc_violin_{method_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(adata.obs["total_counts"], adata.obs["n_genes_by_counts"], s=1, alpha=0.3, c="steelblue")
    ax.set_xlabel("Total Counts")
    ax.set_ylabel("Genes Detected")
    ax.set_title(f"{method_name}: Counts vs Genes")
    fig.savefig(output_dir / f"qc_scatter_{method_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def discover_completed_methods(slide_dir: Path, sample_id: str, output_base: str) -> dict:
    """Scan *_reseg/ dirs and return {method: h5ad_path} for those with results."""
    base = Path(output_base) / slide_dir.name if output_base else slide_dir
    found = {}
    for reseg_dir in sorted(base.glob("*_reseg")):
        method = reseg_dir.name.replace("_reseg", "")
        sample_dir = reseg_dir / sample_id
        h5ads = list(sample_dir.glob("*.h5ad")) if sample_dir.exists() else []
        if h5ads:
            found[method] = h5ads[0]
    return found


def main():
    parser = argparse.ArgumentParser(description="CellSPA QC — all completed segmentation methods")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--slide-dir", required=True, help="Slide folder containing {method}_reseg/ dirs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "cellspa_qc")
    output_base = get_output_base_override(cfg)

    slide_dir = Path(args.slide_dir)
    qc_dir = (Path(output_base) / slide_dir.name if output_base else slide_dir) / "qc" / args.sample_id
    qc_dir.mkdir(parents=True, exist_ok=True)

    configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  CellSPA QC — {args.sample_id}")
    print("=" * 60)

    # Auto-discover completed methods
    discovered = discover_completed_methods(slide_dir, args.sample_id, output_base)
    if not discovered:
        print(f"[WARN] No completed segmentation results found under {slide_dir}")
        elapsed = time.time() - t_start
        save_run_metadata(qc_dir, "qc", method_cfg, elapsed)
        sys.exit(0)

    print(f"[INFO] Found results for: {', '.join(discovered.keys())}\n")

    import scanpy as sc
    cellspa_results = []

    for method, h5ad_path in discovered.items():
        print(f"\n── {method} ──")
        adata = sc.read_h5ad(h5ad_path)
        print(f"[INFO] {adata.n_obs} cells × {adata.n_vars} genes")

        # CellSPA R metrics
        @timed(f"CellSPA metrics: {method}")
        def _cellspa():
            return run_cellspa(adata, method, qc_dir)
        success = _cellspa()

        if success:
            csv = qc_dir / f"cellspa_{method}.csv"
            if csv.exists():
                cellspa_results.append(pd.read_csv(csv))

        # Python QC plots
        generate_qc_plots(adata, method, qc_dir)

    # Combined CellSPA comparison table
    if cellspa_results:
        comparison = pd.concat(cellspa_results, ignore_index=True)
        comparison.to_csv(qc_dir / "cellspa_comparison.csv", index=False)
        print(f"\n── CellSPA Comparison ──")
        print(comparison.to_string(index=False))

    elapsed = time.time() - t_start
    save_run_metadata(qc_dir, "qc", method_cfg, elapsed)
    print(f"\n[DONE] CellSPA QC — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
