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
CELLSPA_R_SCRIPT = """\
suppressPackageStartupMessages({
    library(CellSPA)
    library(zellkonverter)
    library(SpatialExperiment)
    library(SingleCellExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
h5ad_path  <- args[1]
method_name <- args[2]
output_dir  <- args[3]

cat(sprintf("[INFO] CellSPA QC: %s\\n", method_name))
cat(sprintf("[INFO] Reading: %s\\n", h5ad_path))

sce <- readH5AD(h5ad_path, verbose = FALSE)
cat(sprintf("[INFO] Loaded: %d cells x %d genes\\n", ncol(sce), nrow(sce)))

# Extract spatial coordinates (sopa writes them to obsm['spatial'])
coords <- NULL
if ("spatial" %in% reducedDimNames(sce)) {
    coords <- reducedDim(sce, "spatial")
} else {
    cd <- as.data.frame(colData(sce))
    xy_cols <- intersect(c("x", "y", "spatial_x", "spatial_y"), colnames(cd))
    if (length(xy_cols) >= 2) coords <- as.matrix(cd[, xy_cols[1:2]])
}

if (is.null(coords)) {
    cat("[WARN] No spatial coordinates found — cannot build SpatialExperiment\\n")
    quit(status = 0)
}
colnames(coords) <- c("x", "y")

spe <- SpatialExperiment(
    assays      = list(counts = counts(sce)),
    colData     = colData(sce),
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


def run_cellspa(h5ad_path: Path, method: str, qc_dir: Path) -> bool:
    """Write and run the CellSPA R script for one method. Returns True on success."""
    r_script = qc_dir / f"run_cellspa_{method}.R"
    r_script.write_text(CELLSPA_R_SCRIPT)

    result = subprocess.run(
        ["Rscript", str(r_script), str(h5ad_path), method, str(qc_dir)],
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
            return run_cellspa(h5ad_path, method, qc_dir)
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
