"""
run_rough_annotation_classifer.py — Rank-gene XGBoost cell type classifier.

Trains an XGBoost classifier on rank-transformed gene expression from a labeled
reference h5ad, then predicts cell type labels on newly segmented data.

The rank transformation mirrors Geneformer's approach: within each cell, genes
are ranked by expression (rank 1 = most highly expressed, rank 0 = not detected).
This makes the feature space comparable across cells with different total counts
without requiring normalization.

Usage:
    python run_rough_annotation_classifer.py \
        --reference /path/to/reference.h5ad \
        --celltype-col cell_type \
        --query /path/to/segmented.h5ad \
        --output-dir /path/to/output \
        --sample-id XETG... \
        [--gpu]
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy import sparse
from scipy.stats import rankdata as _rankdata
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

warnings.filterwarnings("ignore")


# ── Rank transformation ──────────────────────────────────────────────────────

def counts_to_rank(adata, desc="Ranking genes"):
    """
    Convert a counts matrix to within-cell gene ranks.

    For each cell, genes are ranked by expression value (rank 1 = highest).
    Genes with zero counts receive rank 0 and contribute no signal.
    Uses adata.layers['counts'] if present, otherwise adata.X.

    Returns a dense float32 array of shape (n_cells, n_genes).
    """
    if "counts" in adata.layers:
        X = adata.layers["counts"]
    else:
        X = adata.X

    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    ranks = np.zeros_like(X, dtype=np.float32)
    for i in tqdm(range(X.shape[0]), desc=f"  {desc}", leave=False):
        row = X[i]
        nonzero = row > 0
        if nonzero.any():
            # Dense rank descending: highest expression = rank 1
            ranks[i, nonzero] = _rankdata(-row[nonzero], method="dense").astype(np.float32)

    return ranks


# ── Gene alignment ───────────────────────────────────────────────────────────

def align_genes(ref, query):
    """
    Subset both AnnData objects to their shared genes in the same order.

    Returns (ref_subset, query_subset). Raises if there is no overlap,
    warns if overlap is suspiciously low (likely a gene name format mismatch).
    """
    common = ref.var_names.intersection(query.var_names)
    n_common = len(common)
    if n_common == 0:
        raise ValueError(
            "Reference and query share no gene names. "
            "Check that both use the same format (gene symbols vs Ensembl IDs)."
        )
    n_ref, n_q = len(ref.var_names), len(query.var_names)
    pct = 100 * n_common / min(n_ref, n_q)
    print(f"[INFO] Gene alignment: {n_ref} ref × {n_q} query → {n_common} shared ({pct:.0f}%)")
    if pct < 50:
        print("[WARN] Fewer than 50% of genes overlap — check gene name format.")
    return ref[:, common].copy(), query[:, common].copy()


# ── Classifier ───────────────────────────────────────────────────────────────

def train_classifier(X_train, y_train, use_gpu=False):
    """
    Encode labels and train XGBoost on rank-transformed reference data.

    Runs a quick 3-fold cross-validation for a sanity-check accuracy estimate,
    then fits the final model on all reference cells.

    Returns (clf, label_encoder).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    sample_weights = compute_sample_weight("balanced", y_enc)

    xgb_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        device="cuda" if use_gpu else "cpu",
        random_state=42,
    )

    # Cross-val sanity check
    print("[INFO] Running 3-fold cross-validation on reference data...")
    cv_scores = cross_val_score(
        xgb.XGBClassifier(**xgb_params),
        X_train, y_enc,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="balanced_accuracy",
        n_jobs=1,  # XGBoost already parallelizes; avoid nested parallelism
    )
    print(f"[INFO] CV balanced accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit on full reference
    print("[INFO] Fitting final classifier on full reference data...")
    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X_train, y_enc, sample_weight=sample_weights, verbose=False)
    print("[INFO] Training complete.")

    return clf, le


def predict_labels(clf, le, X_query):
    """
    Predict cell type labels and per-cell confidence score.

    Confidence = max class probability from predict_proba, so it reflects
    how decisive the classifier was, not just whether it was right.

    Returns (labels: ndarray[str], confidence: ndarray[float32]).
    """
    proba = clf.predict_proba(X_query)
    pred_enc = np.argmax(proba, axis=1)
    confidence = proba.max(axis=1).astype(np.float32)
    labels = le.inverse_transform(pred_enc)
    return labels, confidence


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank-gene XGBoost cell type classifier"
    )
    parser.add_argument("--reference",    required=True,
                        help="Path to reference h5ad with cell type labels")
    parser.add_argument("--celltype-col", required=True,
                        help="Column in reference .obs containing cell type labels")
    parser.add_argument("--query",        required=True,
                        help="Path to segmented sample h5ad to annotate")
    parser.add_argument("--output-dir",   required=True,
                        help="Directory to write annotated h5ad and CSV")
    parser.add_argument("--sample-id",    default="",
                        help="Sample ID used for output file naming")
    parser.add_argument("--gpu",          action="store_true",
                        help="Use GPU for XGBoost (requires CUDA, XGBoost >= 2.0)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = args.sample_id or Path(args.query).stem

    print("=" * 60)
    print(f"  Classifier — {sample_id}")
    print(f"  Reference:  {args.reference}")
    print(f"  Query:      {args.query}")
    print(f"  Celltype:   {args.celltype_col}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)

    # ── Load ──
    print("[INFO] Loading reference data...")
    ref = sc.read_h5ad(args.reference)
    print(f"[INFO]   {ref.n_obs} cells × {ref.n_vars} genes")

    if args.celltype_col not in ref.obs.columns:
        raise ValueError(
            f"Column '{args.celltype_col}' not found in reference .obs.\n"
            f"Available columns: {list(ref.obs.columns)}"
        )

    print("[INFO] Loading query (segmented) data...")
    query = sc.read_h5ad(args.query)
    print(f"[INFO]   {query.n_obs} cells × {query.n_vars} genes")

    # ── Align genes — MUST happen before rank transformation ──
    # The Xenium panel is a small targeted set (~500 genes). The reference may
    # have 20k+ genes. If we ranked the reference across all its genes first,
    # rank 5 in reference would mean "5th out of 20,000" while rank 5 in the
    # Xenium query means "5th out of 500" — incomparable features. By filtering
    # to the shared gene set first, both reference and query are ranked within
    # the same Xenium panel universe, making the classifier features directly
    # comparable.
    ref, query = align_genes(ref, query)

    # ── Rank transform (within the shared Xenium panel gene set) ──
    print("[INFO] Rank-transforming reference counts...")
    X_train = counts_to_rank(ref, desc="Ranking reference genes")
    y_train  = ref.obs[args.celltype_col].astype(str).values

    print("[INFO] Rank-transforming query counts...")
    X_query = counts_to_rank(query, desc="Ranking query genes")

    # ── Train ──
    clf, le = train_classifier(X_train, y_train, use_gpu=args.gpu)
    print(f"[INFO] Classes ({len(le.classes_)}): {', '.join(le.classes_)}")

    # ── Predict ──
    print("[INFO] Predicting cell types on query data...")
    labels, confidence = predict_labels(clf, le, X_query)

    # ── Write back to adata ──
    query.obs["predicted_cell_type"]            = labels
    query.obs["predicted_cell_type_confidence"] = confidence

    # ── Save annotated h5ad ──
    h5ad_path = output_dir / f"{sample_id}_annotated.h5ad"
    query.write_h5ad(h5ad_path)
    print(f"[INFO] Annotated h5ad saved: {h5ad_path}")

    # ── Save CSV (for segger / R notebooks) ──
    csv_df = query.obs[["predicted_cell_type", "predicted_cell_type_confidence"]].copy()
    csv_df.index.name = "cell_id"
    csv_path = output_dir / f"{sample_id}_predicted_celltypes.csv"
    csv_df.to_csv(csv_path)
    print(f"[INFO] Predicted cell types CSV saved: {csv_path}")

    # ── Distribution summary ──
    counts_series = pd.Series(labels).value_counts()
    print(f"\n[INFO] Predicted cell type distribution ({query.n_obs} cells):")
    for ct, n in counts_series.items():
        bar = "█" * int(30 * n / len(labels))
        print(f"  {ct:<40} {n:>6}  {bar}")

    print(f"\n[DONE] {sample_id} — {len(labels)} cells annotated")


if __name__ == "__main__":
    main()
