"""
segger_functions — Segmentation comparison metrics.

Adapted from: https://github.com/EliHei2/segger_dev/blob/main/src/segger/validation/utils.py

Functions for computing:
  - MECR (Mutually Exclusive Co-expression Rate)
  - Contamination analysis
  - Sensitivity / marker detection
"""

import pandas as pd
import numpy as np
import anndata as ad
from typing import Dict, List, Tuple
import scanpy as sc
from itertools import combinations
import squidpy as sq

def find_markers(
    adata: ad.AnnData,
    cell_type_column: str,
    pos_percentile: float = 5,
    neg_percentile: float = 10,
    percentage: float = 50,
) -> Dict[str, Dict[str, List[str]]]:
    """Identify positive and negative markers for each cell type based on gene expression and filter by expression percentage.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - cell_type_column: str
        Column name in `adata.obs` that specifies cell types.
    - pos_percentile: float, default=5
        Percentile threshold to determine top x% expressed genes.
    - neg_percentile: float, default=10
        Percentile threshold to determine top x% lowly expressed genes.
    - percentage: float, default=50
        Minimum percentage of cells expressing the marker within a cell type for it to be considered.

    Returns:
    - markers: dict
        Dictionary where keys are cell types and values are dictionaries containing:
            'positive': list of top x% highly expressed genes
            'negative': list of top x% lowly expressed genes.
    """
    markers = {}
    sc.tl.rank_genes_groups(adata, groupby=cell_type_column)
    genes = adata.var_names
    for cell_type in adata.obs[cell_type_column].unique():
        subset = adata[adata.obs[cell_type_column] == cell_type]
        mean_expression = np.asarray(subset.X.mean(axis=0)).flatten()
        cutoff_high = np.percentile(mean_expression, 100 - pos_percentile)
        cutoff_low = np.percentile(mean_expression, neg_percentile)
        pos_indices = np.where(mean_expression >= cutoff_high)[0]
        neg_indices = np.where(mean_expression <= cutoff_low)[0]
        expr_frac = np.asarray((subset.X[:, pos_indices] > 0).mean(axis=0)).flatten()
        valid_pos_indices = pos_indices[expr_frac >= (percentage / 100)]
        positive_markers = genes[valid_pos_indices]
        negative_markers = genes[neg_indices]
        markers[cell_type] = {
            "positive": list(positive_markers),
            "negative": list(negative_markers),
        }
    return markers


def find_mutually_exclusive_genes(
    adata: ad.AnnData, markers: Dict[str, Dict[str, List[str]]], cell_type_column: str
) -> List[Tuple[str, str]]:
    """Identify mutually exclusive genes based on expression criteria.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - markers: dict
        Dictionary where keys are cell types and values are dictionaries containing:
            'positive': list of top x% highly expressed genes
            'negative': list of top x% lowly expressed genes.
    - cell_type_column: str
        Column name in `adata.obs` that specifies cell types.

    Returns:
    - exclusive_pairs: list
        List of mutually exclusive gene pairs.
    """
    exclusive_genes = {}
    all_exclusive = []
    gene_expression = adata.to_df()
    for cell_type, marker_sets in markers.items():
        positive_markers = marker_sets["positive"]
        exclusive_genes[cell_type] = []
        for gene in positive_markers:
            gene_expr = adata[:, gene].X.toarray()
            cell_type_mask = adata.obs[cell_type_column] == cell_type
            non_cell_type_mask = ~cell_type_mask
            if (gene_expr[cell_type_mask] > 0).mean() > 0.2 and (
                gene_expr[non_cell_type_mask] > 0
            ).mean() < 0.05:
                exclusive_genes[cell_type].append(gene)
                all_exclusive.append(gene)
    unique_genes = list(
        {
            gene
            for i in exclusive_genes.keys()
            for gene in exclusive_genes[i]
            if gene in all_exclusive
        }
    )
    filtered_exclusive_genes = {
        i: [gene for gene in exclusive_genes[i] if gene in unique_genes]
        for i in exclusive_genes.keys()
    }
    mutually_exclusive_gene_pairs = [
        (gene1, gene2)
        for key1, key2 in combinations(filtered_exclusive_genes.keys(), 2)
        for gene1 in filtered_exclusive_genes[key1]
        for gene2 in filtered_exclusive_genes[key2]
    ]
    return mutually_exclusive_gene_pairs


def compute_MECR(
    adata: ad.AnnData, gene_pairs: List[Tuple[str, str]]
) -> Dict[Tuple[str, str], float]:
    """Compute the Mutually Exclusive Co-expression Rate (MECR) for each gene pair in an AnnData object.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - gene_pairs: List[Tuple[str, str]]
        List of tuples representing gene pairs to evaluate.

    Returns:
    - mecr_dict: Dict[Tuple[str, str], float]
        Dictionary where keys are gene pairs (tuples) and values are MECR values.
    """
    mecr_dict = {}
    gene_expression = adata.to_df()
    for gene1, gene2 in gene_pairs:
        expr_gene1 = gene_expression[gene1] > 0
        expr_gene2 = gene_expression[gene2] > 0
        both_expressed = (expr_gene1 & expr_gene2).mean()
        at_least_one_expressed = (expr_gene1 | expr_gene2).mean()
        mecr = (
            both_expressed / at_least_one_expressed if at_least_one_expressed > 0 else 0
        )
        mecr_dict[(gene1, gene2)] = mecr
    return mecr_dict


def compute_quantized_mecr_area(
    adata: sc.AnnData, gene_pairs: List[Tuple[str, str]], quantiles: int = 10
) -> pd.DataFrame:
    """Compute the average MECR, variance of MECR, and average cell area for quantiles of cell areas.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - gene_pairs: List[Tuple[str, str]]
        List of tuples representing gene pairs to evaluate.
    - quantiles: int, default=10
        Number of quantiles to divide the data into.

    Returns:
    - quantized_data: pd.DataFrame
        DataFrame containing quantile information, average MECR, variance of MECR, average area, and number of cells.
    """
    adata.obs["quantile"] = pd.qcut(adata.obs["cell_area"], quantiles, labels=False)
    quantized_data = []
    for quantile in range(quantiles):
        cells_in_quantile = adata.obs["quantile"] == quantile
        mecr = compute_MECR(adata[cells_in_quantile, :], gene_pairs)
        average_mecr = np.mean([i for i in mecr.values()])
        variance_mecr = np.var([i for i in mecr.values()])
        average_area = adata.obs.loc[cells_in_quantile, "cell_area"].mean()
        quantized_data.append(
            {
                "quantile": quantile / quantiles,
                "average_mecr": average_mecr,
                "variance_mecr": variance_mecr,
                "average_area": average_area,
                "num_cells": cells_in_quantile.sum(),
            }
        )
    return pd.DataFrame(quantized_data)


def calculate_contamination(
    adata: ad.AnnData,
    markers: Dict[str, Dict[str, List[str]]],
    radius: float = 15,
    n_neighs: int = 10,
    celltype_column: str = "celltype_major",
    num_cells: int = 10000,
) -> pd.DataFrame:
    """Calculate normalized contamination from neighboring cells of different cell types based on positive markers.

    Args:
    - adata: ad.AnnData
        Annotated data object with raw counts and cell type information.
    - markers: dict
        Dictionary where keys are cell types and values are dictionaries containing:
            'positive': list of top x% highly expressed genes
            'negative': list of top x% lowly expressed genes.
    - radius: float, default=15
        Radius for spatial neighbor calculation.
    - n_neighs: int, default=10
        Maximum number of neighbors to consider.
    - celltype_column: str, default='celltype_major'
        Column name in the AnnData object representing cell types.
    - num_cells: int, default=10000
        Number of cells to randomly select for the calculation.

    Returns:
    - contamination_df: pd.DataFrame
        DataFrame containing the normalized level of contamination from each cell type to each other cell type.
    """
    if celltype_column not in adata.obs:
        raise ValueError("Column celltype_column must be present in adata.obs.")
    positive_markers = {ct: markers[ct]["positive"] for ct in markers}
    adata.obsm["spatial"] = (
        adata.obs[["cell_centroid_x", "cell_centroid_y"]].copy().to_numpy()
    )
    # Use kNN-only (no radius) so contamination works regardless of coordinate scale.
    # Radius-based lookup silently returns 0 neighbors when coordinates are in pixels
    # rather than microns (e.g. image-based methods like cellpose).
    sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic")
    neighbors = adata.obsp["spatial_connectivities"].tolil()
    mean_neighbors = np.mean([len(r) for r in neighbors.rows])
    if mean_neighbors < 1:
        print(f"[WARN] contamination: mean neighbors = {mean_neighbors:.1f} — "
              "spatial graph is empty, all values will be 0")
    raw_counts = adata[:, adata.var_names].layers["raw"].toarray()
    cell_types = adata.obs[celltype_column]
    selected_cells = np.random.choice(
        adata.n_obs, size=min(num_cells, adata.n_obs), replace=False
    )
    contamination = {
        ct: {ct2: 0 for ct2 in positive_markers.keys()}
        for ct in positive_markers.keys()
    }
    negighborings = {
        ct: {ct2: 0 for ct2 in positive_markers.keys()}
        for ct in positive_markers.keys()
    }
    for cell_idx in selected_cells:
        cell_type = cell_types[cell_idx]
        if cell_type not in positive_markers:
            continue
        own_markers = set(positive_markers[cell_type])
        for marker in own_markers:
            if marker in adata.var_names:
                total_counts_in_neighborhood = raw_counts[
                    cell_idx, adata.var_names.get_loc(marker)
                ]
                for neighbor_idx in neighbors.rows[cell_idx]:
                    total_counts_in_neighborhood += raw_counts[
                        neighbor_idx, adata.var_names.get_loc(marker)
                    ]
                for neighbor_idx in neighbors.rows[cell_idx]:
                    neighbor_type = cell_types[neighbor_idx]
                    if cell_type == neighbor_type:
                        continue
                    neighbor_markers = set(positive_markers.get(neighbor_type, []))
                    contamination_markers = own_markers - neighbor_markers
                    for marker in contamination_markers:
                        if marker in adata.var_names:
                            marker_counts_in_neighbor = raw_counts[
                                neighbor_idx, adata.var_names.get_loc(marker)
                            ]
                            if total_counts_in_neighborhood > 0:
                                contamination[cell_type][neighbor_type] += (
                                    marker_counts_in_neighbor
                                    / total_counts_in_neighborhood
                                )
                                negighborings[cell_type][neighbor_type] += 1
    contamination_df = pd.DataFrame(contamination).T
    negighborings_df = pd.DataFrame(negighborings).T
    contamination_df.index.name = "Source Cell Type"
    contamination_df.columns.name = "Target Cell Type"
    return contamination_df / (negighborings_df + 1)


def calculate_sensitivity(
    adata: ad.AnnData,
    purified_markers: Dict[str, List[str]],
    max_cells_per_type: int = 1000,
) -> Dict[str, List[float]]:
    """Calculate the sensitivity of the purified markers for each cell type.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - purified_markers: dict
        Dictionary where keys are cell types and values are lists of purified marker genes.
    - max_cells_per_type: int, default=1000
        Maximum number of cells to consider per cell type.

    Returns:
    - sensitivity_results: dict
        Dictionary with cell types as keys and lists of sensitivity values for each cell.
    """
    sensitivity_results = {cell_type: [] for cell_type in purified_markers.keys()}
    for cell_type, markers in purified_markers.items():
        markers = markers["positive"]
        subset = adata[adata.obs["celltype_major"] == cell_type]
        if subset.n_obs > max_cells_per_type:
            cell_indices = np.random.choice(
                subset.n_obs, max_cells_per_type, replace=False
            )
            subset = subset[cell_indices]
        # get_indexer returns -1 for genes absent in this adata — filter them out
        gene_idx = subset.var_names.get_indexer(markers)
        gene_idx = gene_idx[gene_idx >= 0]
        for cell_counts in subset.X:
            import scipy.sparse as _ssp
            if _ssp.issparse(cell_counts):
                cell_arr = np.asarray(cell_counts.todense()).flatten()
            else:
                cell_arr = np.asarray(cell_counts).flatten()
            n_expressed = int((cell_arr[gene_idx] > 0).sum()) if len(gene_idx) > 0 else 0
            sensitivity = n_expressed / len(markers) if markers else 0
            sensitivity_results[cell_type].append(sensitivity)
    return sensitivity_results
