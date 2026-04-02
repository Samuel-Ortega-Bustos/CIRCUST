"""
circust/top_matrix.py
=====================
Stage 3.7: Construct the TOP gene expression matrix.

Combines the 12 core clock genes with the candidate rhythmic genes
selected in the earlier stages, producing the final matrix used by
Stage 4 (Robust Estimation).

The core genes are taken from the full normalised matrix (reordered by
the synchronised sample ordering), and the candidate genes come from
the synchronised candidate matrix.  Any artificial/added genes are
excluded from the candidate list to avoid duplication.

R equivalent
------------
``obtainTop_v2_cores()`` (lines 113–138 of ``functionGTEX_cores.R``).

Pipeline position
-----------------
    CandidateSelector → ReferenceSetBuilder → RandomSelector →
    **TopMatrixBuilder** → RobustEstimator
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_top_matrix(
    core_genes:         list[str],
    candidate_names:    list[str],
    expr_full_norm:     pd.DataFrame,
    candidate_norm_ord: pd.DataFrame,
    sample_order:       np.ndarray,
    genes_to_exclude:   list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the TOP gene matrix by stacking core genes on top of the
    candidate set.

    Parameters
    ----------
    core_genes : list of str
        The 12 core clock gene symbols (e.g. ARNTL, DBP, PER1, …).

    candidate_names : list of str
        Gene symbols selected by the reference-set / random-selection
        pipeline.
        R: ``mName``  → ``refSetRefG[[1]]``.

    expr_full_norm : pd.DataFrame, shape (n_all_genes, n_samples)
        Full normalised expression matrix (genes × samples).
        R: ``mFullNorm``  → ``preOrdRefG2[[3]]``.

    candidate_norm_ord : pd.DataFrame, shape (n_candidates, n_samples)
        Synchronised and normalised candidate expression matrix.
        R: ``mMinCandNormOrd``  → ``refSetRefG[[11]]``.

    sample_order : (n_samples,) int array
        Column indices that reorder ``expr_full_norm`` into the
        synchronised sample order.
        R: ``indOrd``  → ``refSetRefG[[12]]``.

    genes_to_exclude : list of str or None
        Artificial / added genes to remove from the candidate list
        before stacking (avoids duplicates with core genes).
        R: ``genAdd``  → ``refSetRefG[[19]]``.

    Returns
    -------
    pd.DataFrame, shape (n_core + n_cand_clean, n_samples)
        Row-stacked matrix: core genes first, then candidates.
        Row names = gene symbols.
        R equivalent: ``top``  → ``topRefG``.
    """
    # ── Remove excluded genes from candidate list ─────────────────────
    cand = list(candidate_names)
    if genes_to_exclude:
        for g in genes_to_exclude:
            if g in cand:
                cand.remove(g)

    # ── Remove core genes that also appear in the candidate list ──────
    # R: if all core genes are absent from mName → simple stack.
    #    else → drop overlapping core genes from mName.
    overlap = [g for g in core_genes if g in cand]
    for g in overlap:
        cand.remove(g)

    # ── Core gene rows: from full matrix, reordered by sample_order ───
    core_rows = expr_full_norm.loc[
        [g for g in core_genes if g in expr_full_norm.index]
    ].iloc[:, sample_order]
    core_rows.columns = range(core_rows.shape[1])

    # ── Candidate gene rows: from synchronised matrix ─────────────────
    cand_present = [g for g in cand if g in candidate_norm_ord.index]
    if len(cand_present) > 0:
        cand_rows = candidate_norm_ord.loc[cand_present].copy()
        cand_rows.columns = range(cand_rows.shape[1])
    else:
        cand_rows = pd.DataFrame(
            dtype=float, columns=range(core_rows.shape[1])
        )

    # ── Stack: core on top, candidates below ──────────────────────────
    top = pd.concat([core_rows, cand_rows], axis=0)
    top.index = list(core_rows.index) + list(cand_rows.index)

    return top
