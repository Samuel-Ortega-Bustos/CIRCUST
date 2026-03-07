"""
What is CPCA?
-------------
Standard PCA on a gene-expression matrix gives principal components that
explain linear variance. CPCA goes one step further: it takes the loadings
of PC1 and PC2 — one value per sample — and treats them as (x, y)
coordinates on a 2-D plane. Each sample gets projected onto the unit circle
via an angle:

    phi = atan2(PC2_loading, PC1_loading)   in [0, 2*pi)

Sorting samples by phi gives a *circular* ordering that reflects the
underlying circadian rhythm. Samples near the origin (small distance from
(0,0) in PC1-PC2 space) are potential outliers because they do not
participate strongly in the dominant oscillation.

Pipeline position
-----------------
    preprocessing.py  →  cpca.py  →  outlier_detection.py  →  ...

Input:  PreprocessingResult.expr_norm  (full normalised matrix)
Output: CPCAResult                     (sample order + outlier flags)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from circust.constants import SEED_GENES_DEFAULT,OUTLIER_RADIAL_THRESHOLD,OUTLIER_RADIAL_THRESHOLD_LOOSE,N_OUTLIER_CANDIDATES

# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class CPCAResult:
    """
    All outputs produced by :class:`CPCA`.

    Attributes
    ----------
    sample_order : np.ndarray of int, shape (n_samples,)
        Integer indices that sort samples by their circular phase phi.
        Apply this to any matrix column axis to get the CPCA ordering.
        R equivalent: ``orderCPCA8``  (obtainCPCA13 line 3816).

    circular_scale : np.ndarray of float, shape (n_samples,)
        The sorted phi values in [0, 2*pi). This is the circular time
        axis used by all downstream fitting steps.
        R equivalent: ``escalaPhi8``  (obtainCPCA13 line 3817).

    pc1 : np.ndarray of float, shape (n_samples,)
        PC1 loadings — one value per sample.
        R equivalent: ``eigen18 = cp8$rotation[,1]``  (line 3810).

    pc2 : np.ndarray of float, shape (n_samples,)
        PC2 loadings — one value per sample.
        R equivalent: ``eigen28 = cp8$rotation[,2]``  (line 3811).

    pc3 : np.ndarray of float, shape (n_samples,)
        PC3 loadings — one value per sample.
        R equivalent: ``eigen38 = cp8$rotation[,3]``  (line 3812).

    variance_explained : np.ndarray of float, shape (3,)
        Fraction of total variance explained by PC1, PC2, PC3.
        R equivalent: ``varPer8``  (lines 3807-3809).

    outlier_candidate_idx : np.ndarray of int
        Column indices (in the ORIGINAL matrix) of the N_OUTLIER_CANDIDATES
        samples with the smallest PC1-PC2 distance from the origin.
        R equivalent: ``obs8 = order(d8)[1:nOuts]``  (line 3824).

    outlier_idx : np.ndarray of int
        Subset of outlier_candidate_idx that actually fall inside the
        tight (<=0.10) or loose (<=0.15) radius.
        R equivalent: the indices collected in the s8 loop (lines 3831-3846).

    outlier_positions_in_order : np.ndarray of int
        Position of each outlier within ``sample_order`` (not in the
        original matrix). Used by downstream residual plots.
        R equivalent: ``outs = match(obs8[1:ss8], orderCPCA8)`` (line 3872).

    n_outliers : int
        Number of confirmed outliers (= len(outlier_idx)).
        R equivalent: ``ss8``  (line 3847).

    used_loose_radius : bool
        True if the fallback 0.15 radius was needed because no sample
        qualified at 0.10.
        R equivalent: ``rojo8``  (line 3818 / 3843).

    core_genes_found : List[str]
        The core genes that were actually present in the matrix.
        (Generalisation over R: R assumes all 12 are always present.)
    """

    sample_order:              np.ndarray
    circular_scale:            np.ndarray
    pc1:                       np.ndarray
    pc2:                       np.ndarray
    pc3:                       np.ndarray
    variance_explained:        np.ndarray
    outlier_candidate_idx:     np.ndarray
    outlier_idx:               np.ndarray
    outlier_positions_in_order: np.ndarray
    n_outliers:                int
    used_loose_radius:         bool
    core_genes_found:          list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== CPCA Summary ===",
            f"  Core genes used    : {len(self.core_genes_found)}  {self.core_genes_found}",
            f"  PC1 variance       : {self.variance_explained[0]:.1%}",
            f"  PC2 variance       : {self.variance_explained[1]:.1%}",
            f"  PC3 variance       : {self.variance_explained[2]:.1%}",
            f"  Outlier candidates : {len(self.outlier_candidate_idx)}",
            f"  Confirmed outliers : {self.n_outliers}"
            + (" (loose radius used)" if self.used_loose_radius else ""),
        ]
        return "\n".join(lines)

# ===========================================================================
# CPCA class
# ===========================================================================

class CPCA:
    """
    Extract core clock genes and compute a circular sample ordering via PCA.

    This class implements two responsibilities that are tightly coupled in
    the R source but kept separate here for clarity:

    1. **Core gene extraction** — pull the 12 core clock gene rows out of
       the full normalised matrix (R lines 3954-3955).

    2. **CPCA** — run PCA on the row-centred core-gene matrix, project
       each sample onto the unit circle using PC1 and PC2 loadings, sort
       samples by their angle phi, and flag near-origin outliers
       (R function ``obtainCPCA13``, lines 3804-3893).

    Parameters
    ----------
    core_genes : list of str, optional
        Gene symbols to use as the circadian anchor set.
        Defaults to the 12 genes from the CIRCUST paper.
        You can pass a custom list if working with a non-human organism or
        a different gene annotation.

    n_outlier_candidates : int
        How many samples to examine as potential outliers (those with the
        smallest PC1-PC2 norm). R default: 8.

    tight_radius : float
        Primary distance threshold. Samples with norm <= tight_radius are
        confirmed outliers. R: 0.10.

    loose_radius : float
        Fallback threshold used when no sample qualifies at tight_radius.
        R: 0.15.

    verbose : bool
        Print progress messages if True.

    Examples
    --------
    >>> from preprocessing import load_expression_matrix, Preprocessor
    >>> from cpca import CPCA
    >>>
    >>> matrix  = load_expression_matrix("data/raw/expression.csv")
    >>> prep    = Preprocessor().run(matrix)
    >>> result  = CPCA().run(prep.expr_norm)
    >>> print(result.summary())
    """

    def __init__(
        self,
        core_genes:           list[str] = None,
        n_outlier_candidates: int       = N_OUTLIER_CANDIDATES,
        tight_radius:         float     = OUTLIER_RADIAL_THRESHOLD,
        loose_radius:         float     = OUTLIER_RADIAL_THRESHOLD_LOOSE,
        verbose:              bool      = True,
    ) -> None:

        self.core_genes           = core_genes if core_genes is not None else SEED_GENES_DEFAULT
        self.n_outlier_candidates = n_outlier_candidates
        self.tight_radius         = tight_radius
        self.loose_radius         = loose_radius
        self.verbose              = verbose

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self, expr_norm: pd.DataFrame) -> CPCAResult:
        """
        Extract core genes and compute the circular sample ordering.

        Parameters
        ----------
        expr_norm : pd.DataFrame
            Full normalised expression matrix (genes × samples, values in
            [-1, 1]).  This is ``PreprocessingResult.expr_norm``.

        Returns
        -------
        CPCAResult
        """
        # step 1 — pull core gene rows from the full matrix
        core_matrix, genes_found = self._extract_core_genes(expr_norm)

        # step 2 — run CPCA on the core gene matrix
        result = self._run_cpca(core_matrix, genes_found)

        self._log(result.summary())
        return result

    # -----------------------------------------------------------------------
    # Private steps
    # -----------------------------------------------------------------------

    def _extract_core_genes(self, expr_norm: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Select the core clock gene rows from the full normalised matrix.
        """
        genes_found  = []
        missing      = []

        for gene in self.core_genes:
            if gene in expr_norm.index:
                genes_found.append(gene)
            else:
                missing.append(gene)

        if missing:
            self._log(
                f"  WARNING — {len(missing)} core gene(s) not found in matrix "
                f"and will be skipped: {missing}"
            )

        if len(genes_found) < 2:
            raise ValueError(
                f"CPCA requires at least 2 core genes. "
                f"Only found: {genes_found}. "
                f"Check that the matrix rows use standard gene symbols."
            )

        self._log(
            f"  Core genes extracted: {len(genes_found)} / "
            f"{len(self.core_genes)}"
        )

        # preserve coreG ordering, just like R's match()
        core_matrix = expr_norm.loc[genes_found]
        return core_matrix, genes_found

    def _run_cpca(
        self,
        core_matrix: pd.DataFrame,
        genes_found: list[str],
    ) -> CPCAResult:
        """
        Run the full CPCA procedure on the core gene normalised matrix.

        R equivalent: ``obtainCPCA13()`` lines 3804-3893.

        Steps inside this method, with R line references:

        1. Row-centre the matrix (R: centrado(), lines 3806)
        2. Column-scale and run PCA (R: prcomp(scale.=TRUE), line 3806)
        3. Extract PC1, PC2, PC3 loadings per sample (lines 3810-3812)
        4. Project samples onto the unit circle (lines 3813-3815)
        5. Sort samples by angle phi (lines 3816-3817)
        6. Compute PC1-PC2 norm per sample (lines 3821-3823)
        7. Flag outlier samples (lines 3824-3847)
        8. Map outlier positions into the sorted order (line 3872)
        """
        values   = core_matrix.values.astype(float)   # shape: (n_genes, n_samples)
        n_genes, n_samples = values.shape

        # ── Step 1: row-centre ──────────────────────────────────────────────
        # In numpy: axis=1 means "along columns" → each row mean subtracted.
        row_means = values.mean(axis=1, keepdims=True)
        centred   = values - row_means    # shape: (n_genes, n_samples)

        # ── Step 2: column-scale and PCA ────────────────────────────────────
        # R: prcomp(centrado(mNorm8), scale.=TRUE, center=FALSE)
        #
        # In R, prcomp treats COLUMNS as variables and ROWS as observations.
        # Here the matrix is (n_genes × n_samples), so:
        #   - columns = samples  → variables in R's view
        #   - rows    = genes    → observations in R's view
        #
        # scale.=TRUE divides each COLUMN (= each sample) by its std dev.
        # This equalises the contribution of every sample to the PCA.
        #
        # sklearn's PCA also treats rows as observations. So we pass the
        # matrix as-is (genes × samples) and sklearn sees genes as
        # observations and samples as features — exactly matching R.
        #
        # After scaling columns:
        col_rms = np.sqrt(np.sum(centred**2, axis=0) / (n_genes - 1))
        col_rms[col_rms == 0] = 1.0          # evitar división por cero
        scaled  = centred / col_rms          # shape: (n_genes, n_samples)

        # WHY NOT sklearn PCA?
        # sklearn PCA always subtracts column means internally before SVD
        # and has no center=False option. Passing our matrix to PCA would
        # add a second, unwanted centering pass on top of centrado().
        #
        # WHY NOT TruncatedSVD.explained_variance_ratio_?
        # TruncatedSVD computes its ratio as var(projections) / var(X_input),
        # which uses a different denominator than R. R uses
        # sigma_k^2 / sum(ALL sigma^2) — a ratio purely of singular values.
        # The numbers differ by ~0.1%, which matters for a faithful port.
        #
        # SOLUTION: TruncatedSVD for the decomposition (no centering),
        # scipy full_svd for ALL singular values to get the correct denominator.
        n_components = min(3, n_genes, n_samples)
        _, sigma_all, Vt = np.linalg.svd(scaled, full_matrices=False)

        # Las componentes (loadings) en Scipy/Numpy salen transpuestas (Vt)
        # Vt tiene shape (n_genes, n_samples). Las filas son el equivalente a cp8$rotation
        pc1 = Vt[0]
        pc2 = Vt[1]
        pc3 = Vt[2] if n_components >= 3 else np.zeros(n_samples)


        # ── Step 4: variance explained ──────────────────────────────────────
        var_exp = np.zeros(3)
        var_exp[:n_components] = (sigma_all[:n_components]**2) / np.sum(sigma_all**2)
        
        # ── Step 5: project onto unit circle ────────────────────────────────
        norm12 = np.sqrt(pc1**2 + pc2**2)
        safe_norm = np.where(norm12 == 0, 1.0, norm12)

        xi  = pc1 / safe_norm
        yi  = pc2 / safe_norm
        phi = np.arctan2(yi, xi) % (2 * np.pi)                    # R: phi8

        # ── Step 6: sort samples by angle ───────────────────────────────────
        sample_order   = np.argsort(phi)                            # R: orderCPCA8
        circular_scale = phi[sample_order]                          # R: escalaPhi8
        
        # ── Step 7: outlier logic (continúa igual) ──────────────────────────
        d = norm12                                                  # R: d8

        # ── Step 7: flag outlier samples ────────────────────────────────────
        # R (line 3824): obs8 <- order(d8)[1:nOuts]
        # Take the nOuts samples with the SMALLEST norm (closest to origin)
        n_cands            = min(self.n_outlier_candidates, n_samples)
        candidate_idx      = np.argsort(d)[:n_cands]               # R: obs8

        # R (lines 3831-3847): check radii in two passes
        #   First pass: count samples with d <= 0.10 (tight radius)
        #   Second pass (only if first found nothing): count d <= 0.15
        tight_mask = d[candidate_idx] <= self.tight_radius
        used_loose = False

        if tight_mask.any():
            outlier_idx = candidate_idx[tight_mask]
        else:
            loose_mask  = d[candidate_idx] <= self.loose_radius
            outlier_idx = candidate_idx[loose_mask]
            used_loose  = loose_mask.any()                          # R: rojo8

        n_outliers = len(outlier_idx)                               # R: ss8

        # ── Step 8: map outlier positions into the sorted order ─────────────
        # So for each outlier original index, we find where it sits in sample_order.
        if n_outliers > 0:
            outlier_positions = np.array([
                int(np.where(sample_order == idx)[0][0])
                for idx in outlier_idx
            ])
        else:
            outlier_positions = np.array([], dtype=int)

        return CPCAResult(
            sample_order               = sample_order,
            circular_scale             = circular_scale,
            pc1                        = pc1,
            pc2                        = pc2,
            pc3                        = pc3,
            variance_explained         = var_exp,
            outlier_candidate_idx      = candidate_idx,
            outlier_idx                = outlier_idx,
            outlier_positions_in_order = outlier_positions,
            n_outliers                 = n_outliers,
            used_loose_radius          = used_loose,
            core_genes_found           = genes_found,
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

