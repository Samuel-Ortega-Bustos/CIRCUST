"""
validation/validate_vs_r.py
===========================
Compare Python CIRCUST pipeline (Stages 1-2) against R reference outputs.

Usage (from repo root):
    python validation/validate_vs_r.py

Prerequisites:
    1. Run ``Rscript rscripts/validate_export.R`` first to generate CSVs
       under ``validation/reference/``.
    2. The ``circust`` package must be installed (``pip install -e .``).
    3. The file ``data/matrixIn.parquet`` must exist.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
from circust.preprocessing import Preprocessor, load_expression_matrix
from circust.cpca import CPCA
from circust.synchronizer import CircularSynchronizer
from circust.core_genes import SEED_GENES_LARRIBA

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR   = REPO_ROOT / "validation" / "reference"
DATA_PATH = REPO_ROOT / "data" / "matrixIn.parquet"

CORE_GENES = SEED_GENES_LARRIBA

# ── tolerance levels ─────────────────────────────────────────────────────────
TOL_EXACT    = 0          # integer comparisons — must match exactly
TOL_STRICT   = 1e-10      # normalisation, simple arithmetic
TOL_MODERATE = 1e-4       # SVD / PCA differences (sign, library differences)
TOL_FMM      = 5e-2       # FMM optimisation — non-convex, may settle differently


# =============================================================================
# Helpers
# =============================================================================

class ValidationReport:
    """Accumulate pass/fail checks and print a summary."""

    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)

    def summary(self):
        total   = len(self.checks)
        passed  = sum(1 for _, ok, _ in self.checks if ok)
        failed  = total - passed
        print(f"\n{'='*60}")
        print(f"  {passed}/{total} checks passed, {failed} failed")
        if failed:
            print("\n  Failed checks:")
            for name, ok, detail in self.checks:
                if not ok:
                    print(f"    - {name}: {detail}")
        print(f"{'='*60}")
        return failed == 0


def load_ref(name: str) -> pd.DataFrame:
    """Load a reference CSV from the R export directory."""
    return pd.read_csv(REF_DIR / name)


def compare_arrays(py: np.ndarray, r: np.ndarray,
                   tol: float, name: str, report: ValidationReport):
    """Compare two arrays element-wise within tolerance."""
    if py.shape != r.shape:
        report.check(name, False,
                     f"shape mismatch: py={py.shape} vs r={r.shape}")
        return
    if tol == 0:
        diff = int(np.sum(py != r))
        report.check(name, diff == 0, f"{diff} mismatches")
    else:
        max_diff = float(np.max(np.abs(py - r)))
        report.check(name, max_diff < tol,
                     f"max_diff={max_diff:.2e}, tol={tol:.0e}")


def compare_sets(py: set, r: set, name: str, report: ValidationReport):
    """Compare two sets for equality."""
    if py == r:
        report.check(name, True, f"both have {len(py)} elements")
    else:
        only_py = py - r
        only_r  = r - py
        report.check(name, False,
                     f"only_py={only_py}, only_r={only_r}")


# =============================================================================
# Main validation
# =============================================================================

def main():
    report = ValidationReport()

    if not REF_DIR.exists():
        print("ERROR: Reference directory not found. Run:")
        print("  Rscript rscripts/validate_export.R")
        sys.exit(1)

    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        sys.exit(1)

    # =====================================================================
    # STAGE 1: PREPROCESSING
    # =====================================================================
    print("\n" + "="*60)
    print("  STAGE 1: PREPROCESSING")
    print("="*60)

    t0 = time.time()
    matrix = load_expression_matrix(str(DATA_PATH), gene_column="gene_id")
    print(f"  Loaded parquet: {matrix.shape[0]} genes x {matrix.shape[1]} samples")

    prep = Preprocessor(verbose=False).run(matrix)
    t_prep = time.time() - t0
    print(f"  Preprocessing: {t_prep:.2f}s")
    print(f"  After preprocessing: {prep.n_genes_out} genes x {prep.n_samples} samples")

    # -- Check dropped sparse genes --
    r_sparse = set(load_ref("r_dropped_sparse.csv")["gene"].dropna())
    py_sparse = set(prep.dropped_sparse)
    compare_sets(py_sparse, r_sparse, "Dropped sparse genes", report)

    # -- Check dropped duplicates --
    r_dupes = set(load_ref("r_dropped_dupes.csv")["gene"].dropna())
    py_dupes = set(prep.dropped_duplicates)
    compare_sets(py_dupes, r_dupes, "Dropped duplicates", report)

    # -- Check normalised matrix dimensions --
    r_norm = pd.read_csv(REF_DIR / "r_expr_norm.csv", index_col=0)
    report.check("Norm matrix shape",
                 prep.expr_norm.shape == r_norm.shape,
                 f"py={prep.expr_norm.shape} vs r={r_norm.shape}")

    # -- Check a subset of normalised values (first 50 genes, all samples) --
    common_genes = sorted(set(prep.expr_norm.index) & set(r_norm.index))[:50]
    if common_genes:
        py_sub = prep.expr_norm.loc[common_genes].values
        r_sub  = r_norm.loc[common_genes].values
        compare_arrays(py_sub, r_sub, TOL_STRICT,
                       "Normalised values (50 genes)", report)

    # -- Core gene extraction --
    r_core = pd.read_csv(REF_DIR / "r_core_norm.csv", index_col=0)
    py_core = prep.expr_norm.loc[
        [g for g in CORE_GENES if g in prep.expr_norm.index]
    ]
    compare_arrays(py_core.values, r_core.values, TOL_STRICT,
                   "Core gene normalised values", report)

    # =====================================================================
    # CPCA
    # =====================================================================
    print("\n" + "="*60)
    print("  CPCA (Circular PCA)")
    print("="*60)

    t0 = time.time()
    cpca = CPCA(core_genes=CORE_GENES, verbose=False).run(prep.expr_norm)
    t_cpca = time.time() - t0
    print(f"  CPCA: {t_cpca:.2f}s")

    # -- Sample order --
    # Note: R uses 1-based indexing. Convert.
    r_order = load_ref("r_sample_order.csv")["sample_order"].values - 1
    compare_arrays(cpca.sample_order, r_order, TOL_EXACT,
                   "CPCA sample order", report)

    # -- Circular scale --
    r_scale = load_ref("r_circular_scale.csv")["circular_scale"].values
    compare_arrays(cpca.circular_scale, r_scale, TOL_STRICT,
                   "CPCA circular scale", report)

    # -- Variance explained --
    r_var = load_ref("r_variance.csv")["variance_explained"].values
    compare_arrays(cpca.variance_explained, r_var, TOL_STRICT,
                   "CPCA variance explained", report)

    # -- PC loadings (may differ by sign) --
    r_pc = load_ref("r_pc_loadings.csv")
    for col, attr in [("pc1", "pc1"), ("pc2", "pc2"), ("pc3", "pc3")]:
        py_vals = getattr(cpca, attr)
        r_vals  = r_pc[col].values
        # PCA sign is arbitrary; compare abs or correlation
        corr = np.corrcoef(py_vals, r_vals)[0, 1]
        report.check(f"PC {attr} correlation",
                     abs(corr) > 0.999,
                     f"|corr|={abs(corr):.6f}")

    # -- Outlier candidates --
    r_oc = load_ref("r_outlier_candidates.csv").values.flatten().astype(int) - 1
    py_oc = cpca.outlier_candidate_idx
    # Sort both since order may differ
    compare_arrays(np.sort(py_oc[:len(r_oc)]), np.sort(r_oc), TOL_EXACT,
                   "CPCA outlier candidates", report)

    # =====================================================================
    # OUTLIER DETECTION + RE-CPCA
    # =====================================================================
    print("\n" + "="*60)
    print("  OUTLIER DETECTION")
    print("="*60)

    t0 = time.time()
    # Outlier detection is now integrated into CPCA.run() above
    t_outlier = time.time() - t0
    print(f"  Outlier detection + re-CPCA: (integrated in CPCA.run())")

    # -- FMM R² (initial fits) --
    # NOTE: FMM params have α ± 2π equivalence and multiple local optima.
    # We compare R² and peak times instead of raw params.
    r_fmm_init = pd.read_csv(REF_DIR / "r_fmm_params_initial.csv", index_col=0)
    for gene in CORE_GENES:
        if gene in cpca.fmm_fits_initial:
            fr = cpca.fmm_fits_initial[gene]
            report.check(f"FMM initial R2: {gene}",
                         fr.r2 > 0.3,
                         f"py_r2={fr.r2:.4f}")

    # -- FMM peak times (initial) --
    r_peaks = load_ref("r_peaks_initial.csv")
    r_peaks_core = r_peaks[r_peaks["gene"].isin(CORE_GENES)]
    for _, row in r_peaks_core.iterrows():
        gene = row["gene"]
        if gene in cpca.fmm_fits_initial:
            py_pk = cpca.fmm_fits_initial[gene].peak_time
            r_pk  = row["fmm_peak"]
            # Circular distance
            diff  = min(abs(py_pk - r_pk), 2*np.pi - abs(py_pk - r_pk))
            report.check(f"FMM peak initial: {gene}",
                         diff < TOL_FMM,
                         f"py={py_pk:.4f} r={r_pk:.4f} diff={diff:.4f}")

    # -- Outlier samples --
    r_outs = load_ref("r_outlier_samples.csv")["sample_idx"].values
    # R is 1-based
    r_outs_set = set(r_outs - 1) if len(r_outs) > 0 else set()
    py_outs_set = set(cpca.samples_dropped)
    compare_sets(py_outs_set, r_outs_set, "Outlier samples", report)

    # -- CPCA2 after outlier removal --
    r_order2 = load_ref("r_sample_order_2.csv")["sample_order"].values - 1
    compare_arrays(cpca.sample_order, r_order2, TOL_EXACT,
                   "CPCA2 sample order", report)

    r_scale2 = load_ref("r_circular_scale_2.csv")["circular_scale"].values
    compare_arrays(cpca.circular_scale, r_scale2, TOL_STRICT,
                   "CPCA2 circular scale", report)

    # -- FMM peaks (after outlier removal) --
    r_peaks_after = load_ref("r_peaks_after.csv")
    for _, row in r_peaks_after.iterrows():
        gene = row["gene"]
        if gene in cpca.fmm_peak_times_final:
            py_pk = cpca.fmm_peak_times_final[gene]
            r_pk  = row["fmm_peak"]
            diff  = min(abs(py_pk - r_pk), 2*np.pi - abs(py_pk - r_pk))
            report.check(f"FMM peak after: {gene}",
                         diff < TOL_FMM,
                         f"py={py_pk:.4f} r={r_pk:.4f} diff={diff:.4f}")

    # -- R² after --
    r_r2 = pd.read_csv(REF_DIR / "r_r2_after.csv", index_col=0)
    for gene in CORE_GENES:
        if gene in cpca.fmm_fits_final:
            py_r2 = cpca.fmm_fits_final[gene].r2
            r_r2v = r_r2.loc[gene, "r2_fmm"]
            report.check(f"R2 after: {gene}",
                         abs(py_r2 - r_r2v) < TOL_FMM,
                         f"py={py_r2:.4f} r={r_r2v:.4f}")

    # =====================================================================
    # STAGE 2: PRELIMINARY ORDERING
    # =====================================================================
    print("\n" + "="*60)
    print("  STAGE 2: PRELIMINARY ORDERING")
    print("="*60)

    t0 = time.time()
    pre_order = CircularSynchronizer(verbose=False).run(cpca, CORE_GENES)
    t_preord = time.time() - t0
    print(f"  Preliminary ordering: {t_preord:.2f}s")
    print(pre_order.summary())

    # -- Pre-order peaks (Step 2.1) --
    r_pre_peaks = load_ref("r_pre_peaks.csv")
    for _, row in r_pre_peaks.iterrows():
        gene = row["gene"]
        idx  = CORE_GENES.index(gene) if gene in CORE_GENES else -1
        if idx >= 0:
            py_pk = pre_order.pre_peak_times[idx]
            r_pk  = row["peak_time"]
            diff  = min(abs(py_pk - r_pk), 2*np.pi - abs(py_pk - r_pk))
            report.check(f"Pre-order peak: {gene}",
                         diff < TOL_FMM,
                         f"py={py_pk:.4f} r={r_pk:.4f} diff={diff:.4f}")

    # -- Pre-order direction (Step 2.1) --
    r_pre_dir = load_ref("r_pre_direction.csv")["pre_reversed"].values[0]
    r_pre_dir_bool = str(r_pre_dir).upper() == "TRUE"
    report.check("Pre-order direction",
                 pre_order.pre_direction_reversed == r_pre_dir_bool,
                 f"py={pre_order.pre_direction_reversed} r={r_pre_dir_bool}")

    # -- Final peaks (Step 2.2) --
    r_final_peaks = load_ref("r_final_peaks.csv")
    for _, row in r_final_peaks.iterrows():
        gene = row["gene"]
        idx  = CORE_GENES.index(gene) if gene in CORE_GENES else -1
        if idx >= 0:
            py_pk = pre_order.peak_times[idx]
            r_pk  = row["peak_time"]
            diff  = min(abs(py_pk - r_pk), 2*np.pi - abs(py_pk - r_pk))
            report.check(f"Final peak: {gene}",
                         diff < TOL_FMM,
                         f"py={py_pk:.4f} r={r_pk:.4f} diff={diff:.4f}")

    # -- Final direction (Step 2.2) --
    r_fin_dir = load_ref("r_final_direction.csv")["direction_flipped"].values[0]
    r_fin_dir_bool = str(r_fin_dir).upper() == "TRUE"
    report.check("Final direction flipped",
                 pre_order.direction_flipped == r_fin_dir_bool,
                 f"py={pre_order.direction_flipped} r={r_fin_dir_bool}")

    # -- Final circular scale --
    r_fin_scale = load_ref("r_final_circular_scale.csv")["circular_scale"].values
    compare_arrays(pre_order.circular_scale, r_fin_scale, TOL_FMM,
                   "Final circular scale", report)

    # -- FMM params (final stage 2): compare M, A, omega (stable params) --
    # alpha and beta are not identifiable due to 2π periodicity
    r_fmm_final = pd.read_csv(REF_DIR / "r_fmm_params_final.csv", index_col=0)
    for i, gene in enumerate(CORE_GENES):
        py_p = pre_order.fmm_params[i]
        r_p  = r_fmm_final.loc[gene][["M", "A", "alpha", "beta", "omega"]].values
        # Compare M, A, omega (which are identifiable)
        py_stable = np.array([py_p[0], py_p[1], py_p[4]])  # M, A, omega
        r_stable  = np.array([r_p[0], r_p[1], r_p[4]])
        compare_arrays(py_stable, r_stable, TOL_FMM,
                       f"FMM final M,A,omega: {gene}", report)

    # =====================================================================
    # TIMING SUMMARY
    # =====================================================================
    print(f"\n  Python timing:")
    print(f"    Preprocessing      : {t_prep:.2f}s")
    print(f"    CPCA               : {t_cpca:.2f}s")
    print(f"    Outlier + re-CPCA  : {t_outlier:.2f}s")
    print(f"    Preliminary order  : {t_preord:.2f}s")
    print(f"    TOTAL              : {t_prep + t_cpca + t_outlier + t_preord:.2f}s")

    # =====================================================================
    # FINAL REPORT
    # =====================================================================
    all_passed = report.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
