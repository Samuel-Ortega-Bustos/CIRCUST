"""
scripts/validate_compare.py
============================
Loads the reference outputs produced by rscripts/r_export.R and runs
the Python CIRCUST preprocessing + CPCA on identical input data.

Compares every intermediate value numerically and prints a PASS/FAIL
report. Also times each Python step and shows a side-by-side table
against the R timings.

Run from the repo ROOT (circust/):
    python scripts/validate_compare.py

Expects:
    data/raw/matrixIn.parquet     — your parquet input file
    validation/reference/         — folder produced by rscripts/r_export.R
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — all relative to the repo root
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).parent.parent      # circust/
DATA_PATH  = REPO_ROOT / "data" / "raw" / "matrixIn.parquet"
REF_DIR    = REPO_ROOT / "validation" / "reference"

# Add the circust package to sys.path so imports work without pip install
sys.path.insert(0, str(REPO_ROOT / "circust"))

from circust.preprocessing import load_expression_matrix, Preprocessor, PreprocessingResult
from circust.cpca import CPCA

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
ATOL       = 1e-6   # strict: for normalised values, circular scale, PC loadings
ATOL_VAR   = 1e-4   # loose:  for variance explained (small formula difference)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return passed


def max_diff_str(a: np.ndarray, b: np.ndarray) -> str:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return f"max={d.max():.2e}  mean={d.mean():.2e}"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("\n=== Loading data ===")

if not DATA_PATH.exists():
    sys.exit(f"ERROR: parquet file not found at {DATA_PATH}\n"
             f"Check that matrixIn.parquet is in data/raw/")

if not REF_DIR.exists():
    sys.exit(f"ERROR: reference folder not found at {REF_DIR}\n"
             f"Run  Rscript rscripts/r_export.R  first.")

t0 = time.perf_counter()
matrix = load_expression_matrix(str(DATA_PATH),gene_column="gene_id")
#matrix = load_expression_matrix(str(DATA_PATH))
t_load = time.perf_counter() - t0
print(f"  {matrix.shape[0]} genes × {matrix.shape[1]} samples  "
      f"loaded in {t_load:.3f}s")

# ---------------------------------------------------------------------------
# 2. Run Python preprocessing — timed per step
# ---------------------------------------------------------------------------
print("\n=== Running Python preprocessing ===")

timing = {}
prep_obj = Preprocessor(verbose=False)

t0 = time.perf_counter()
mat = prep_obj._drop_unnamed(matrix)
timing["drop_unnamed"] = time.perf_counter() - t0

t0 = time.perf_counter()
mat, dropped_sparse = prep_obj._drop_sparse(mat)
timing["drop_sparse"] = time.perf_counter() - t0

t0 = time.perf_counter()
mat, dropped_dupes = prep_obj._resolve_duplicates(mat)
timing["resolve_dupes"] = time.perf_counter() - t0

t0 = time.perf_counter()
expr_norm = prep_obj._normalise(mat)
timing["normalise"] = time.perf_counter() - t0

prep_result = PreprocessingResult(
    expr_norm          = expr_norm,
    expr_raw           = mat,
    dropped_sparse     = dropped_sparse,
    dropped_duplicates = dropped_dupes,
    n_genes_in         = matrix.shape[0],
    n_genes_out        = len(expr_norm),
    n_samples          = matrix.shape[1],
)
print(f"  Done — {prep_result.n_genes_out} genes survived")

# ---------------------------------------------------------------------------
# 3. Run Python CPCA — timed
# ---------------------------------------------------------------------------
print("=== Running Python CPCA ===")

t0 = time.perf_counter()
cpca_result = CPCA(verbose=False).run(prep_result.expr_norm)
timing["cpca"] = time.perf_counter() - t0
print(f"  Done — {cpca_result.n_outliers} outlier(s) found")

# ---------------------------------------------------------------------------
# 4. Load R reference outputs
# ---------------------------------------------------------------------------
print("\n=== Loading R reference outputs ===")

def load_ref(filename: str) -> pd.DataFrame:
    path = REF_DIR / filename
    if not path.exists():
        sys.exit(f"ERROR: missing reference file {path}\n"
                 f"Re-run  Rscript rscripts/r_export.R")
    return pd.read_csv(path)

r_expr_norm   = load_ref("r_expr_norm.csv").set_index(
                    load_ref("r_expr_norm.csv").columns[0])
r_dropped_sp  = load_ref("r_dropped_sparse.csv")["gene"].tolist()
r_dropped_du  = load_ref("r_dropped_dupes.csv")["gene"].tolist()

# R uses 1-based indices — subtract 1 to get 0-based Python indices
r_sample_order = load_ref("r_sample_order.csv")["sample_order"].values - 1
r_circ_scale   = load_ref("r_circular_scale.csv")["circular_scale"].values
r_pc           = load_ref("r_pc_loadings.csv")
r_variance     = load_ref("r_variance.csv")["variance_explained"].values
r_candidates   = load_ref("r_outlier_candidates.csv")["candidate_idx"].values - 1
r_timing       = load_ref("r_timing.csv")

print(f"  Loaded from {REF_DIR}/")

# ---------------------------------------------------------------------------
# 5. Preprocessing checks
# ---------------------------------------------------------------------------
print("\n=== Preprocessing Comparison ===")
results = {}

results["Gene count matches"] = check(
    "Gene count matches",
    len(prep_result.expr_norm) == len(r_expr_norm),
    f"Python={len(prep_result.expr_norm)}  R={len(r_expr_norm)}"
)

py_genes = set(prep_result.expr_norm.index)
r_genes  = set(r_expr_norm.index)
results["Same genes survived"] = check(
    "Same genes survived filtering",
    py_genes == r_genes,
    f"only_in_python={len(py_genes - r_genes)}  "
    f"only_in_R={len(r_genes - py_genes)}"
)

results["Same sparse genes dropped"] = check(
    "Same sparse genes dropped",
    set(prep_result.dropped_sparse) == set(r_dropped_sp),
    f"Python={len(prep_result.dropped_sparse)}  R={len(r_dropped_sp)}"
)

results["Same duplicate genes dropped"] = check(
    "Same duplicate genes dropped",
    set(prep_result.dropped_duplicates) == set(r_dropped_du),
    f"Python={len(prep_result.dropped_duplicates)}  R={len(r_dropped_du)}"
)

# Align rows by gene name before comparing values
common = sorted(py_genes & r_genes)
py_vals = prep_result.expr_norm.loc[common].values.astype(float)
r_vals  = r_expr_norm.loc[common].values.astype(float)
close   = np.allclose(py_vals, r_vals, atol=ATOL)
results["Normalised values match"] = check(
    f"Normalised values match (atol={ATOL:.0e})",
    close,
    max_diff_str(py_vals, r_vals) if not close else ""
)

# ---------------------------------------------------------------------------
# 5b. Core matrix diagnostics — printed always, critical for debugging
# ---------------------------------------------------------------------------
print("\n=== Core Matrix Diagnostics ===")

r_core_norm = load_ref("r_core_norm.csv").set_index(
                  load_ref("r_core_norm.csv").columns[0])

# Which genes does each side actually use?
cpca_obj2     = CPCA(verbose=False)
core_matrix_py, genes_found_py = cpca_obj2._extract_core_genes(prep_result.expr_norm)
r_core_genes  = list(r_core_norm.index)

print(f"  Python core genes ({len(genes_found_py)}): {genes_found_py}")
print(f"  R      core genes ({len(r_core_genes)}): {r_core_genes}")
print(f"  Gene order matches: {genes_found_py == r_core_genes}")

# Compare actual core matrix values
if genes_found_py == r_core_genes:
    common_order = genes_found_py
    py_core_vals = core_matrix_py.loc[common_order].values.astype(float)
    r_core_vals  = r_core_norm.loc[common_order].values.astype(float)
    core_match   = np.allclose(py_core_vals, r_core_vals, atol=1e-6)
    print(f"  Core matrix values match: {core_match}")
    if not core_match:
        diff = np.abs(py_core_vals - r_core_vals)
        print(f"  Core matrix max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}")

# Show per-gene std across samples (this drives the scaling step)
print()
print(f"  {'Gene':<10} {'Python row-std':>16} {'R row-std':>14} {'match':>8}")
print(f"  {'-'*10} {'-'*16} {'-'*14} {'-'*8}")
for gene in genes_found_py:
    if gene in r_core_norm.index:
        py_std = float(core_matrix_py.loc[gene].values.astype(float).std(ddof=1))
        r_std  = float(r_core_norm.loc[gene].values.astype(float).std(ddof=1))
        match  = abs(py_std - r_std) < 1e-6
        print(f"  {gene:<10} {py_std:>16.6f} {r_std:>14.6f} {'OK' if match else 'DIFF':>8}")

# Check the column-scaled matrix that goes into SVD
core_vals  = core_matrix_py.values.astype(float)
centred    = core_vals - core_vals.mean(axis=1, keepdims=True)
col_std    = centred.std(axis=0, ddof=1)
print(f"\n  Column std (across {core_vals.shape[0]} genes) stats:")
print(f"    min={col_std.min():.4f}  max={col_std.max():.4f}  "
      f"mean={col_std.mean():.4f}  zeros={(col_std==0).sum()}")

from scipy.linalg import svd as _svd
col_std[col_std == 0] = 1.0
scaled_py = centred / col_std
_, s_py, _ = _svd(scaled_py, full_matrices=False)
var_py_true = s_py**2 / (s_py**2).sum()
print(f"  Variance from scipy full SVD (Python data): {np.round(var_py_true[:4], 4)}")

# ---------------------------------------------------------------------------
# 6. CPCA checks
# ---------------------------------------------------------------------------
print("\n=== CPCA Comparison ===")

results["Sample order matches"] = check(
    "Sample order matches exactly",
    np.array_equal(cpca_result.sample_order, r_sample_order),
    f"first mismatches: "
    f"{np.where(cpca_result.sample_order != r_sample_order)[0][:5].tolist()}"
    if not np.array_equal(cpca_result.sample_order, r_sample_order) else ""
)

close_scale = np.allclose(cpca_result.circular_scale, r_circ_scale, atol=ATOL)
results["Circular scale matches"] = check(
    f"Circular scale matches (atol={ATOL:.0e})",
    close_scale,
    max_diff_str(cpca_result.circular_scale, r_circ_scale)
    if not close_scale else ""
)

# SVD sign can flip: compare absolute values for PC loadings
for name, py_pc, r_col in [
    ("PC1", cpca_result.pc1, r_pc["pc1"].values),
    ("PC2", cpca_result.pc2, r_pc["pc2"].values),
    ("PC3", cpca_result.pc3, r_pc["pc3"].values),
]:
    close_pc = np.allclose(np.abs(py_pc), np.abs(r_col), atol=ATOL)
    results[f"{name} loadings match"] = check(
        f"{name} loadings match |abs| (atol={ATOL:.0e})",
        close_pc,
        max_diff_str(np.abs(py_pc), np.abs(r_col)) if not close_pc else ""
    )

close_var = np.allclose(cpca_result.variance_explained, r_variance, atol=ATOL_VAR)
results["Variance explained matches"] = check(
    f"Variance explained matches (atol={ATOL_VAR:.0e})",
    close_var,
    f"Python={np.round(cpca_result.variance_explained, 5)}  "
    f"R={np.round(r_variance, 5)}"
)

results["Outlier candidates match"] = check(
    "Outlier candidates match (same 8 samples)",
    set(cpca_result.outlier_candidate_idx.tolist()) == set(r_candidates.tolist()),
    f"Python={sorted(cpca_result.outlier_candidate_idx)}  "
    f"R={sorted(r_candidates)}"
)

# ---------------------------------------------------------------------------
# 7. Timing comparison
# ---------------------------------------------------------------------------
print("\n=== Timing Comparison ===")

r_timing_dict = dict(zip(r_timing["step"], r_timing["seconds"]))

step_labels = {
    "drop_unnamed" : "drop_unnamed",
    "drop_sparse"  : "drop_sparse",
    "resolve_dupes": "resolve_dupes",
    "normalise"    : "normalise",
    "cpca"         : "cpca",
}

col_w = 22
print(f"\n  {'Step':<{col_w}} {'Python (s)':>12} {'R (s)':>10} {'Speedup':>10}")
print(f"  {'-'*col_w} {'-'*12} {'-'*10} {'-'*10}")

total_py, total_r = 0.0, 0.0

for py_key, r_key in step_labels.items():
    py_t = timing.get(py_key, float("nan"))
    r_t  = r_timing_dict.get(r_key, float("nan"))
    speedup = r_t / py_t if (py_t > 0 and not np.isnan(r_t)) else float("nan")
    total_py += py_t if not np.isnan(py_t) else 0
    total_r  += r_t  if not np.isnan(r_t)  else 0
    speedup_str = f"{speedup:.1f}x" if not np.isnan(speedup) else "n/a"
    print(f"  {py_key:<{col_w}} {py_t:>12.4f} {r_t:>10.3f} {speedup_str:>10}")

print(f"  {'─'*col_w} {'─'*12} {'─'*10} {'─'*10}")
total_speedup = total_r / total_py if total_py > 0 else float("nan")
print(f"  {'TOTAL (excl. load)':<{col_w}} {total_py:>12.4f} {total_r:>10.3f} "
      f"{total_speedup:>9.1f}x")
print(f"\n  (Python load time: {t_load:.3f}s from parquet)")

# ---------------------------------------------------------------------------
# 8. Final summary
# ---------------------------------------------------------------------------
n_pass  = sum(results.values())
n_total = len(results)

print(f"\n{'='*52}")
print(f"RESULT: {n_pass}/{n_total} checks passed")
print(f"{'='*52}")

if n_pass < n_total:
    print("Failed checks:")
    for label, passed in results.items():
        if not passed:
            print(f"  ✗  {label}")
    sys.exit(1)
else:
    print("✓  Python output matches R on all checks.")