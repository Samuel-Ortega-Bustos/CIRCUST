#!/usr/bin/env python3
"""
validation/validate_full.py
============================
Validacion completa del pipeline CIRCUST (Etapas 1-4) con comparacion
de tiempos frente a la implementacion original en R.

Etapas 1-2 (Preprocesamiento → Orden preliminar)
    Comparacion exacta/tolerada contra los CSVs de referencia generados
    por ``rscripts/validate_export.R``.

Etapas 3-4 (Candidatos → Estimacion robusta)
    Smoke-tests de integridad (dimensiones, rangos, sin NaN/Inf).
    No existe referencia R exportada para estas etapas — son Python-only.

Uso rapido
----------
    # Dataset mini (rapido, ~1 min) — solo smoke tests, sin comparacion R
    python validation/validate_full.py --dataset mini

    # Dataset sub (medio)
    python validation/validate_full.py --dataset sub

    # Dataset completo + comparacion exacta contra referencia R
    python validation/validate_full.py --dataset full

    # Benchmark paralelo: comparar n_jobs=1 vs n_jobs=-1 en Etapa 4
    python validation/validate_full.py --dataset sub --benchmark-jobs

Requisitos (solo para --dataset full)
    Rscript rscripts/validate_export.R   # genera validation/reference/*.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ruta de proyecto ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from circust.preprocessing   import Preprocessor, load_expression_matrix
from circust.cpca             import CPCA
from circust.outlier          import OutlierRefiner
from circust.synchronizer     import CircularSynchronizer
from circust.nonparametric    import NonParametricScorer
from circust.candidate_selection import CandidateSelector
from circust.reference_set    import ReferenceSetBuilder
from circust.random_selection import RandomSelector
from circust.top_matrix       import build_top_matrix
from circust.robust_estimation import RobustEstimator
from circust.core_genes        import SEED_GENES_LARRIBA

CORE_GENES = SEED_GENES_LARRIBA
REF_DIR    = REPO_ROOT / "validation" / "reference"

DATASETS = {
    "mini": REPO_ROOT / "data" / "matrixIn_mini.parquet",
    "sub":  REPO_ROOT / "data" / "matrixIn_sub.parquet",
    "full": REPO_ROOT / "data" / "matrixIn.parquet",
}

# ── Tolerancias ───────────────────────────────────────────────────────────────
TOL_EXACT    = 0      # enteros — coincidencia exacta
TOL_STRICT   = 1e-10  # normalizacion / aritmetica simple
TOL_FMM      = 5e-2   # FMM — no convexo, puede converger a optimos distintos


# =============================================================================
# Informe de validacion
# =============================================================================

class Report:
    """Acumula checks y genera resumen final."""

    def __init__(self, title: str):
        self.title  = title
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))
        tag = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
        msg = f"  [{tag}] {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)

    def summary(self) -> bool:
        total  = len(self.checks)
        passed = sum(1 for _, ok, _ in self.checks if ok)
        failed = total - passed
        bar    = "=" * 60
        print(f"\n{bar}")
        print(f"  {self.title}: {passed}/{total} checks OK, {failed} fallidos")
        if failed:
            print("  Checks fallidos:")
            for name, ok, detail in self.checks:
                if not ok:
                    print(f"    ✗ {name}: {detail}")
        print(bar)
        return failed == 0


def _load_ref(name: str) -> pd.DataFrame:
    return pd.read_csv(REF_DIR / name)


def _cmp_arrays(py: np.ndarray, r: np.ndarray,
                tol: float, name: str, rep: Report):
    if py.shape != r.shape:
        rep.check(name, False, f"shape py={py.shape} r={r.shape}")
        return
    if tol == 0:
        diff = int(np.sum(py != r))
        rep.check(name, diff == 0, f"{diff} discrepancias")
    else:
        md = float(np.max(np.abs(py - r)))
        rep.check(name, md < tol, f"max_diff={md:.2e} tol={tol:.0e}")


def _cmp_sets(py: set, r: set, name: str, rep: Report):
    rep.check(name, py == r,
              "" if py == r else f"solo_py={py-r} solo_r={r-py}")


# =============================================================================
# Etapa 1-2: Validacion contra referencia R
# =============================================================================

def validate_stages_1_2(
    matrix: pd.DataFrame,
    rep: Report,
    check_ref: bool = False,
) -> dict:
    """Ejecuta y valida Etapas 1-2. Devuelve objetos intermedios."""
    timings: dict[str, float] = {}

    # ── Preprocesamiento ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ETAPA 1: PREPROCESAMIENTO")
    print("=" * 60)

    t0   = time.perf_counter()
    prep = Preprocessor(verbose=False).run(matrix)
    timings["preprocessing"] = time.perf_counter() - t0
    print(f"  {prep.n_genes_out} genes × {prep.n_samples} muestras  "
          f"({timings['preprocessing']:.2f}s)")

    if check_ref:
        _cmp_sets(set(prep.dropped_sparse),
                  set(_load_ref("r_dropped_sparse.csv")["gene"].dropna()),
                  "Genes eliminados (sparse)", rep)

        _cmp_sets(set(prep.dropped_duplicates),
                  set(_load_ref("r_dropped_dupes.csv")["gene"].dropna()),
                  "Genes eliminados (duplicados)", rep)

        r_norm = pd.read_csv(REF_DIR / "r_expr_norm.csv", index_col=0)
        rep.check("Dimensiones matriz normalizada",
                  prep.expr_norm.shape == r_norm.shape,
                  f"py={prep.expr_norm.shape} r={r_norm.shape}")

        common = sorted(set(prep.expr_norm.index) & set(r_norm.index))[:50]
        if common:
            _cmp_arrays(prep.expr_norm.loc[common].values,
                        r_norm.loc[common].values,
                        TOL_STRICT, "Valores normalizados (50 genes)", rep)

        r_core = pd.read_csv(REF_DIR / "r_core_norm.csv", index_col=0)
        py_core = prep.expr_norm.loc[
            [g for g in CORE_GENES if g in prep.expr_norm.index]
        ]
        _cmp_arrays(py_core.values, r_core.values,
                    TOL_STRICT, "Genes core normalizados", rep)

    # ── CPCA ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CPCA")
    print("=" * 60)

    t0   = time.perf_counter()
    cpca = CPCA(core_genes=CORE_GENES, verbose=False).run(prep.expr_norm)
    timings["cpca"] = time.perf_counter() - t0
    print(f"  CPCA: {timings['cpca']:.2f}s")

    if check_ref:
        r_order = _load_ref("r_sample_order.csv")["sample_order"].values - 1
        _cmp_arrays(cpca.sample_order, r_order, TOL_EXACT,
                    "Orden muestras CPCA", rep)

        r_scale = _load_ref("r_circular_scale.csv")["circular_scale"].values
        _cmp_arrays(cpca.circular_scale, r_scale, TOL_STRICT,
                    "Escala circular CPCA", rep)

        r_var = _load_ref("r_variance.csv")["variance_explained"].values
        _cmp_arrays(cpca.variance_explained, r_var, TOL_STRICT,
                    "Varianza explicada", rep)

        r_pc = _load_ref("r_pc_loadings.csv")
        for col in ["pc1", "pc2", "pc3"]:
            py_v = getattr(cpca, col)
            r_v  = r_pc[col].values
            corr = float(np.corrcoef(py_v, r_v)[0, 1])
            rep.check(f"Correlacion {col.upper()}",
                      abs(corr) > 0.999, f"|corr|={abs(corr):.6f}")

    # ── Deteccion de outliers + re-CPCA ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DETECCION DE OUTLIERS + RE-CPCA")
    print("=" * 60)

    t0      = time.perf_counter()
    refined = OutlierRefiner(verbose=False).run(cpca, prep.expr_norm)
    timings["outlier"] = time.perf_counter() - t0
    print(f"  Outliers + re-CPCA: {timings['outlier']:.2f}s")
    print(refined.summary())

    if check_ref:
        r_outs = _load_ref("r_outlier_samples.csv")["sample_idx"].values
        r_outs_set = set(r_outs - 1) if len(r_outs) > 0 else set()
        _cmp_sets(set(refined.samples_dropped), r_outs_set,
                  "Muestras outlier", rep)

        r_order2 = _load_ref("r_sample_order_2.csv")["sample_order"].values - 1
        _cmp_arrays(refined.cpca_final.sample_order, r_order2, TOL_EXACT,
                    "Orden muestras CPCA2", rep)

        r_scale2 = _load_ref("r_circular_scale_2.csv")["circular_scale"].values
        _cmp_arrays(refined.cpca_final.circular_scale, r_scale2, TOL_STRICT,
                    "Escala circular CPCA2", rep)

        r_peaks_after = _load_ref("r_peaks_after.csv")
        for _, row in r_peaks_after.iterrows():
            gene = row["gene"]
            if gene in refined.fmm_peak_times_final:
                py_pk = refined.fmm_peak_times_final[gene]
                r_pk  = row["fmm_peak"]
                diff  = min(abs(py_pk - r_pk), 2 * pi - abs(py_pk - r_pk))
                rep.check(f"FMM peak (post-outlier): {gene}",
                           diff < TOL_FMM,
                           f"py={py_pk:.4f} r={r_pk:.4f} diff={diff:.4f}")

        r_r2 = pd.read_csv(REF_DIR / "r_r2_after.csv", index_col=0)
        for gene in CORE_GENES:
            if gene in refined.fmm_fits_final:
                py_r2 = refined.fmm_fits_final[gene].r2
                r_r2v = float(r_r2.loc[gene, "r2_fmm"])
                rep.check(f"R² (post-outlier): {gene}",
                           abs(py_r2 - r_r2v) < TOL_FMM,
                           f"py={py_r2:.4f} r={r_r2v:.4f}")

    # ── Orden preliminar ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ETAPA 2: ORDEN CIRCULAR PRELIMINAR")
    print("=" * 60)

    t0        = time.perf_counter()
    pre_order = CircularSynchronizer(verbose=False).run(refined, CORE_GENES)
    timings["preliminary_order"] = time.perf_counter() - t0
    print(f"  Orden preliminar: {timings['preliminary_order']:.2f}s")
    print(pre_order.summary())

    if check_ref:
        r_fin_scale = _load_ref("r_final_circular_scale.csv")["circular_scale"].values
        _cmp_arrays(pre_order.circular_scale, r_fin_scale, TOL_FMM,
                    "Escala circular final", rep)

        r_fin_dir = _load_ref("r_final_direction.csv")["direction_flipped"].values[0]
        r_fin_dir_bool = str(r_fin_dir).upper() == "TRUE"
        rep.check("Direccion final (flipped)",
                  pre_order.direction_flipped == r_fin_dir_bool,
                  f"py={pre_order.direction_flipped} r={r_fin_dir_bool}")

        r_final_peaks = _load_ref("r_final_peaks.csv")
        for _, row in r_final_peaks.iterrows():
            gene = row["gene"]
            idx  = CORE_GENES.index(gene) if gene in CORE_GENES else -1
            if idx >= 0:
                py_pk = pre_order.peak_times[idx]
                r_pk  = row["peak_time"]
                diff  = min(abs(py_pk - r_pk), 2 * pi - abs(py_pk - r_pk))
                rep.check(f"Pico final: {gene}",
                           diff < TOL_FMM,
                           f"py={py_pk:.4f} r={r_pk:.4f} diff={diff:.4f}")

    return {
        "prep":      prep,
        "cpca":      cpca,
        "refined":   refined,
        "pre_order": pre_order,
        "timings":   timings,
    }


# =============================================================================
# Etapas 3-4: Smoke-tests
# =============================================================================

def smoke_test_stages_3_4(
    results_12: dict,
    rep: Report,
    n_jobs: int = 1,
) -> dict:
    """
    Ejecuta Etapas 3-4 y verifica integridad basica del resultado.
    No compara contra R (no hay referencia exportada).
    """
    pre_order = results_12["pre_order"]
    timings   = {}

    n_samp = pre_order.expr_ordered.shape[1]

    # ── NP scorer ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ETAPA 3.1: AJUSTE NO PARAMETRICO (NP)")
    print("=" * 60)

    t0       = time.perf_counter()
    np_res   = NonParametricScorer(verbose=False).run(pre_order.expr_ordered)
    timings["np_scorer"] = time.perf_counter() - t0
    print(f"  NP scorer: {timings['np_scorer']:.2f}s  ({len(np_res.gene_names)} genes)")
    rep.check("NP scorer sin NaN en R²",
              not np.any(np.isnan(np_res.r2)),
              f"genes={len(np_res.gene_names)}")

    # ── Seleccion de candidatos ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ETAPA 3.2: SELECCION DE CANDIDATOS")
    print("=" * 60)

    t0   = time.perf_counter()
    csel = CandidateSelector(verbose=False, n_jobs=1).run(
        r2_np          = np_res.r2,
        expr_norm      = pre_order.expr_ordered,
        circular_scale = pre_order.circular_scale,
        r2_par_core    = pre_order.r2_fmm,
    )
    timings["candidate_selection"] = time.perf_counter() - t0
    print(f"  CandidateSelector: {timings['candidate_selection']:.2f}s")
    n_cand = len(csel.gene_names)
    rep.check("Candidatos seleccionados >= 1", n_cand >= 1, f"n={n_cand}")
    rep.check("Sin NaN en r2_par candidatos",
              not np.any(np.isnan(csel.r2_par)), "")

    # ── Conjunto de referencia ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ETAPA 3.3: CONJUNTO DE REFERENCIA")
    print("=" * 60)

    t0   = time.perf_counter()
    rset = ReferenceSetBuilder(verbose=False).run(
        candidate      = csel,
        expr_full_norm = pre_order.expr_ordered,
        circular_scale = pre_order.circular_scale,
        sample_order   = pre_order.sample_order,
        r2_np_all      = np_res.r2,
        gene_names_all = np_res.gene_names,
    )
    timings["reference_set"] = time.perf_counter() - t0
    print(f"  ReferenceSetBuilder: {timings['reference_set']:.2f}s")
    rep.check("Conjunto de referencia no vacio",
              len(rset.gene_names) > 0,
              f"n={len(rset.gene_names)}")

    # ── Seleccion aleatoria ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ETAPA 3.5: SELECCION ALEATORIA")
    print("=" * 60)

    t0   = time.perf_counter()
    rsel = RandomSelector(n_reps=5, seed=42, verbose=False).run(
        reference      = rset,
        expr_full_norm = pre_order.expr_ordered,
    )
    timings["random_selection"] = time.perf_counter() - t0
    K = rsel.selection_indices.shape[0]
    print(f"  RandomSelector: {timings['random_selection']:.2f}s  K={K}")
    rep.check("5 repeticiones completadas", K == 5, f"K={K}")

    # ── Matriz TOP ────────────────────────────────────────────────────────────
    t0      = time.perf_counter()
    core_found = [g for g in CORE_GENES if g in pre_order.expr_ordered.index]
    top_mat = build_top_matrix(
        core_genes         = core_found,
        candidate_names    = rset.gene_names,
        expr_full_norm     = pre_order.expr_ordered,
        candidate_norm_ord = rset.candidate_matrix,
        sample_order       = np.arange(pre_order.expr_ordered.shape[1]),
        genes_to_exclude   = rset.added_genes,
    )
    timings["top_matrix"] = time.perf_counter() - t0
    print(f"  TOP matrix: {top_mat.shape}  ({timings['top_matrix']:.2f}s)")
    rep.check("TOP matrix no vacia", top_mat.shape[0] > 0,
              f"shape={top_mat.shape}")

    # ── Estimacion robusta (n_jobs parametrizable) ────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ETAPA 4: ESTIMACION ROBUSTA  (n_jobs={n_jobs})")
    print("=" * 60)

    t0  = time.perf_counter()
    rob = RobustEstimator(n_jobs=n_jobs, verbose=True).run(
        random_selection = rsel,
        top_matrix       = top_mat,
        expr_full_norm   = pre_order.expr_ordered,
        core_genes       = core_found,
    )
    timings["robust_estimation"] = time.perf_counter() - t0
    print(f"  RobustEstimator: {timings['robust_estimation']:.2f}s")

    rep.check("stats_table shape correcta",
              rob.stats_table.shape == (K * top_mat.shape[0], 25),
              f"shape={rob.stats_table.shape}")
    rep.check("Sin NaN en stats_table",
              not np.any(np.isnan(rob.stats_table)), "")
    rep.check("consensus_phase en [0, 2π)",
              bool(np.all(rob.consensus_phase >= 0) and
                   np.all(rob.consensus_phase < 2 * pi)),
              f"min={rob.consensus_phase.min():.4f} max={rob.consensus_phase.max():.4f}")
    rep.check("sample_orders forma correcta",
              rob.sample_orders.shape == (K, n_samp),
              f"shape={rob.sample_orders.shape}")
    print(rob.summary())

    return {**timings, "robust_result": rob}


# =============================================================================
# Benchmark n_jobs
# =============================================================================

def benchmark_jobs(results_12: dict, n_reps_bench: int = 3):
    """Compara tiempo de Etapa 4 con n_jobs=1 vs n_jobs=-1."""
    pre_order = results_12["pre_order"]

    np_res = NonParametricScorer(verbose=False).run(pre_order.expr_ordered)
    csel   = CandidateSelector(verbose=False, n_jobs=1).run(
        r2_np          = np_res.r2,
        expr_norm      = pre_order.expr_ordered,
        circular_scale = pre_order.circular_scale,
        r2_par_core    = pre_order.r2_fmm,
    )
    rset = ReferenceSetBuilder(verbose=False).run(
        candidate      = csel,
        expr_full_norm = pre_order.expr_ordered,
        circular_scale = pre_order.circular_scale,
        sample_order   = pre_order.sample_order,
        r2_np_all      = np_res.r2,
        gene_names_all = np_res.gene_names,
    )
    rsel = RandomSelector(n_reps=n_reps_bench, seed=42, verbose=False).run(
        reference=rset, expr_full_norm=pre_order.expr_ordered
    )
    core_found = [g for g in CORE_GENES if g in pre_order.expr_ordered.index]
    top_mat = build_top_matrix(
        core_genes         = core_found,
        candidate_names    = rset.gene_names,
        expr_full_norm     = pre_order.expr_ordered,
        candidate_norm_ord = rset.candidate_matrix,
        sample_order       = np.arange(pre_order.expr_ordered.shape[1]),
        genes_to_exclude   = rset.added_genes,
    )

    print(f"\n  TOP matrix: {top_mat.shape}  K={n_reps_bench}")
    times = {}
    for nj, label in [(1, "n_jobs=1 (secuencial)"),
                      (-1, f"n_jobs=-1 ({os.cpu_count()} nucleos)")]:
        t0 = time.perf_counter()
        RobustEstimator(n_jobs=nj, verbose=False).run(
            random_selection = rsel,
            top_matrix       = top_mat,
            expr_full_norm   = pre_order.expr_ordered,
            core_genes       = core_found,
        )
        elapsed = time.perf_counter() - t0
        times[nj] = elapsed
        print(f"  {label}: {elapsed:.2f}s")

    speedup = times[1] / times[-1] if times[-1] > 0 else float("nan")
    print(f"\n  Speedup n_jobs=-1 vs n_jobs=1: {speedup:.2f}x")


# =============================================================================
# Comparacion de tiempos Python vs R
# =============================================================================

def print_timing_comparison(py_timings: dict[str, float]):
    """Compara los tiempos Python con la referencia R exportada."""
    if not (REF_DIR / "r_timing.csv").exists():
        print("\n  (sin referencia de tiempos R — ejecuta validate_export.R)")
        return

    r_timing = pd.read_csv(REF_DIR / "r_timing.csv")
    r_map = dict(zip(r_timing["step"], r_timing["seconds"]))

    # Agrupaciones de pasos R a etapas Python
    r_groups = {
        "preprocessing": ["load", "drop_unnamed", "drop_sparse",
                          "resolve_dupes", "normalise"],
        "cpca":          ["cpca"],
        "outlier":       ["fmm_cosinor", "outliers", "cpca2", "fmm_after"],
        "preliminary_order": ["pre_order", "basic_order"],
    }

    print("\n" + "=" * 60)
    print("  COMPARACION DE TIEMPOS  Python vs R")
    print(f"  {'Etapa':<28} {'Python':>8}  {'R':>8}  {'Ratio':>7}")
    print("  " + "-" * 56)

    total_py = 0.0
    total_r  = 0.0
    for stage, r_steps in r_groups.items():
        py_t = py_timings.get(stage, float("nan"))
        r_t  = sum(r_map.get(s, 0.0) for s in r_steps)
        ratio = py_t / r_t if r_t > 0 and not np.isnan(py_t) else float("nan")
        marker = "  ✓" if ratio < 2 else ("  ~" if ratio < 5 else "  !")
        label = stage.replace("_", " ").title()
        print(f"{marker} {label:<26} {py_t:>7.2f}s  {r_t:>7.2f}s  {ratio:>6.2f}x")
        if not np.isnan(py_t):
            total_py += py_t
            total_r  += r_t

    print("  " + "-" * 56)
    total_ratio = total_py / total_r if total_r > 0 else float("nan")
    print(f"  {'TOTAL':<28} {total_py:>7.2f}s  {total_r:>7.2f}s  {total_ratio:>6.2f}x")
    print("=" * 60)
    print("  Ratio < 1.0 → Python mas rapido que R")
    print("  Ratio > 1.0 → R mas rapido que Python")
    print("  (Tiempos R registrados en la maquina donde se ejecuto validate_export.R)")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mini", "sub", "full"], default="sub",
                   help="Dataset a usar (default: sub)")
    p.add_argument("--skip-stages-34", action="store_true",
                   help="Ejecutar solo Etapas 1-2 (mas rapido)")
    p.add_argument("--benchmark-jobs", action="store_true",
                   help="Comparar n_jobs=1 vs n_jobs=-1 en Etapa 4")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="n_jobs para RobustEstimator (default: 1)")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = DATASETS[args.dataset]
    if not data_path.exists():
        print(f"ERROR: no se encuentra {data_path}")
        sys.exit(1)

    has_ref = REF_DIR.exists() and args.dataset == "full"
    if args.dataset != "full" and REF_DIR.exists():
        print("NOTA: La referencia R se genero con el dataset 'full'.")
        print("      Los checks de Etapas 1-2 se omiten para datasets != full.\n")

    matrix = load_expression_matrix(str(data_path), gene_column="gene_id")
    print(f"Dataset: {args.dataset}  |  {matrix.shape[0]} genes × {matrix.shape[1]} muestras")

    rep12 = Report("Etapas 1-2 (vs R)" if has_ref else "Etapas 1-2 (sin ref)")
    results_12 = validate_stages_1_2(matrix, rep12, check_ref=has_ref)
    all_ok = rep12.summary()

    if not args.skip_stages_34:
        rep34 = Report("Etapas 3-4 (smoke tests)")
        smoke_test_stages_3_4(results_12, rep34, n_jobs=args.n_jobs)
        all_ok = rep34.summary() and all_ok

    if args.benchmark_jobs:
        print("\n" + "=" * 60)
        print("  BENCHMARK: n_jobs=1 vs n_jobs=-1")
        print("=" * 60)
        benchmark_jobs(results_12, n_reps_bench=3)

    if has_ref:
        print_timing_comparison(results_12["timings"])

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
