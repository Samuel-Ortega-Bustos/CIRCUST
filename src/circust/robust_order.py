"""
circust/robust_order.py
========================
Etapa 4: Estimacion robusta del orden circular.

Pipeline en dos fases:

  **Fase A — Etapa comun** (no cambia entre metodos):
      1. Genera K subconjuntos aleatorios de genes TOP (tamano configurable,
         por defecto 2/3 del total) con restricciones de cobertura circular
         y calidad parametrica.
      2. Para cada subconjunto k=1..K ejecuta:
         a) CPCA sobre la submatriz de genes seleccionada.
         b) Sincronizacion biologica (rotacion + orientacion).
         c) Re-ajuste FMM / Cosinor / NP a cada gen del TOP para obtener
            la tabla de estadisticos de 25 columnas.
      3. Calcula la mediana R² (FMM) por repeticion sobre genes de evaluacion.

  **Fase B — Tarea 1: eleccion del orden robusto** (segun ``method``):
      - ``"best_k"``: elige la repeticion k* que maximiza la mediana de R²
        sobre los genes de evaluacion. Sencillo y rapido.
      - ``"aggregate"``: agregacion circular de los K ordenes (Barragan et al.,
        2015) usando el modulo ``circular_aggregation``. Por defecto TSP3.

Posicion en el pipeline
-----------------------
    TopGeneSelector  ->  **RobustOrderEstimator**  ->  (Etapa 5: visualizacion)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from math import pi
from types import SimpleNamespace

from circust.circular_aggregation import aggregate_circular_orders
from circust.cpca                 import CPCA
from circust.fitting.fmm          import FMMModel
from circust.fitting.fmm_stabilized import StabilizedFMMModel, prepare_dataset
from circust.fitting.cosinor     import CosinorModel
from circust.fitting.ori         import circular_unimodal_fit
from circust.synchronizer        import CircularSynchronizer


# ===========================================================================
# Helpers a nivel de modulo (serializables por pickle, reutilizables)
# ===========================================================================

def _mini_cpca_phi(sub: np.ndarray) -> np.ndarray:
    """Calcula la escala circular phi para una submatriz de genes.

    Replica los pasos esenciales de CPCA (centrar por filas, escalar por
    columnas, SVD, proyectar sobre el circulo unitario). Usado por
    ``_random_selection`` para verificar que un subconjunto candidato
    produce una distribucion temporal razonable.
    """
    n_genes = sub.shape[0]
    if n_genes < 2:
        return np.linspace(0, 2 * pi, sub.shape[1], endpoint=False)

    centred = sub - sub.mean(axis=1, keepdims=True)
    col_rms = np.sqrt(np.sum(centred ** 2, axis=0) / max(n_genes - 1, 1))
    col_rms[col_rms == 0] = 1.0
    scaled = centred / col_rms

    _, _, vt = np.linalg.svd(scaled, full_matrices=False)
    pc1 = vt[0]
    pc2 = vt[1] if vt.shape[0] > 1 else np.zeros_like(pc1)

    norm12 = np.sqrt(pc1 ** 2 + pc2 ** 2)
    norm12[norm12 == 0] = 1.0
    xi = pc1 / norm12
    yi = pc2 / norm12

    phi = np.arctan2(yi, xi) % (2.0 * pi)
    return np.sort(phi)


def _max_consecutive_gap(circ_sorted: np.ndarray) -> float:
    """Mayor hueco entre muestras consecutivas (con cierre circular)."""
    n = len(circ_sorted)
    if n < 2:
        return 0.0
    t = circ_sorted / (2.0 * pi) * n
    gaps = np.diff(t)
    wrap_gap = n - t[-1] + t[0]
    return float(max(gaps.max(), wrap_gap))


def _fit_gene(args):
    """Ajusta FMM + Cosinor + NP para un unico gen en una repeticion.

    Funcion a nivel de modulo para serializacion por ``ProcessPoolExecutor``.
    El segundo elemento de ``args`` puede ser:
      - dict de kwargs para ``FMMModel`` (camino "base")
      - un ``StabilizedFMMModel`` ya preparado (camino "stabilized");
        en ese caso se reusa directamente sin construir nada.
    """
    (i, gene, vvv, esc_k, fmm_obj, arntl_peak_raw,
     orientation_changed, order_rot, n_samp) = args

    if isinstance(fmm_obj, StabilizedFMMModel):
        fmm_loc = fmm_obj                         # reusa modelo compartido
    else:
        fmm_loc = FMMModel(**fmm_obj)             # base, construye por gen
    cosm_loc = CosinorModel()

    two_pi = 2.0 * pi

    # ---- FMM ----
    fr = fmm_loc.fit(vvv, esc_k)
    M, A = fr.params["M"], fr.params["A"]
    alpha = fr.params["alpha"]
    beta  = fr.params["beta"]
    omega = fr.params["omega"]
    al = (alpha - arntl_peak_raw + pi) % two_pi
    if not orientation_changed:
        pars_fmm = (M, A, al, beta, omega)
    else:
        pars_fmm = (
            M, A,
            (two_pi - al) % two_pi,
            (two_pi - beta) % two_pi,
            omega,
        )
    pkU = FMMModel.peak_time(pars_fmm[2], pars_fmm[3], pars_fmm[4])
    pkL = (pkU + pi) % two_pi
    peaks_fmm = (pkU, pkL, pkU / two_pi * 100, pkL / two_pi * 100)
    resid = vvv - fr.fitted
    sFMM   = float(np.sum(resid**2) / max(n_samp - 5, 1))
    mseFMM = float(np.sum(resid**2) / n_samp)
    r2FMM  = float(fr.r2)
    if not orientation_changed:
        fitted_fmm_reord = fr.fitted[order_rot]
    else:
        fitted_fmm_reord = fr.fitted[order_rot][::-1]
    stat_fmm = list(pars_fmm) + list(peaks_fmm) + [sFMM, mseFMM, r2FMM]

    # ---- Cosinor ----
    cr = cosm_loc.fit(vvv, esc_k)
    Mc, Ac, phiC = cr.params["M"], cr.params["A"], cr.params["phi"]
    phiC_rot = (phiC - arntl_peak_raw + pi) % two_pi
    if not orientation_changed:
        pars_cos = (Mc, Ac, phiC_rot)
        pk1 = phiC_rot % two_pi
        pk2 = (phiC_rot + pi) % two_pi
    else:
        pars_cos = (Mc, Ac, (two_pi - phiC_rot) % two_pi)
        pk1 = phiC % two_pi
        pk2 = (phiC + pi) % two_pi
    peaks_cos = (pk1, pk2, pk1 / two_pi * 100, pk2 / two_pi * 100)
    resid_c = vvv - cr.fitted
    sCos   = float(np.sum(resid_c**2) / max(n_samp - 3, 1))
    mseCos = float(np.sum(resid_c**2) / n_samp)
    r2Cos  = float(cr.r2)
    if not orientation_changed:
        fitted_cos_reord = cr.fitted[order_rot]
    else:
        fitted_cos_reord = cr.fitted[order_rot][::-1]
    stat_cos = list(pars_cos) + list(peaks_cos) + [sCos, mseCos, r2Cos]

    # ---- NP ----
    vvv_anch = vvv[order_rot]
    if orientation_changed:
        vvv_anch = vvv_anch[::-1]
    npres = circular_unimodal_fit(vvv_anch)
    if npres is not None:
        np_fit = npres[0]
        mse_np = float(npres[1])
    else:
        np_fit = np.full(n_samp, vvv_anch.mean())
        mse_np = float(np.mean((vvv_anch - vvv_anch.mean())**2))
    resid_n = vvv_anch - np_fit
    sNp   = float(np.sum(resid_n**2) / max(n_samp - 3, 1))
    mseNp = float(np.sum(resid_n**2) / n_samp)
    var_y = float(np.var(vvv_anch))
    r2Np  = float(1.0 - mse_np / var_y) if var_y > 0 else 0.0
    stat_np = [sNp, mseNp, r2Np]

    return (i, gene, stat_fmm, stat_cos, stat_np,
            fitted_fmm_reord, fitted_cos_reord, np_fit)


# ===========================================================================
# Resultado
# ===========================================================================

@dataclass
class RobustOrderResult:
    """Salida de :class:`RobustOrderEstimator`.

    Atributos -- orden final
    ------------------------
    sample_order : np.ndarray int, shape (n_muestras,)
        Orden de muestras final (best_k o agregacion).
    circular_scale : np.ndarray, shape (n_muestras,)
        Escala circular del orden final.
    method_used : str
        ``"best_k"`` o ``"aggregate"``.
    aggregation_method : str | None
        ``"tsp3"`` / ``"hods"`` si method=aggregate, None si best_k.
    selected_rep : int | None
        Indice (0-based) de la repeticion seleccionada (solo en best_k).
    selection_score : float | None
        Mediana R² de la rep seleccionada (solo en best_k).
    aggregation_msce, aggregation_cktau : float | None
        Diagnostico de calidad del consenso (solo en aggregate).

    Atributos -- por repeticion (diagnostico)
    -----------------------------------------
    sample_orders : np.ndarray int, shape (K, n_muestras)
    circular_scales : np.ndarray, shape (K, n_muestras)
    direction_flipped : np.ndarray bool, shape (K,)
    stats_table : np.ndarray, shape (K * n_top, 25)
    per_rep_fmm_fit, per_rep_cos_fit, per_rep_np_fit : list de pd.DataFrame
    median_r2_per_rep : np.ndarray, shape (K,)

    Atributos -- seleccion aleatoria
    --------------------------------
    selection_names : np.ndarray object, shape (K, tam)
    n_attempts : int
    fallback_used : bool
    top_gene_names : list[str]
    """

    # ── Orden final ──────────────────────────────────────────────────────
    sample_order:        np.ndarray
    circular_scale:      np.ndarray
    method_used:         str
    aggregation_method:  str | None        = None
    selected_rep:        int | None        = None
    selection_score:     float | None      = None
    aggregation_msce:    float | None      = None
    aggregation_cktau:   float | None      = None

    # ── Por repeticion ────────────────────────────────────────────────────
    sample_orders:     np.ndarray = field(default_factory=lambda: np.array([]))
    circular_scales:   np.ndarray = field(default_factory=lambda: np.array([]))
    direction_flipped: np.ndarray = field(default_factory=lambda: np.array([]))
    stats_table:       np.ndarray = field(default_factory=lambda: np.zeros((0, 25)))
    per_rep_fmm_fit:   list       = field(default_factory=list)
    per_rep_cos_fit:   list       = field(default_factory=list)
    per_rep_np_fit:    list       = field(default_factory=list)
    median_r2_per_rep: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Seleccion aleatoria ──────────────────────────────────────────────
    selection_names: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=object))
    n_attempts:      int        = 0
    fallback_used:   bool       = False
    top_gene_names:  list       = field(default_factory=list)

    def summary(self) -> str:
        K = self.sample_orders.shape[0] if self.sample_orders.ndim == 2 else 0
        n_top = len(self.top_gene_names)
        flipped = int(self.direction_flipped.sum()) if len(self.direction_flipped) > 0 else 0
        lines = [
            "=== Resumen de Orden Robusto ===",
            f"  Repeticiones (K)         : {K}",
            f"  Genes en TOP             : {n_top}",
            f"  Metodo de seleccion      : {self.method_used}",
        ]
        if self.method_used == "aggregate":
            lines.append(f"  Metodo de agregacion     : {self.aggregation_method}")
            if self.aggregation_msce is not None:
                lines.append(f"  MSCE consenso            : {self.aggregation_msce:.4f}")
            if self.aggregation_cktau is not None:
                lines.append(f"  CKTau consenso           : {self.aggregation_cktau:.4f}")
        elif self.method_used == "best_k" and self.selected_rep is not None:
            lines.append(f"  Repeticion seleccionada  : {self.selected_rep + 1}")
            lines.append(f"  Mediana R² (seleccion)   : {self.selection_score:.4f}")
        lines.append(f"  Repeticiones con flip    : {flipped}/{K}")
        if len(self.median_r2_per_rep) > 0:
            lines.append(
                f"  Mediana R² por rep       : "
                f"{np.round(self.median_r2_per_rep, 4).tolist()}"
            )
        lines.append(f"  Intentos muestreo        : {self.n_attempts}")
        lines.append(f"  Fallback activado        : {self.fallback_used}")
        return "\n".join(lines)


# ===========================================================================
# RobustOrderEstimator
# ===========================================================================

class RobustOrderEstimator:
    """Etapa 4 de CIRCUST: estimacion robusta del orden circular.

    Parameters
    ----------
    n_reps : int
        Numero K de repeticiones aleatorias. Por defecto 5.
    sample_size_fraction : float
        Fraccion de genes TOP por extraccion. Por defecto 2/3.
    method : str
        Tarea 1 — eleccion del orden final:
        - ``"best_k"``: maximizar la mediana de R² (FMM) sobre genes de evaluacion.
        - ``"aggregate"``: agregar los K ordenes via ``circular_aggregation``.
    aggregation_method : str
        Solo si method=aggregate. ``"tsp3"`` (default) o ``"hods"``.
    r2_min : float
        R² parametrico minimo en cada extraccion. Por defecto 0.5.
    max_attempts : int
        Limite de intentos de muestreo. Por defecto 5000.
    anchor_gene, direction_gene, consistency_gene : str
        Genes ancla para sincronizacion.
    fmm_length_alpha_grid, fmm_length_omega_grid, fmm_num_reps : int
        Hiperparametros FMM base.
    fmm_method : str
        ``"base"`` (FMMModel) o ``"stabilized"`` (StabilizedFMMModel).
    fmm_stab_kwargs : dict | None
        Kwargs adicionales para ``prepare_dataset`` cuando stab.
    lambda_star : float | None
        Lambda heredado de CPCA (modo stab) — evita recalibrar.
    recalibrate_lambda_per_rep : bool
        Si True (default), recalibra lambda por rep. Si False, reusa
        ``lambda_star``.
    n_jobs : int
        Workers para ajuste paralelo de genes (-1 = todos).
    seed : int | None
        Semilla para reproducibilidad.
    verbose : bool
        Mensajes de progreso.
    """

    def __init__(
        self,
        n_reps:                int   = 5,
        sample_size_fraction:  float = 2.0 / 3.0,
        method:                str   = "aggregate",
        aggregation_method:    str   = "tsp3",
        r2_min:                float = 0.5,
        max_attempts:          int   = 5000,
        anchor_gene:           str   = "ARNTL",
        direction_gene:        str   = "DBP",
        consistency_gene:      str   = "CRY1",
        fmm_length_alpha_grid: int   = 48,
        fmm_length_omega_grid: int   = 24,
        fmm_num_reps:          int   = 3,
        fmm_method:            str   = "base",
        fmm_stab_kwargs:       dict|None  = None,
        lambda_star:           float|None = None,
        recalibrate_lambda_per_rep: bool   = True,
        n_jobs:                int   = 1,
        seed:                  int|None   = None,
        verbose:               bool  = True,
    ) -> None:
        if method not in ("best_k", "aggregate"):
            raise ValueError(f"method debe ser 'best_k' o 'aggregate', recibido: {method!r}")
        if aggregation_method not in ("tsp3", "hods"):
            raise ValueError(f"aggregation_method debe ser 'tsp3' o 'hods', recibido: {aggregation_method!r}")
        if fmm_method not in ("base", "stabilized"):
            raise ValueError(f"fmm_method debe ser 'base' o 'stabilized', recibido: {fmm_method!r}")

        self.n_reps               = n_reps
        self.sample_size_fraction = sample_size_fraction
        self.method               = method
        self.aggregation_method   = aggregation_method
        self.r2_min               = r2_min
        self.max_attempts         = max_attempts
        self.anchor_gene          = anchor_gene
        self.direction_gene       = direction_gene
        self.consistency_gene     = consistency_gene
        self._fmm_kwargs = dict(
            length_alpha_grid=fmm_length_alpha_grid,
            length_omega_grid=fmm_length_omega_grid,
            num_reps=fmm_num_reps,
        )
        self.fmm_method      = fmm_method
        self.fmm_stab_kwargs = dict(fmm_stab_kwargs or {})
        self.lambda_star_inherited      = lambda_star
        self.recalibrate_lambda_per_rep = recalibrate_lambda_per_rep
        self.n_jobs  = n_jobs
        self.seed    = seed
        self.verbose = verbose

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(
        self,
        top_result,
        expr_full_norm: pd.DataFrame,
        core_genes:     list[str],
        eval_genes:     list[str] | None = None,
    ) -> RobustOrderResult:
        """Ejecuta la estimacion robusta completa.

        Parameters
        ----------
        top_result : TopGeneResult
            Salida de ``TopGeneSelector.run()``.
        expr_full_norm : pd.DataFrame (n_genes, n_muestras)
            Matriz normalizada completa, en el orden circular preliminar.
        core_genes : list[str]
            Genes reloj centrales.
        eval_genes : list[str] | None
            Genes para evaluar la mediana R² por rep. Default: ``core_genes``.
        """
        self._log("=== Etapa 4: Estimacion de Orden Robusto ===")
        self._log(f"  Metodo: {self.method}  |  K={self.n_reps}  |  "
                  f"fraccion={self.sample_size_fraction:.2f}")
        if self.method == "aggregate":
            self._log(f"  Agregacion: {self.aggregation_method}")

        if eval_genes is None:
            eval_genes = list(core_genes)

        top_matrix = top_result.candidate_matrix
        names_top  = list(top_matrix.index)
        n_top      = len(names_top)
        n_samp     = top_matrix.shape[1]

        # ── Fase A.1: Seleccion aleatoria ───────────────────────────────
        sel_names, n_attempts, fallback = self._random_selection(
            top_result, expr_full_norm,
        )
        K = sel_names.shape[0]

        # ── Fase A.2: CPCA + Sincronizacion + ajustes por rep ─────────
        (sample_orders, circ_scales, flipped_arr,
         stats_table, per_rep_fmm, per_rep_cos, per_rep_np) = \
            self._run_repetitions(
                sel_names, top_matrix, expr_full_norm,
                core_genes, names_top, n_top, n_samp, K,
            )

        # ── Fase A.3: Mediana R² por rep ──────────────────────────────
        median_r2 = self._compute_median_r2(
            stats_table, names_top, eval_genes, K, n_top,
        )
        self._log(f"  Mediana R² por rep: {np.round(median_r2, 4).tolist()}")

        # ── Fase B: Tarea 1 — eleccion del orden ──────────────────────
        if self.method == "best_k":
            final_order, final_scale, sel_rep, sel_score = \
                self._select_best_k_by_r2_median(
                    sample_orders, circ_scales, median_r2,
                )
            return RobustOrderResult(
                sample_order       = final_order,
                circular_scale     = final_scale,
                method_used        = "best_k",
                aggregation_method = None,
                selected_rep       = sel_rep,
                selection_score    = sel_score,
                sample_orders      = sample_orders,
                circular_scales    = circ_scales,
                direction_flipped  = flipped_arr,
                stats_table        = stats_table,
                per_rep_fmm_fit    = per_rep_fmm,
                per_rep_cos_fit    = per_rep_cos,
                per_rep_np_fit     = per_rep_np,
                median_r2_per_rep  = median_r2,
                selection_names    = sel_names,
                n_attempts         = n_attempts,
                fallback_used      = fallback,
                top_gene_names     = names_top,
            )
        else:
            final_order, final_scale, agg_msce, agg_cktau = \
                self._aggregate_orders_via_consensus(
                    sample_orders, circ_scales, n_samp,
                )
            return RobustOrderResult(
                sample_order       = final_order,
                circular_scale     = final_scale,
                method_used        = "aggregate",
                aggregation_method = self.aggregation_method,
                selected_rep       = None,
                selection_score    = None,
                aggregation_msce   = agg_msce,
                aggregation_cktau  = agg_cktau,
                sample_orders      = sample_orders,
                circular_scales    = circ_scales,
                direction_flipped  = flipped_arr,
                stats_table        = stats_table,
                per_rep_fmm_fit    = per_rep_fmm,
                per_rep_cos_fit    = per_rep_cos,
                per_rep_np_fit     = per_rep_np,
                median_r2_per_rep  = median_r2,
                selection_names    = sel_names,
                n_attempts         = n_attempts,
                fallback_used      = fallback,
                top_gene_names     = names_top,
            )

    # ==================================================================
    # Fase A.1 — Seleccion aleatoria controlada
    # ==================================================================

    def _random_selection(
        self,
        top_result,
        expr_full_norm: pd.DataFrame,
    ) -> tuple[np.ndarray, int, bool]:
        """Genera K extracciones aleatorias del TOP con restricciones.

        Restricciones:
          1. Cobertura de cuadrantes (≥3 cuadrantes presentes).
          2. R² minimo (todos los genes con r2 > r2_min, o ancla).
          3. Distancias intra-cuadrante (no demasiado proximos).
          4. Si la distribucion del TOP tiene un hueco grande, comprobar
             que el subconjunto preserva cobertura via mini-CPCA.
        """
        self._log("  --- Seleccion aleatoria controlada ---")

        rng = np.random.default_rng(self.seed)

        names    = np.asarray(top_result.gene_names, dtype=object)
        peaks    = np.asarray(top_result.fmm_peaks,  dtype=np.float64)
        r2       = np.asarray(top_result.r2_par,     dtype=np.float64)
        sectors8 = np.asarray(top_result.sector_labels, dtype=int)
        cos_pk   = np.asarray(top_result.cosinor_peaks, dtype=np.float64)

        cand_matrix_index = list(top_result.candidate_matrix.index)
        cos_pk_by_name = {
            cand_matrix_index[i]: cos_pk[i]
            for i in range(len(cand_matrix_index))
        }
        cos_peaks_ref = np.array(
            [cos_pk_by_name.get(g, 0.0) for g in names],
            dtype=np.float64,
        )

        # Eliminar anclas forzadas del pool
        remove_mask = np.zeros(len(names), dtype=bool)
        for anchor in top_result.added_genes:
            remove_mask |= (names == anchor)

        names_pool    = names[~remove_mask]
        peaks_pool    = peaks[~remove_mask]
        r2_pool       = r2[~remove_mask]
        sectors8_pool = sectors8[~remove_mask]
        cos_pool      = cos_peaks_ref[~remove_mask]

        n_pool = len(names_pool)
        tam    = max(1, int(np.ceil(self.sample_size_fraction * n_pool)))
        if tam > n_pool:
            tam = n_pool

        self._log(f"    Pool: {n_pool} genes  |  tam={tam}  |  K={self.n_reps}")

        # Cuadrantes (sectores → cuadrantes 1..4)
        quadrants = ((sectors8_pool - 1) // 2) + 1
        unique_q  = np.unique(quadrants)
        only_two  = len(unique_q) <= 2
        min_q     = 2 if only_two else 3

        # Hueco maximo (criterio 4)
        circ_pre = np.sort(np.asarray(top_result.circular_scale, dtype=np.float64))
        max_gap_orig = _max_consecutive_gap(circ_pre)
        gaps_pre = np.diff(circ_pre)
        iqr_pre = float(np.subtract(*np.percentile(gaps_pre, [75, 25]))) if len(gaps_pre) else 0.0
        require_cpca_check = max_gap_orig > 1.5 * iqr_pre and iqr_pre > 0

        full_index = expr_full_norm.index

        sel_names    = np.empty((self.n_reps, tam), dtype=object)
        store_sel:    list[np.ndarray] = []
        store_metric: list[float]      = []

        k = 0
        attempts = 0
        fallback = False

        while k < self.n_reps:
            attempts += 1
            sel = rng.choice(n_pool, size=tam, replace=False)

            # Criterio 1: cobertura cuadrantes
            sel_quads = np.unique(quadrants[sel])
            if len(sel_quads) < min_q:
                if attempts >= self.max_attempts:
                    fallback = True
                    break
                continue

            # Criterio 2: R² minimo
            r2_sel = r2_pool[sel]
            min_idx = int(np.argmin(r2_sel))
            high_r2 = r2_sel[min_idx] > self.r2_min
            if not high_r2:
                worst = names_pool[sel[min_idx]]
                if worst in (self.anchor_gene, self.direction_gene):
                    high_r2 = True
            if not high_r2:
                if attempts >= self.max_attempts:
                    fallback = True
                    break
                continue

            # Criterio 3: distancias intra-cuadrante
            cos_sel = cos_pool[sel]
            order_cos = np.argsort(cos_sel)
            cos_sorted = cos_sel[order_cos]
            quad_sorted = quadrants[sel][order_cos]

            diffs = np.empty(len(sel))
            diffs[:-1] = 1.0 - np.cos(cos_sorted[:-1] - cos_sorted[1:])
            diffs[-1]  = 1.0 - np.cos(cos_sorted[-1] - cos_sorted[0])

            ok_quads = True
            for q in (1, 2, 3, 4):
                mask = quad_sorted == q
                if not mask.any():
                    continue
                d_q = diffs[mask]
                tol = max(1, int(np.ceil(0.1 * len(d_q))))
                if d_q.min() <= 1e-5 and (d_q < 1e-5).sum() > tol:
                    ok_quads = False
                    break
            if not ok_quads:
                if attempts >= self.max_attempts:
                    fallback = True
                    break
                continue

            # Criterio 4: chequeo CPCA del subconjunto (opcional)
            cpca_ok = True
            cpca_metric = 0.0
            if require_cpca_check:
                rows_in_full = [
                    g for g in names_pool[sel].tolist()
                    if g in full_index
                ]
                if len(rows_in_full) >= 2:
                    sub = expr_full_norm.loc[rows_in_full].values.astype(np.float64)
                    phi_sub = _mini_cpca_phi(sub)
                    cpca_metric = _max_consecutive_gap(phi_sub)
                    if cpca_metric > max_gap_orig:
                        cpca_ok = False

            if not cpca_ok:
                store_sel.append(sel)
                store_metric.append(cpca_metric)
                if attempts >= self.max_attempts:
                    fallback = True
                    break
                continue

            # Aceptar extraccion
            sel_names[k] = names_pool[sel]
            k += 1
            self._log(f"    Extraccion {k}/{self.n_reps} aceptada tras {attempts} intento(s)")

        # Fallback: si el bucle termino sin completar K, completar con candidatos
        # de criterio-4 fallido (los mejores por hueco). Si tampoco hay, repetir
        # selecciones aleatorias sin restriccion para no devolver Nones.
        if fallback and k < self.n_reps:
            faltan = self.n_reps - k
            self._log(f"    Fallback: {faltan} extracciones por completar.")

            if store_sel:
                order_metric = np.argsort(store_metric)
                use = min(faltan, len(order_metric))
                for j, idx_sorted in enumerate(order_metric[:use]):
                    sel = store_sel[idx_sorted]
                    sel_names[k + j] = names_pool[sel]
                k += use
                faltan = self.n_reps - k

            # Si aun faltan, completar con selecciones aleatorias sin restricciones
            for j in range(faltan):
                sel = rng.choice(n_pool, size=tam, replace=False)
                sel_names[k + j] = names_pool[sel]
            self._log(f"    Fallback completo: {self.n_reps} extracciones aseguradas.")

        self._log(f"    Seleccion completada: {self.n_reps} extracciones en "
                  f"{attempts} intentos (fallback={fallback})")
        return sel_names, attempts, fallback

    # ==================================================================
    # Fase A.2 — CPCA + Sincronizacion por repeticion
    # ==================================================================

    def _run_repetitions(
        self,
        sel_names:      np.ndarray,
        top_matrix:     pd.DataFrame,
        expr_full_norm: pd.DataFrame,
        core_genes:     list[str],
        names_top:      list[str],
        n_top:          int,
        n_samp:         int,
        K:              int,
    ) -> tuple:
        """Por cada rep: CPCA(submatriz) + Synchronizer + ajuste FMM/Cos/NP.

        Las muestras se preservan: el CPCA interno NO elimina outliers
        adicionales (``max_outlier_fraction=0``) — los outliers ya se quitaron
        en el CPCA principal del pipeline. Esto preserva ``n_samp`` constante
        en todas las reps.
        """
        self._log("  --- Ejecutando repeticiones ---")

        # FMM "base" para core genes dentro de la sincronizacion previa
        # (en stab se reasigna por rep porque depende de la escala k)
        fmm = FMMModel(**self._fmm_kwargs) if self.fmm_method == "base" else None
        prelim = CircularSynchronizer(
            anchor_gene      = self.anchor_gene,
            direction_gene   = self.direction_gene,
            consistency_gene = self.consistency_gene,
            verbose          = False,
        )

        sample_orders = np.zeros((K, n_samp), dtype=int)
        circ_scales   = np.zeros((K, n_samp), dtype=np.float64)
        flipped_arr   = np.zeros(K, dtype=bool)
        stats_table   = np.zeros((K * n_top, 25), dtype=np.float64)
        per_rep_fmm   = []
        per_rep_cos   = []
        per_rep_np    = []

        for k in range(K):
            self._log(f"    --- Repeticion {k+1}/{K} ---")
            genes_k = [
                g for g in sel_names[k].tolist()
                if g is not None and g in expr_full_norm.index
            ]
            if len(genes_k) < 2:
                raise ValueError(
                    f"Repeticion {k}: menos de 2 genes disponibles para CPCA. "
                    f"sel_names[{k}] = {sel_names[k].tolist()}"
                )

            # ── 1. CPCA sobre submatriz seleccionada ────────────────────
            # max_outlier_fraction=0: NO elimina muestras adicionales (los
            # outliers ya se quitaron en el CPCA principal del pipeline).
            cpca_stab_kwargs = dict(self.fmm_stab_kwargs)
            if (self.fmm_method == "stabilized"
                    and not self.recalibrate_lambda_per_rep
                    and self.lambda_star_inherited is not None
                    and "lambda_star" not in cpca_stab_kwargs):
                cpca_stab_kwargs["lambda_star"] = float(self.lambda_star_inherited)

            sub_df = expr_full_norm.loc[genes_k]
            cpca_k = CPCA(
                core_genes           = genes_k,
                max_outlier_fraction = 0.0,
                fmm_method           = self.fmm_method,
                fmm_stab_kwargs      = cpca_stab_kwargs,
                verbose              = False,
            ).run(sub_df)
            order_k = cpca_k.sample_order
            esc_k   = cpca_k.circular_scale

            # ── 2. Reordenar la matriz TOP ──────────────────────────────
            top_k = top_matrix.iloc[:, order_k].copy()
            top_k.columns = range(n_samp)

            # ── 3. Construir modelo FMM y ajustar core genes para sync ──
            if self.fmm_method == "stabilized":
                stab_kwargs = dict(self.fmm_stab_kwargs)
                if (not self.recalibrate_lambda_per_rep
                        and self.lambda_star_inherited is not None
                        and "lambda_star" not in stab_kwargs):
                    stab_kwargs["lambda_star"] = float(self.lambda_star_inherited)
                prepared_k = prepare_dataset(esc_k, **stab_kwargs)
                fmm = StabilizedFMMModel(prepared=prepared_k)

            core_present = [g for g in core_genes if g in top_k.index]
            fmm_fits   = {}
            peak_times = {}
            for g in core_present:
                fr = fmm.fit(top_k.loc[g].values.astype(np.float64), esc_k)
                fmm_fits[g]   = fr
                peak_times[g] = fr.peak_time

            # Namespace para CircularSynchronizer (espera CPCAResult-like)
            fake_refined = SimpleNamespace(
                cpca_final           = SimpleNamespace(circular_scale=esc_k),
                circular_scale       = esc_k,
                expr_norm_final      = top_k,
                fmm_fits_final       = fmm_fits,
                fmm_peak_times_final = peak_times,
            )

            (o_pre, esc_pre, mat_pre, peaks_pre, r2_fmm, par_pre,
             names_day, names_night, reversed_21) = prelim._pre_order(
                fake_refined, core_present
            )
            (o_fin, esc_fin, mat_fin, peaks_fin,
             pars_fin, flipped_22, _ind) = prelim._basic_order(
                o_pre, esc_pre, mat_pre, peaks_pre, par_pre, core_present
            )
            orientation_changed = bool(reversed_21 != flipped_22)
            flipped_arr[k] = orientation_changed

            # Orden global (en indices originales 0..n-1)
            global_order = order_k[o_fin]
            sample_orders[k] = global_order
            circ_scales[k]   = esc_fin

            # Pico ARNTL para rotacion de la escala
            arntl_peak_raw = (
                fmm_fits[self.anchor_gene].peak_time
                if self.anchor_gene in fmm_fits else 0.0
            )

            # ── 4. Re-ajuste FMM/Cosinor/NP a cada gen del TOP ──────────
            fmm_mat = np.zeros((n_top, n_samp))
            cos_mat = np.zeros((n_top, n_samp))
            np_mat  = np.zeros((n_top, n_samp))

            esc_rot   = (esc_k - arntl_peak_raw + pi) % (2.0 * pi)
            order_rot = np.argsort(esc_rot)

            # Para stab reusamos el modelo ya preparado; para base, dict de kwargs
            fmm_obj = fmm if self.fmm_method == "stabilized" else self._fmm_kwargs

            gene_args = [
                (
                    i, gene,
                    top_k.loc[gene].values.astype(np.float64),
                    esc_k,
                    fmm_obj,
                    arntl_peak_raw,
                    orientation_changed,
                    order_rot,
                    n_samp,
                )
                for i, gene in enumerate(names_top)
            ]

            n_workers = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
            if n_workers == 1:
                gene_results = [_fit_gene(a) for a in gene_args]
            elif self.fmm_method == "stabilized":
                # Threading: comparte prepared en memoria; BLAS/Numba liberan GIL
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    gene_results = list(pool.map(_fit_gene, gene_args))
            else:
                # Multiproceso: argumentos pequenos por gen
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    gene_results = list(pool.map(_fit_gene, gene_args))

            for res in gene_results:
                (i, gene, stat_fmm, stat_cos, stat_np,
                 fitted_fmm_reord, fitted_cos_reord, np_fit) = res
                fmm_mat[i] = fitted_fmm_reord
                cos_mat[i] = fitted_cos_reord
                np_mat[i]  = np_fit
                stats_table[k * n_top + i] = stat_fmm + stat_cos + stat_np

            per_rep_fmm.append(pd.DataFrame(fmm_mat, index=names_top))
            per_rep_cos.append(pd.DataFrame(cos_mat, index=names_top))
            per_rep_np.append(pd.DataFrame(np_mat,  index=names_top))

        return (sample_orders, circ_scales, flipped_arr,
                stats_table, per_rep_fmm, per_rep_cos, per_rep_np)

    # ==================================================================
    # Fase A.3 — Mediana R² (FMM) por repeticion
    # ==================================================================

    def _compute_median_r2(
        self,
        stats_table: np.ndarray,
        names_top:   list[str],
        eval_genes:  list[str],
        K:           int,
        n_top:       int,
    ) -> np.ndarray:
        """Mediana de R² (FMM) sobre ``eval_genes`` para cada repeticion.

        La columna r2_fmm en ``stats_table`` esta en posicion 11 (0-indexed)
        dentro de las 25 columnas (5 params + 4 peaks + 3 metricas FMM).
        """
        eval_indices = [i for i, name in enumerate(names_top) if name in eval_genes]

        median_r2 = np.zeros(K, dtype=np.float64)
        for k in range(K):
            start = k * n_top
            r2_vals = [stats_table[start + i, 11] for i in eval_indices]
            if r2_vals:
                median_r2[k] = float(np.median(r2_vals))
        return median_r2

    # ==================================================================
    # Fase B — TAREA 1A: best_k por R² mediano
    # ==================================================================

    def _select_best_k_by_r2_median(
        self,
        sample_orders: np.ndarray,
        circ_scales:   np.ndarray,
        median_r2:     np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Elige la repeticion k* que maximiza la mediana de R² (genes core).

        Es la opcion mas simple y rapida: sin agregacion. Devuelve
        ``(sample_order_k*, circular_scale_k*, k*, score_k*)``.
        """
        best_k = int(np.argmax(median_r2))
        self._log(
            f"  best_k: rep {best_k + 1} seleccionada "
            f"(mediana R² = {median_r2[best_k]:.4f})"
        )
        return (
            sample_orders[best_k].copy(),
            circ_scales[best_k].copy(),
            best_k,
            float(median_r2[best_k]),
        )

    # ==================================================================
    # Fase B — TAREA 1B: aggregate via consenso circular
    # ==================================================================

    def _aggregate_orders_via_consensus(
        self,
        sample_orders: np.ndarray,
        circ_scales:   np.ndarray,
        n_samp:        int,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Agrega los K ordenes en uno solo via TSP3 (Barragan et al. 2015).

        Construye la matriz ``Theta`` (K, n) con los angulos asignados a cada
        muestra en cada repeticion, y delega la agregacion al modulo
        ``circular_aggregation``. Despues deriva la escala circular del orden
        consenso como la media circular de las escalas por repeticion.
        """
        K = sample_orders.shape[0]

        # Construir Theta: para la rep k, el angulo asignado a la muestra i es
        # la posicion de i en sample_orders[k] convertida a [0, 2*pi).
        # Usamos la escala circular real (circ_scales) que ya esta sincronizada.
        Theta = np.zeros((K, n_samp), dtype=np.float64)
        for k in range(K):
            # circ_scales[k] esta en orden ascendente y corresponde a
            # sample_orders[k]. Reasignamos: Theta[k, sample_i] = phi
            for pos, sample_i in enumerate(sample_orders[k]):
                Theta[k, sample_i] = circ_scales[k, pos]

        self._log(f"    Construyendo matriz Theta ({K} x {n_samp}) y agregando "
                  f"con method={self.aggregation_method}...")
        out = aggregate_circular_orders(
            Theta=Theta, weights=None, method=self.aggregation_method,
        )

        consensus_order = out["order"]               # permutacion de muestras
        msce_consensus  = out["msce"]
        cktau_consensus = out["cktau"]
        self._log(f"    Consenso: MSCE = {msce_consensus:.4f}, "
                  f"CKTau = {cktau_consensus:.4f}")

        # Derivar la escala circular del orden consenso como media circular
        # de las escalas por repeticion (rotada al orden consenso).
        final_scale = np.zeros(n_samp, dtype=np.float64)
        for pos, sample_j in enumerate(consensus_order):
            angles = []
            for k in range(K):
                # posicion de sample_j en sample_orders[k]
                pos_k = int(np.where(sample_orders[k] == sample_j)[0][0])
                angles.append(circ_scales[k, pos_k])
            angles = np.asarray(angles)
            final_scale[pos] = np.arctan2(
                np.sin(angles).mean(), np.cos(angles).mean()
            ) % (2.0 * pi)

        return consensus_order, final_scale, msce_consensus, cktau_consensus

    # ==================================================================
    # Utilidades
    # ==================================================================

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)
