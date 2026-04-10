"""
circust/robust_estimation.py
============================
Etapa 4: Estimacion Robusta del orden circular.

Para cada una de las K extracciones aleatorias producidas por
:class:`RandomSelector`, esta etapa:

  1. Ejecuta CPCA sobre la submatriz de genes seleccionada (extraida
     de la matriz normalizada completa) para obtener un nuevo orden
     de muestras y una nueva escala circular.

  2. Reordena la matriz TOP segun ese nuevo orden.

  3. Ajusta FMM a cada gen "core" del TOP, aplica el anclaje
     biologico ARNTL/DBP (basicPreOder + basicOder) y obtiene la
     orientacion final.

  4. Re-ajusta FMM, Cosinor y NP a cada fila del TOP en el marco
     anclado, y empaqueta una tabla de 25 columnas por gen y por
     repeticion (parametros + picos + estadisticos).

El resultado es un conjunto de K ordenamientos sincronizados
independientes, listos para promediarse en un consenso circular
robusto.

Equivalente en R
-----------------
``robustEst_v3_cores()`` (lineas 618-808 de ``functionGTEX_cores.R``)
mas la fase de empaquetado de ``robustSincroDBP_v3_cores()``
(lineas 2278-2339).

Posicion en el pipeline
-----------------------
    RandomSelector  →  TopMatrixBuilder  →  **RobustEstimator**
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from math import pi
from types import SimpleNamespace

from circust.cpca import CPCA
from circust.fitting.fmm import FMMModel
from circust.fitting.cosinor import CosinorModel
from circust.nonparametric import circular_unimodal_fit
from circust.preliminary_order import PreliminaryOrderEstimator
from circust.random_selection import RandomSelectionResult


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class RobustEstimationResult:
    """
    Salida de :class:`RobustEstimator`.

    Atributos
    ---------
    sample_orders : np.ndarray int, forma (K, n_muestras)
        Indices de orden de muestras finales por repeticion.
        R: ``oF``.

    circular_scales : np.ndarray, forma (K, n_muestras)
        Escala circular sincronizada por repeticion (en [0, 2π)).
        R: ``pF``.

    direction_flipped : np.ndarray bool, forma (K,)
        Bandera de cambio de orientacion por repeticion.

    stats_table : np.ndarray, forma (K * n_top, 25)
        Tabla apilada por repeticion: por cada gen del TOP, 25
        columnas con parametros FMM (5), picos FMM (4), estadisticos
        FMM (3), parametros Cosinor (3), picos Cosinor (4),
        estadisticos Cosinor (3) y estadisticos NP (3).
        R: ``tabSamples``.

    per_rep_fmm_fit : list de pd.DataFrame
        Matrices ajustadas FMM (n_top × n_muestras) por repeticion.

    per_rep_cos_fit : list de pd.DataFrame
        Matrices ajustadas Cosinor (n_top × n_muestras) por repeticion.

    per_rep_np_fit : list de pd.DataFrame
        Matrices ajustadas NP (n_top × n_muestras) por repeticion.

    consensus_phase : np.ndarray, forma (n_muestras,)
        Fase circular promedio (media circular) de cada muestra a lo
        largo de las K repeticiones, en [0, 2π).

    top_gene_names : list[str]
        Nombres de genes del TOP en el orden de filas usado.
    """

    sample_orders:    np.ndarray
    circular_scales:  np.ndarray
    direction_flipped: np.ndarray
    stats_table:      np.ndarray
    per_rep_fmm_fit:  list
    per_rep_cos_fit:  list
    per_rep_np_fit:   list
    consensus_phase:  np.ndarray
    top_gene_names:   list

    def summary(self) -> str:
        K = self.sample_orders.shape[0]
        n_top = len(self.top_gene_names)
        flipped = int(self.direction_flipped.sum())
        return "\n".join([
            "=== Resumen de Estimacion Robusta ===",
            f"  Repeticiones (K)         : {K}",
            f"  Genes en TOP             : {n_top}",
            f"  Repeticiones con flip    : {flipped}/{K}",
            f"  Filas en stats_table     : {self.stats_table.shape[0]}",
        ])


# ---------------------------------------------------------------------------
# RobustEstimator
# ---------------------------------------------------------------------------

class RobustEstimator:
    """
    Etapa 4 de CIRCUST: estimacion robusta por consenso de K
    repeticiones aleatorias.

    Parametros
    ----------
    arntl_gene, dbp_gene, cry1_gene : str
        Genes ancla pasados a :class:`PreliminaryOrderEstimator`.

    fmm_length_alpha_grid, fmm_length_omega_grid, fmm_num_reps : int
        Hiperparametros del modelo FMM en cada re-ajuste.

    verbose : bool
        Imprimir progreso.
    """

    def __init__(
        self,
        arntl_gene:           str  = "ARNTL",
        dbp_gene:             str  = "DBP",
        cry1_gene:            str  = "CRY1",
        fmm_length_alpha_grid: int = 48,
        fmm_length_omega_grid: int = 24,
        fmm_num_reps:         int  = 3,
        verbose:              bool = True,
    ) -> None:
        self.arntl_gene = arntl_gene
        self.dbp_gene   = dbp_gene
        self.cry1_gene  = cry1_gene
        self._fmm_kwargs = dict(
            length_alpha_grid=fmm_length_alpha_grid,
            length_omega_grid=fmm_length_omega_grid,
            num_reps=fmm_num_reps,
        )
        self.verbose = verbose

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(
        self,
        random_selection: RandomSelectionResult,
        top_matrix:       pd.DataFrame,
        expr_full_norm:   pd.DataFrame,
        core_genes:       list[str],
    ) -> RobustEstimationResult:
        """
        Ejecuta la estimacion robusta sobre las K repeticiones.

        Parametros
        ----------
        random_selection : RandomSelectionResult
            Salida de :class:`RandomSelector`.

        top_matrix : pd.DataFrame, forma (n_top, n_muestras)
            Matriz TOP de genes (cores + candidatos) en el orden de
            muestras circular preliminar.
            R: ``top``.

        expr_full_norm : pd.DataFrame, forma (n_genes, n_muestras)
            Matriz normalizada completa, en el mismo orden de
            muestras que ``top_matrix``.
            R: ``mFullNorm`` (en orden preOrden preliminar).

        core_genes : list[str]
            Genes reloj centrales (mismo conjunto usado en CPCA / Etapa 2).

        Devuelve
        --------
        RobustEstimationResult
        """
        self._log("=== Etapa 4: Estimacion Robusta ===")

        K     = random_selection.selection_indices.shape[0]
        names_top = list(top_matrix.index)
        n_top = len(names_top)
        n_samp = top_matrix.shape[1]

        fmm   = FMMModel(**self._fmm_kwargs)
        cosm  = CosinorModel()
        prelim = PreliminaryOrderEstimator(
            arntl_gene=self.arntl_gene,
            dbp_gene=self.dbp_gene,
            cry1_gene=self.cry1_gene,
            verbose=False,
        )

        sample_orders   = np.zeros((K, n_samp), dtype=int)
        circ_scales     = np.zeros((K, n_samp), dtype=np.float64)
        flipped_arr     = np.zeros(K, dtype=bool)
        stats_table     = np.zeros((K * n_top, 25), dtype=np.float64)
        per_rep_fmm     = []
        per_rep_cos     = []
        per_rep_np      = []

        for k in range(K):
            self._log(f"  --- Repeticion {k+1}/{K} ---")
            sel_names = [
                g for g in random_selection.selection_names[k].tolist()
                if g in expr_full_norm.index
            ]
            if len(sel_names) < 2:
                raise ValueError(
                    f"Repeticion {k}: menos de 2 genes seleccionados disponibles "
                    "en expr_full_norm para CPCA."
                )

            # ── 1. CPCA sobre la submatriz seleccionada ──────────────────
            sub_df = expr_full_norm.loc[sel_names]
            cpca_k = CPCA(
                core_genes=sel_names, verbose=False
            ).run(sub_df)
            order_k = cpca_k.sample_order
            esc_k   = cpca_k.circular_scale

            # ── 2. Reordenar la matriz TOP por el nuevo orden ─────────────
            top_k = top_matrix.iloc[:, order_k].copy()
            top_k.columns = range(n_samp)

            # ── 3. Ajustar FMM a cada gen core (en top_k) ────────────────
            core_present = [g for g in core_genes if g in top_k.index]
            fmm_fits   = {}
            peak_times = {}
            for g in core_present:
                fr = fmm.fit(top_k.loc[g].values.astype(np.float64), esc_k)
                fmm_fits[g]   = fr
                peak_times[g] = fr.peak_time

            # Refined namespace para reusar PreliminaryOrderEstimator
            fake_refined = SimpleNamespace(
                cpca_final          = SimpleNamespace(circular_scale=esc_k),
                expr_norm_final     = top_k,
                fmm_fits_final      = fmm_fits,
                fmm_peak_times_final = peak_times,
            )

            # basicPreOder + basicOder → orientacion definitiva
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

            # mat_fin esta en columnas reordenadas dentro de top_k.
            # El sample_order GLOBAL es: top_matrix → order_k → o_fin
            global_order = order_k[o_fin]
            sample_orders[k] = global_order
            circ_scales[k]   = esc_fin

            # Pico ARNTL en el marco "raw" (FMM sobre esc_k)
            arntl_idx_core = (
                core_present.index(self.arntl_gene)
                if self.arntl_gene in core_present else None
            )
            arntl_peak_raw = (
                fmm_fits[self.arntl_gene].params["alpha"]
                if self.arntl_gene in fmm_fits else 0.0
            )
            # Reusamos solo arntl_peak para reparametrizar genes no-core.
            # Para alinear con R: usar el FMM-peak (compUU) de ARNTL, no alpha.
            if self.arntl_gene in fmm_fits:
                arntl_peak_raw = fmm_fits[self.arntl_gene].peak_time

            # ── 4. Re-ajustar FMM/Cosinor/NP a cada fila del TOP ─────────
            fmm_mat = np.zeros((n_top, n_samp))
            cos_mat = np.zeros((n_top, n_samp))
            np_mat  = np.zeros((n_top, n_samp))

            # esc rotado a marco ARNTL en el espacio "no sincronizado"
            esc_rot = (esc_k - arntl_peak_raw + pi) % (2.0 * pi)
            order_rot = np.argsort(esc_rot)

            for i, gene in enumerate(names_top):
                vvv = top_k.loc[gene].values.astype(np.float64)

                # ---- FMM (sobre esc_k crudo) ----
                fr = fmm.fit(vvv, esc_k)
                M, A = fr.params["M"], fr.params["A"]
                alpha = fr.params["alpha"]
                beta  = fr.params["beta"]
                omega = fr.params["omega"]
                al = (alpha - arntl_peak_raw + pi) % (2.0 * pi)
                if not orientation_changed:
                    pars_fmm = (M, A, al, beta, omega)
                else:
                    pars_fmm = (
                        M, A,
                        (2.0 * pi - al)   % (2.0 * pi),
                        (2.0 * pi - beta) % (2.0 * pi),
                        omega,
                    )
                pkU = FMMModel.peak_time(pars_fmm[2], pars_fmm[3], pars_fmm[4])
                pkL = (pkU + pi) % (2.0 * pi)
                peaks_fmm = (pkU, pkL, pkU/(2*pi)*100, pkL/(2*pi)*100)
                fitted_fmm = fr.fitted
                resid = vvv - fitted_fmm
                sFMM   = float(np.sum(resid**2) / max(n_samp - 5, 1))
                mseFMM = float(np.sum(resid**2) / n_samp)
                r2FMM  = float(fr.r2)
                if not orientation_changed:
                    fmm_mat[i] = fitted_fmm[order_rot]
                else:
                    fmm_mat[i] = fitted_fmm[order_rot][::-1]

                stat_fmm = list(pars_fmm) + list(peaks_fmm) + [sFMM, mseFMM, r2FMM]

                # ---- Cosinor ----
                cr = cosm.fit(vvv, esc_k)
                Mc, Ac, phiC = cr.params["M"], cr.params["A"], cr.params["phi"]
                phiC_rot = (phiC - arntl_peak_raw + pi) % (2.0 * pi)
                if not orientation_changed:
                    pars_cos = (Mc, Ac, phiC_rot)
                    pk1 = (-phiC_rot) % (2*pi)
                    pk2 = (pi - phiC_rot) % (2*pi)
                else:
                    pars_cos = (Mc, Ac, (2.0*pi - phiC_rot) % (2.0*pi))
                    pk1 = (-phiC) % (2*pi)
                    pk2 = (pi - phiC) % (2*pi)
                peaks_cos = (pk1, pk2, pk1/(2*pi)*100, pk2/(2*pi)*100)
                fitted_cos = cr.fitted
                resid_c = vvv - fitted_cos
                sCos   = float(np.sum(resid_c**2) / max(n_samp - 3, 1))
                mseCos = float(np.sum(resid_c**2) / n_samp)
                r2Cos  = float(cr.r2)
                if not orientation_changed:
                    cos_mat[i] = fitted_cos[order_rot]
                else:
                    cos_mat[i] = fitted_cos[order_rot][::-1]
                stat_cos = list(pars_cos) + list(peaks_cos) + [sCos, mseCos, r2Cos]

                # ---- NP (sobre la senal en el orden anclado) ----
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
                np_mat[i] = np_fit
                resid_n = vvv_anch - np_fit
                sNp   = float(np.sum(resid_n**2) / max(n_samp - 3, 1))
                mseNp = float(np.sum(resid_n**2) / n_samp)
                var_y = float(np.var(vvv_anch))
                r2Np  = float(1.0 - mse_np / var_y) if var_y > 0 else 0.0
                stat_np = [sNp, mseNp, r2Np]

                stats_table[k * n_top + i] = stat_fmm + stat_cos + stat_np

            per_rep_fmm.append(pd.DataFrame(fmm_mat, index=names_top))
            per_rep_cos.append(pd.DataFrame(cos_mat, index=names_top))
            per_rep_np.append(pd.DataFrame(np_mat,  index=names_top))

        # ── Consenso circular: media circular muestra-a-muestra ─────────
        # Para cada muestra original (j en [0, n_samp)), recolectar la
        # fase asignada en cada repeticion y computar la media circular.
        consensus = np.zeros(n_samp, dtype=np.float64)
        for j in range(n_samp):
            angles = []
            for k in range(K):
                pos = int(np.where(sample_orders[k] == j)[0][0])
                angles.append(circ_scales[k, pos])
            angles = np.asarray(angles)
            consensus[j] = np.arctan2(
                np.sin(angles).mean(), np.cos(angles).mean()
            ) % (2.0 * pi)

        result = RobustEstimationResult(
            sample_orders     = sample_orders,
            circular_scales   = circ_scales,
            direction_flipped = flipped_arr,
            stats_table       = stats_table,
            per_rep_fmm_fit   = per_rep_fmm,
            per_rep_cos_fit   = per_rep_cos,
            per_rep_np_fit    = per_rep_np,
            consensus_phase   = consensus,
            top_gene_names    = names_top,
        )
        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Utilidad
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
