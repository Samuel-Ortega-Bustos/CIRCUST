"""
circust/cpca.py
===============
Etapas 1.1 + 1.2: Ordenamiento circular (CPCA) y deteccion de outliers.

Que es CPCA?
------------
PCA sobre los 12 genes core del reloj. Los loadings de PC1 y PC2 de cada
muestra se tratan como coordenadas (x, y) en el plano; el angulo:

    phi = atan2(PC2, PC1)   en [0, 2pi)

da un *ordenamiento circular* que refleja el ritmo circadiano subyacente.
Muestras cercanas al origen tienen norma pequena y son outliers potenciales.

Deteccion de outliers (Seccion 3.3 del suplementario de CIRCUST)
-----------------------------------------------------------------
Dos criterios en logica OR — una muestra se elimina si viola cualquiera:

    1. Radial CPCA:  LEi < tight_radius (0.10)
       Muestras con distancia al origen en espacio PC1-PC2 muy pequena
       disrumpen la estructura circular y reflejan desalineaciones individuales.

    2. Residuos FMM: |ri| > multi_threshold (3.0)  para cualquier gen core
       Detecta errores de medicion u otras anomalias comunes entre genes.

Si se detectan outliers, se eliminan, se renormaliza la matriz core y se
re-ejecuta CPCA para obtener el ordenamiento final limpio.

Posicion en el pipeline
-----------------------
    Preprocessor  ->  CPCA  ->  CircularSynchronizer  ->  ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from circust.core_genes import SEED_GENES_DEFAULT
from circust.fitting.fmm import FMMModel
from circust.fitting.cosinor import CosinorModel
from circust.fitting.rhythm_model import FitResult
from circust.preprocessing import normalise_matrix


# ===========================================================================
# Dataclass de resultado
# ===========================================================================

@dataclass
class CPCAResult:
    """
    Salida de :class:`CPCA` — incluye el ordenamiento circular final y la
    deteccion de outliers (Etapas 1.1 + 1.2 del pipeline).

    Campos del pipeline (usados por etapas posteriores)
    ---------------------------------------------------
    sample_order : np.ndarray de int, forma (n_muestras,)
        Indices que ordenan las muestras por fase circular final.
        Equivalente en R: ``orderCPCA8`` (final, tras eliminar outliers).

    circular_scale : np.ndarray de float, forma (n_muestras,)
        Valores de phi ordenados en [0, 2pi). Eje temporal para ajustes.
        Equivalente en R: ``escalaPhi8`` (final).

    core_genes_found : list[str]
        Genes centrales presentes en la matriz y usados para CPCA.

    variance_explained : np.ndarray de float, forma (3,)
        Fraccion de varianza explicada por PC1, PC2, PC3 (CPCA final).

    expr_norm_final : pd.DataFrame, forma (n_genes, n_muestras_limpias)
        Matriz de expresion completa con muestras outlier eliminadas,
        reordenada por fase circular y renormalizada.
        Equivalente en R: ``mFullTissueNorm`` (linea 4099).

    core_norm_final : pd.DataFrame, forma (n_genes_core, n_muestras_limpias)
        Submatriz core normalizada, sin outliers, ordenada por fase.
        Equivalente en R: ``mTissueCoreGNorm`` (linea 4085).

    fmm_fits_final : dict[str, FitResult]
        Ajustes FMM sobre genes core usando el ordenamiento final.
        Equivalente en R: ``allParAfter`` (lineas 4109-4136).

    fmm_peak_times_final : dict[str, float]
        Tiempos de pico FMM (via compUU) para cada gen core (final).
        Equivalente en R: ``phisFMMAfter`` (linea 4118).

    samples_dropped : list[int]
        Indices originales de columna de las muestras eliminadas.
        Equivalente en R: ``dropTissueOut`` (linea 4083).

    fmm_outliers : list[int]
        Subconjunto de samples_dropped detectados exclusivamente por
        el criterio de residuos FMM (|ri| > multi_threshold).

    Campos de diagnostico (para visualizacion)
    ------------------------------------------
    pc1, pc2, pc3 : np.ndarray
        Loadings de PC1/PC2/PC3 del CPCA final (un valor por muestra).

    std_residuals_fmm : pd.DataFrame o None
        Residuos FMM estandarizados (genes+PCs x muestras, orden inicial).
        Equivalente en R: ``resParStTissue``.

    fmm_fits_initial : dict[str, FitResult]
        Ajustes FMM sobre genes core + PC1/PC2/PC3 en orden CPCA inicial.
        Equivalente en R: ``fitParCore``.

    cosinor_fits_initial : dict[str, FitResult]
        Ajustes Cosinor sobre genes core + PC1/PC2/PC3 en orden inicial.
        Equivalente en R: ``fitCosCore``.
    """

    # ── Pipeline ─────────────────────────────────────────────────────────
    sample_order:          np.ndarray
    circular_scale:        np.ndarray
    core_genes_found:      list[str]
    variance_explained:    np.ndarray
    expr_norm_final:       pd.DataFrame
    core_norm_final:       pd.DataFrame
    fmm_fits_final:        dict
    fmm_peak_times_final:  dict
    samples_dropped:       list[int]   = field(default_factory=list)
    fmm_outliers:          list[int]   = field(default_factory=list)

    # ── Diagnostico ───────────────────────────────────────────────────────
    pc1:                   np.ndarray                = field(default_factory=lambda: np.array([]))
    pc2:                   np.ndarray                = field(default_factory=lambda: np.array([]))
    pc3:                   np.ndarray                = field(default_factory=lambda: np.array([]))
    std_residuals_fmm:     Optional[pd.DataFrame]    = None
    fmm_fits_initial:      dict                      = field(default_factory=dict)
    cosinor_fits_initial:  dict                      = field(default_factory=dict)

    def summary(self) -> str:
        n_cpca = len(self.samples_dropped) - len(self.fmm_outliers)
        lines = [
            "=== Resumen CPCA + Deteccion de Outliers ===",
            f"  Genes core usados           : {len(self.core_genes_found)}  {self.core_genes_found}",
            f"  Varianza PC1 / PC2 / PC3    : "
            f"{self.variance_explained[0]:.1%} / "
            f"{self.variance_explained[1]:.1%} / "
            f"{self.variance_explained[2]:.1%}",
            f"  Muestras eliminadas         : {len(self.samples_dropped)}",
            f"    Criterio CPCA radial      : {n_cpca}",
            f"    Criterio FMM residuos     : {len(self.fmm_outliers)}",
            f"  Muestras finales            : {self.expr_norm_final.shape[1]}",
        ]
        return "\n".join(lines)


# ===========================================================================
# Clase CPCA
# ===========================================================================

class CPCA:
    """
    Etapas 1.1 y 1.2: ordenamiento circular de muestras y deteccion de
    outliers residuales, segun la Seccion 3.3 del suplementario de CIRCUST.

    Parametros — CPCA
    -----------------
    core_genes : list[str], opcional
        Genes ancla circadianos. Por defecto los 12 del articulo CIRCUST.

    n_outlier_candidates : int
        Muestras a examinar como candidatas a outlier (menor norma PC1-PC2).
        Por defecto: 8.

    tight_radius : float
        Umbral radial primario. Muestras con LEi <= tight_radius se confirman
        como outliers CPCA. Por defecto: 0.10.

    loose_radius : float
        Umbral de respaldo cuando ninguna muestra califica con tight_radius.
        Por defecto: 0.15.

    Parametros — deteccion de outliers FMM
    ---------------------------------------
    multi_threshold : float
        Umbral de residuo FMM estandarizado. Muestras con |ri| > multi_threshold
        en cualquier gen core se declaran outliers. Por defecto: 3.0.

    max_outlier_fraction : float
        Limite maximo de muestras eliminables: ceil(fraccion x n_muestras).
        Por defecto: 0.05.

    Parametros — ajuste FMM
    ------------------------
    fmm_length_alpha_grid : int
        Resolucion de rejilla para el parametro alfa de FMM. Por defecto: 48.

    fmm_length_omega_grid : int
        Resolucion de rejilla para el parametro omega de FMM. Por defecto: 24.

    fmm_num_reps : int
        Iteraciones de refinamiento FMM. Por defecto: 3.

    verbose : bool
        Imprimir mensajes de progreso.

    Ejemplo
    -------
    >>> from circust.preprocessing import load_expression_matrix, Preprocessor
    >>> from circust.cpca import CPCA
    >>>
    >>> matrix = load_expression_matrix("data/raw/expression.csv")
    >>> prep   = Preprocessor().run(matrix)
    >>> result = CPCA().run(prep.expr_norm)
    >>> print(result.summary())
    """

    def __init__(
        self,
        core_genes:               list[str] = None,
        # CPCA
        n_outlier_candidates:     int       = 8,
        tight_radius:             float     = 0.10,
        loose_radius:             float     = 0.15,
        # Outlier detection
        multi_threshold:          float     = 3.0,
        max_outlier_fraction:     float     = 0.05,
        # FMM fitting
        fmm_length_alpha_grid:    int       = 48,
        fmm_length_omega_grid:    int       = 24,
        fmm_num_reps:             int       = 3,
        verbose:                  bool      = True,
    ) -> None:
        self.core_genes               = core_genes if core_genes is not None else SEED_GENES_DEFAULT
        self.n_outlier_candidates     = n_outlier_candidates
        self.tight_radius             = tight_radius
        self.loose_radius             = loose_radius
        self.multi_threshold          = multi_threshold
        self.max_outlier_fraction     = max_outlier_fraction
        self.fmm_length_alpha_grid    = fmm_length_alpha_grid
        self.fmm_length_omega_grid    = fmm_length_omega_grid
        self.fmm_num_reps             = fmm_num_reps
        self.verbose                  = verbose

    # -----------------------------------------------------------------------
    # API publica
    # -----------------------------------------------------------------------

    def run(self, expr_norm: pd.DataFrame) -> CPCAResult:
        """
        Ejecuta CPCA + deteccion de outliers sobre la matriz completa.

        Parametros
        ----------
        expr_norm : pd.DataFrame, forma (n_genes, n_muestras)
            Matriz de expresion normalizada completa (valores en [-1, 1]).
            Es ``PreprocessingResult.expr_norm``.

        Devuelve
        --------
        CPCAResult
        """
        self._log("=== Etapa 1.1: CPCA ===")

        # ── Paso 1: extraer submatriz de genes core ───────────────────────
        core_matrix, genes_found = self._extract_core_genes(expr_norm)

        # ── Paso 2: CPCA inicial ──────────────────────────────────────────
        (sample_order, circular_scale,
         pc1, pc2, pc3, var_exp,
         outlier_idx) = self._cpca_step(core_matrix)

        self._log(
            f"  Varianza PC1/PC2/PC3: {var_exp[0]:.1%} / "
            f"{var_exp[1]:.1%} / {var_exp[2]:.1%}"
        )
        self._log(f"  Outliers radiales CPCA (LEi < {self.tight_radius}): {len(outlier_idx)}")

        # ── Paso 3: ajuste FMM + Cosinor en orden inicial ────────────────
        self._log("=== Etapa 1.2: Deteccion de Outliers ===")
        self._log("  Ajustando FMM y Cosinor en genes core (orden inicial) ...")
        fmm_fits_ini, cos_fits_ini, std_res_df = self._fit_initial(
            core_matrix, genes_found, sample_order, circular_scale,
            pc1, pc2, pc3,
        )

        # ── Paso 4: detectar outliers (CPCA OR FMM residuos) ─────────────
        cpca_outs, fmm_outs, all_dropped = self._detect_outliers(
            outlier_idx, sample_order, std_res_df, genes_found,
        )
        self._log(f"  Criterio CPCA radial     : {len(cpca_outs)} muestra(s)")
        self._log(f"  Criterio FMM residuos    : {len(fmm_outs)} muestra(s)")
        self._log(f"  Total eliminados         : {len(all_dropped)}")

        # ── Paso 5: limpiar + re-ejecutar CPCA si hay outliers ───────────
        if all_dropped:
            self._log("  Re-ejecutando CPCA sobre datos limpios ...")
            core_clean = self._drop_samples(core_matrix, all_dropped)
            core_norm_clean = normalise_matrix(core_clean)
            (sample_order, circular_scale,
             pc1, pc2, pc3, var_exp, _) = self._cpca_step(core_norm_clean)
        else:
            self._log("  Sin outliers — manteniendo CPCA inicial.")
            core_norm_clean = core_matrix

        # ── Paso 6: ordenar y renormalizar la matriz completa ─────────────
        self._log("  Ordenando y renormalizando la matriz completa ...")
        expr_norm_final = self._order_full_matrix(expr_norm, all_dropped, sample_order)

        # ── Paso 7: ajuste FMM final sobre genes core ─────────────────────
        self._log("  Ajustando FMM en genes core (orden final) ...")
        fmm_fits_fin, fmm_peaks_fin = self._fit_final(
            expr_norm_final, genes_found, circular_scale,
        )

        result = CPCAResult(
            sample_order         = sample_order,
            circular_scale       = circular_scale,
            core_genes_found     = genes_found,
            variance_explained   = var_exp,
            expr_norm_final      = expr_norm_final,
            core_norm_final      = core_norm_clean.iloc[:, sample_order],
            fmm_fits_final       = fmm_fits_fin,
            fmm_peak_times_final = fmm_peaks_fin,
            samples_dropped      = list(all_dropped),
            fmm_outliers         = fmm_outs,
            # diagnostico
            pc1                  = pc1,
            pc2                  = pc2,
            pc3                  = pc3,
            std_residuals_fmm    = std_res_df,
            fmm_fits_initial     = fmm_fits_ini,
            cosinor_fits_initial = cos_fits_ini,
        )

        self._log(result.summary())
        return result

    # -----------------------------------------------------------------------
    # Pasos privados
    # -----------------------------------------------------------------------

    def _extract_core_genes(
        self, expr_norm: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Selecciona las filas de genes core de la matriz normalizada."""
        genes_found, missing = [], []
        for gene in self.core_genes:
            (genes_found if gene in expr_norm.index else missing).append(gene)

        if missing:
            self._log(
                f"  AVISO: {len(missing)} gen(es) core no encontrado(s): {missing}"
            )
        if len(genes_found) < 2:
            raise ValueError(
                f"CPCA requiere al menos 2 genes core. "
                f"Solo se encontraron: {genes_found}."
            )

        self._log(
            f"  Genes core extraidos: {len(genes_found)} / {len(self.core_genes)}"
        )
        return expr_norm.loc[genes_found], genes_found

    def _cpca_step(
        self, core_matrix: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """
        Ejecuta un paso de CPCA sobre la matriz core normalizada.

        Devuelve
        --------
        (sample_order, circular_scale, pc1, pc2, pc3, var_exp, outlier_idx)
        """
        values = core_matrix.values.astype(float)
        n_genes, n_samples = values.shape

        # Centrar por filas, escalar por columnas (R: prcomp scale.=TRUE center=FALSE)
        centred  = values - values.mean(axis=1, keepdims=True)
        col_rms  = np.sqrt(np.sum(centred**2, axis=0) / (n_genes - 1))
        col_rms[col_rms == 0] = 1.0
        scaled   = centred / col_rms

        n_comp = min(3, n_genes, n_samples)
        _, sigma_all, Vt = np.linalg.svd(scaled, full_matrices=False)

        pc1 = Vt[0]
        pc2 = Vt[1]
        pc3 = Vt[2] if n_comp >= 3 else np.zeros(n_samples)

        var_exp = np.zeros(3)
        var_exp[:n_comp] = (sigma_all[:n_comp]**2) / np.sum(sigma_all**2)

        norm12    = np.sqrt(pc1**2 + pc2**2)
        safe_norm = np.where(norm12 == 0, 1.0, norm12)
        phi       = np.arctan2(pc2 / safe_norm, pc1 / safe_norm) % (2 * np.pi)

        sample_order   = np.argsort(phi)
        circular_scale = phi[sample_order]

        # Outliers radiales: n_outlier_candidates menores normas
        n_cands      = min(self.n_outlier_candidates, n_samples)
        cand_idx     = np.argsort(norm12)[:n_cands]
        tight_mask   = norm12[cand_idx] <= self.tight_radius

        if tight_mask.any():
            outlier_idx = cand_idx[tight_mask]
        else:
            loose_mask  = norm12[cand_idx] <= self.loose_radius
            outlier_idx = cand_idx[loose_mask]

        return sample_order, circular_scale, pc1, pc2, pc3, var_exp, outlier_idx

    def _fit_initial(
        self,
        core_matrix:   pd.DataFrame,
        genes_found:   list[str],
        sample_order:  np.ndarray,
        circular_scale: np.ndarray,
        pc1: np.ndarray,
        pc2: np.ndarray,
        pc3: np.ndarray,
    ) -> tuple[dict, dict, pd.DataFrame]:
        """
        Ajusta FMM y Cosinor en genes core + eigengenes (PC1/PC2/PC3)
        usando el ordenamiento CPCA inicial.

        Devuelve
        --------
        fmm_fits   : {nombre: FitResult}
        cos_fits   : {nombre: FitResult}
        std_res_df : DataFrame (senales x muestras) de residuos estandarizados
        """
        cm_vals = core_matrix.values

        signals: dict[str, np.ndarray] = {}
        for i, gene in enumerate(genes_found):
            signals[gene] = cm_vals[i, sample_order]
        signals["PC1"] = pc1[sample_order]
        signals["PC2"] = pc2[sample_order]
        signals["PC3"] = pc3[sample_order]

        fmm_model = FMMModel(
            length_alpha_grid = self.fmm_length_alpha_grid,
            length_omega_grid = self.fmm_length_omega_grid,
            num_reps          = self.fmm_num_reps,
        )
        cos_model = CosinorModel()

        fmm_fits:      dict[str, FitResult] = {}
        cos_fits:      dict[str, FitResult] = {}
        std_res_rows:  list[np.ndarray]     = []
        signal_names:  list[str]            = []

        n_total = len(signals)
        for idx, (name, data) in enumerate(signals.items(), 1):
            self._log(f"    [{idx}/{n_total}] {name}", end="\r")
            fr = fmm_model.fit(data, circular_scale)
            cr = cos_model.fit(data, circular_scale)
            fmm_fits[name] = fr
            cos_fits[name] = cr
            std_res_rows.append(fr.residuals_std)
            signal_names.append(name)

        self._log(f"    Ajuste completado: {n_total} senales.           ")

        std_res_df = pd.DataFrame(
            np.vstack(std_res_rows),
            index   = signal_names,
            columns = np.arange(len(circular_scale)),
        )
        return fmm_fits, cos_fits, std_res_df

    def _detect_outliers(
        self,
        outlier_idx:   np.ndarray,
        sample_order:  np.ndarray,
        std_res_df:    pd.DataFrame,
        genes_found:   list[str],
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Aplica criterios de outlier en logica OR (Sec. 3.3 suplementario):

          1. CPCA radial (LEi < tight_radius): ya en outlier_idx.
          2. FMM residuos (|ri| > multi_threshold): para cualquier gen core.

        Devuelve
        --------
        cpca_outs  : eliminados por criterio radial
        fmm_outs   : eliminados solo por criterio FMM
        all_dropped: union ordenada (con tope max_outlier_fraction)
        """
        n_samples = len(sample_order)
        cap       = int(np.ceil(self.max_outlier_fraction * n_samples))

        # Criterio 1: CPCA radial
        cpca_set = set(outlier_idx.tolist())

        # Criterio 2: residuos FMM sobre genes core (no eigengenes)
        core_res = std_res_df.loc[genes_found].values  # (n_genes, n_samples)
        fmm_cands: list[int] = []
        for i in range(len(genes_found)):
            for pos in np.where(np.abs(core_res[i]) > self.multi_threshold)[0]:
                fmm_cands.append(int(sample_order[pos]))

        # Union OR con tope
        seen:       set[int]  = set()
        all_dropped: list[int] = []
        for idx in list(outlier_idx) + fmm_cands:
            if idx not in seen:
                seen.add(idx)
                all_dropped.append(idx)
            if len(all_dropped) >= cap:
                break

        cpca_outs = [i for i in all_dropped if i in cpca_set]
        fmm_outs  = [i for i in all_dropped if i not in cpca_set]

        return cpca_outs, fmm_outs, all_dropped

    def _drop_samples(
        self, matrix: pd.DataFrame, dropped: list[int],
    ) -> pd.DataFrame:
        """Elimina columnas por indice original de muestra."""
        all_cols  = np.arange(matrix.shape[1])
        keep_mask = ~np.isin(all_cols, dropped)
        return matrix.iloc[:, keep_mask]

    def _order_full_matrix(
        self,
        expr_norm:    pd.DataFrame,
        dropped:      list[int],
        sample_order: np.ndarray,
    ) -> pd.DataFrame:
        """
        Elimina outliers de la matriz completa, reordena por fase circular
        final y renormaliza gen a gen.
        """
        mat_clean   = self._drop_samples(expr_norm, dropped)
        mat_ordered = mat_clean.iloc[:, sample_order]
        return normalise_matrix(mat_ordered)

    def _fit_final(
        self,
        expr_norm_final: pd.DataFrame,
        genes_found:     list[str],
        circular_scale:  np.ndarray,
    ) -> tuple[dict, dict]:
        """
        Ajusta FMM en cada gen core con el ordenamiento final.

        Devuelve
        --------
        fmm_fits_final : {gen: FitResult}
        peak_times     : {gen: float}
        """
        fmm_model = FMMModel(
            length_alpha_grid = self.fmm_length_alpha_grid,
            length_omega_grid = self.fmm_length_omega_grid,
            num_reps          = self.fmm_num_reps,
        )

        fmm_fits: dict[str, FitResult] = {}
        peaks:    dict[str, float]     = {}

        n_total = len(genes_found)
        for idx, gene in enumerate(genes_found, 1):
            self._log(f"    [{idx}/{n_total}] {gene}", end="\r")
            if gene not in expr_norm_final.index:
                self._log(f"    AVISO: {gene} no en la matriz final, omitido.")
                continue
            data = expr_norm_final.loc[gene].values
            fr   = fmm_model.fit(data, circular_scale)
            fmm_fits[gene] = fr
            peaks[gene]    = fr.peak_time

        self._log(f"    Ajuste final completado: {len(fmm_fits)} genes core.  ")
        return fmm_fits, peaks

    # -----------------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------------

    def _log(self, message: str, end: str = "\n") -> None:
        if self.verbose:
            print(message, end=end, flush=True)
