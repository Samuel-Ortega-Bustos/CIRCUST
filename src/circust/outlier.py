"""
circust/outlier.py
==============================
Deteccion de outliers residuales y re-ejecucion de CPCA.

Este modulo implementa la segunda mitad de ``giveMatIniNP_v3_cores`` de R,
partiendo del resultado CPCA inicial y procediendo a traves de:

  1. Ajustar Cosinor + FMM en cada gen central y autogen usando el
     ordenamiento circular inicial.
  2. Calcular residuos FMM estandarizados para cada gen × muestra.
  3. Marcar muestras como outliers residuales bajo dos criterios:
       - Multivariante: |res_std| > 3 Y ya es candidato a outlier CPCA
       - Univariante:   |res_std| > 4 para cualquier gen individual
     Ambos criterios estan limitados a ceil(5% × n_muestras) total.
  4. Si se encuentran outliers: eliminarlos, renormalizar la matriz core,
     re-ejecutar CPCA para obtener el ordenamiento final limpio.
  5. Aplicar el ordenamiento final a la matriz de expresion completa y
     renormalizarla — produciendo la matriz usada para la puntuacion
     genomica completa.

Equivalente en R
-----------------
Lineas 3962–4101 de ``giveMatIniNP_v3_cores`` en ``functionGTEX_cores.R``.

Posicion en el pipeline
-----------------------
    Preprocessor  →  CPCA  →  OutlierRefiner  →  RhythmicityScorer
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from circust.cpca import CPCA, CPCAResult
from circust.fitting.cosinor import CosinorModel
from circust.fitting.fmm import FMMModel, fmm_peak_time
from circust.fitting.rhythm_model import FitResult
import circust.constants as const


# ---------------------------------------------------------------------------
# Auxiliar — normalizar cada fila a [−1, 1]  (vectorizado)
# ---------------------------------------------------------------------------

def _normalise_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """Normaliza min-max cada fila a [−1, 1]. Filas constantes → ceros."""
    values  = mat.values.astype(float)
    row_min = values.min(axis=1, keepdims=True)
    row_max = values.max(axis=1, keepdims=True)
    span    = row_max - row_min
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.where(span == 0, 0.0, 2.0 * (values - row_min) / span - 1.0)
    return pd.DataFrame(normed, index=mat.index, columns=mat.columns)


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class OutlierRefinementResult:
    """
    Todas las salidas producidas por :class:`OutlierRefiner`.

    Atributos
    ---------
    cpca_final : CPCAResult
        Resultado CPCA tras eliminar outliers residuales. Si no se
        encontraron outliers es identico al resultado CPCA de entrada.

    expr_norm_final : pd.DataFrame, forma (n_genes, n_muestras_limpias)
        Matriz de expresion normalizada completa con muestras outlier
        eliminadas y filas reordenadas por la fase circular final.
        Equivalente en R: ``mFullTissueNorm`` (linea 4099).

    core_norm_final : pd.DataFrame, forma (n_genes_core, n_muestras_limpias)
        Matriz normalizada de genes centrales tras eliminacion de outliers,
        tambien ordenada.
        Equivalente en R: ``mTissueCoreGNorm`` tras la eliminacion (linea 4085).

    samples_dropped : list[int]
        Indices de columna (en el espacio de muestras original) de las
        muestras eliminadas como outliers residuales.
        Equivalente en R: ``dropTissueOut`` (linea 4083).

    univariate_outliers : list[int]
        Indices de muestras marcadas por el criterio univariante (|res_std| > 4).
        Equivalente en R: ``outsUni`` (linea 4071).

    multivariate_outliers : list[int]
        Indices de muestras marcadas por el criterio multivariante (|res_std| > 3
        Y ya candidato CPCA).
        Equivalente en R: ``outsMult`` (linea 4062).

    fmm_fits_initial : dict[str, FitResult]
        Ajustes FMM sobre genes centrales + autogenes usando el ordenamiento
        CPCA inicial. Claves: simbolos de genes + "PC1", "PC2", "PC3".
        Equivalente en R: ``fitParCore`` filas / ``FMMParCoreG`` (lineas 3999/4007).

    cosinor_fits_initial : dict[str, FitResult]
        Ajustes Cosinor sobre genes centrales + autogenes usando el
        ordenamiento CPCA inicial.
        Equivalente en R: ``fitCosCore`` filas / ``CosParCoreG`` (linea 4008).

    std_residuals_fmm : pd.DataFrame, forma (n_senales, n_muestras)
        Residuos FMM estandarizados usados para deteccion de outliers.
        Filas = genes centrales + PC1/PC2/PC3. Columnas = muestras en orden CPCA.
        Equivalente en R: ``resParStTissue`` (linea 4003).

    fmm_peak_times_initial : dict[str, float]
        Tiempo de pico FMM (t_U) para cada senal tras el ajuste inicial.
        Equivalente en R: ``peaksCoreG`` (linea 4009).

    cosinor_peak_times_initial : dict[str, float]
        Acrofase Cosinor para cada senal tras el ajuste inicial.
        Equivalente en R: ``phisCoreG`` (linea 4006).

    outliers_were_found : bool
        True si se detectaron y eliminaron outliers residuales.

    fmm_fits_final : dict[str, FitResult]
        Ajustes FMM sobre genes centrales usando el ordenamiento CPCA
        **final** (tras eliminacion de outliers y renormalizacion).
        Claves = simbolos de genes centrales.
        Equivalente en R: ``allParAfter`` (lineas 4109-4136).

    fmm_peak_times_final : dict[str, float]
        Tiempo de pico FMM (t_U via compUU) para cada gen central del
        ajuste con ordenamiento final.
        Equivalente en R: ``phisFMMAfter`` (linea 4118).
    """

    cpca_final:               CPCAResult
    expr_norm_final:          pd.DataFrame
    core_norm_final:          pd.DataFrame
    samples_dropped:          list[int]       = field(default_factory=list)
    univariate_outliers:      list[int]       = field(default_factory=list)
    multivariate_outliers:    list[int]       = field(default_factory=list)
    fmm_fits_initial:         dict            = field(default_factory=dict)
    cosinor_fits_initial:     dict            = field(default_factory=dict)
    std_residuals_fmm:        Optional[pd.DataFrame] = None
    fmm_peak_times_initial:   dict            = field(default_factory=dict)
    cosinor_peak_times_initial: dict          = field(default_factory=dict)
    outliers_were_found:      bool            = False
    fmm_fits_final:           dict            = field(default_factory=dict)
    fmm_peak_times_final:     dict            = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Resumen de Refinamiento de Outliers ===",
            f"  Outliers residuales encontrados : {len(self.samples_dropped)}",
            f"    Univariante  (|res|>4)        : {len(self.univariate_outliers)}",
            f"    Multivariante(|res|>3)        : {len(self.multivariate_outliers)}",
            f"  Re-ejecucion CPCA               : {'si' if self.outliers_were_found else 'no'}",
            f"  n_muestras final                 : {self.expr_norm_final.shape[1]}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clase OutlierRefiner
# ---------------------------------------------------------------------------

class OutlierRefiner:
    """
    Detecta muestras outlier residuales usando ajustes FMM sobre los genes
    reloj centrales, y luego re-ejecuta CPCA sobre el dataset limpio.

    El algoritmo ajusta tanto Cosinor como FMM a cada gen central y
    autogen usando el ordenamiento circular CPCA inicial. Las muestras
    cuyos residuos FMM estandarizados superan los umbrales se marcan como
    outliers, se eliminan, y se re-ejecuta CPCA para producir el
    ordenamiento final limpio.

    Parametros
    ----------
    multi_threshold : float
        Umbral de residuo FMM estandarizado para el criterio multivariante.
        Una muestra es candidata multivariante si |res_std| > este valor
        para cualquier gen central Y la muestra ya era candidata radial CPCA.
        Por defecto en R: 3.0 (linea 4057).

    uni_threshold : float
        Umbral de residuo FMM estandarizado para el criterio univariante.
        Una muestra es outlier univariante si |res_std| > este valor para
        cualquier gen individual.
        Por defecto en R: 4.0 (linea 4069).

    max_outlier_fraction : float
        Fraccion maxima de muestras que pueden eliminarse como outliers
        residuales. El limite se aplica como ``ceil(fraccion × n_muestras)``.
        Por defecto en R: 0.05 (lineas 4057, 4069).

    fmm_length_alpha_grid : int
        Resolucion de rejilla para el parametro α de FMM. Por defecto en R: 48.

    fmm_length_omega_grid : int
        Resolucion de rejilla para el parametro ω de FMM. Por defecto en R: 24.

    fmm_num_reps : int
        Numero de iteraciones de refinamiento FMM. Por defecto en R: 3.

    cpca_n_outlier_candidates : int
        Pasado a la re-ejecucion de CPCA si se encuentran outliers.
        Por defecto en R: 8.

    cpca_tight_radius : float
        Pasado a la re-ejecucion de CPCA. Por defecto en R: 0.10.

    cpca_loose_radius : float
        Pasado a la re-ejecucion de CPCA. Por defecto en R: 0.15.

    verbose : bool
        Imprimir mensajes de progreso.

    Ejemplos
    --------
    >>> from circust.preprocessing import Preprocessor, load_expression_matrix
    >>> from circust.cpca import CPCA
    >>> from circust.outlier_refinement import OutlierRefiner
    >>>
    >>> matrix  = load_expression_matrix("data/BA46.csv")
    >>> prep    = Preprocessor().run(matrix)
    >>> cpca    = CPCA().run(prep.expr_norm)
    >>> refined = OutlierRefiner().run(cpca, prep.expr_norm)
    >>> print(refined.summary())
    """

    def __init__(
        self,
        multi_threshold:           float = const.OUTLIER_RESIDUAL_THRESHOLD,
        uni_threshold:             float = 4.0,
        max_outlier_fraction:      float = 0.05,
        fmm_length_alpha_grid:     int   = 48,
        fmm_length_omega_grid:     int   = 24,
        fmm_num_reps:              int   = 3,
        cpca_n_outlier_candidates: int   = const.N_OUTLIER_CANDIDATES,
        cpca_tight_radius:         float = const.OUTLIER_RADIAL_THRESHOLD,
        cpca_loose_radius:         float = const.OUTLIER_RADIAL_THRESHOLD_LOOSE,
        verbose:                   bool  = True,
    ) -> None:
        self.multi_threshold           = multi_threshold
        self.uni_threshold             = uni_threshold
        self.max_outlier_fraction      = max_outlier_fraction
        self.fmm_length_alpha_grid     = fmm_length_alpha_grid
        self.fmm_length_omega_grid     = fmm_length_omega_grid
        self.fmm_num_reps              = fmm_num_reps
        self.cpca_n_outlier_candidates = cpca_n_outlier_candidates
        self.cpca_tight_radius         = cpca_tight_radius
        self.cpca_loose_radius         = cpca_loose_radius
        self.verbose                   = verbose

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(
        self,
        cpca:      CPCAResult,
        expr_norm: pd.DataFrame,
    ) -> OutlierRefinementResult:
        """
        Ejecuta el procedimiento completo de refinamiento de outliers.

        Parametros
        ----------
        cpca : CPCAResult
            Salida de ``CPCA.run()``. Debe haberse producido con
            ``store_core_matrix=True`` (el valor por defecto).
        expr_norm : pd.DataFrame
            Matriz de expresion normalizada completa (genes × muestras) —
            es decir, ``PreprocessingResult.expr_norm``. Se usa para producir
            la matriz final ordenada genomica completa.

        Devuelve
        --------
        OutlierRefinementResult
        """
        if cpca.core_matrix is None:
            raise ValueError(
                "cpca.core_matrix es None. "
                "Ejecuta CPCA con store_core_matrix=True (el valor por defecto)."
            )

        self._log("=== Refinamiento de Outliers ===")

        # ── Paso 1: ajustar Cosinor + FMM en genes centrales + autogenes ─
        self._log("  Paso 1 — ajustando modelos en genes centrales y autogenes ...")
        fmm_fits, cos_fits, std_res_df, peak_fmm, peak_cos = self._fit_initial(cpca)

        # ── Paso 2: detectar outliers residuales ─────────────────────────
        self._log("  Paso 2 — detectando outliers residuales ...")
        uni_outs, mult_outs = self._detect_outliers(cpca, std_res_df)

        dropped = sorted(set(uni_outs) | set(mult_outs))
        self._log(
            f"    Univariante  (|res|>{self.uni_threshold}): {len(uni_outs)} muestra(s)"
        )
        self._log(
            f"    Multivariante(|res|>{self.multi_threshold}): {len(mult_outs)} muestra(s)"
        )
        self._log(f"    Total eliminados: {len(dropped)}")

        # ── Paso 3: limpiar + re-ejecutar CPCA si es necesario ──────────
        if dropped:
            self._log("  Paso 3 — re-ejecutando CPCA sobre datos limpios ...")
            cpca_final, core_norm_clean = self._rerun_cpca(cpca, dropped)
        else:
            self._log("  Paso 3 — no se encontraron outliers, manteniendo resultado CPCA inicial")
            cpca_final     = cpca
            core_norm_clean = cpca.core_matrix

        # ── Paso 4: aplicar ordenamiento final a la matriz completa ──────
        self._log("  Paso 4 — ordenando la matriz de expresion completa ...")
        expr_norm_final = self._order_full_matrix(
            expr_norm, dropped, cpca_final.sample_order
        )

        # ── Paso 5: ajustar FMM en genes centrales con ordenamiento final ─
        # Equivalente en R: lineas 4109-4136 (allParAfter, phisFMMAfter)
        self._log("  Paso 5 — ajustando FMM en genes centrales (ordenamiento final) ...")
        fmm_fits_fin, peak_fmm_fin = self._fit_final(expr_norm_final, cpca_final)

        self._log("  Hecho.")

        result = OutlierRefinementResult(
            cpca_final               = cpca_final,
            expr_norm_final          = expr_norm_final,
            core_norm_final          = core_norm_clean,
            samples_dropped          = dropped,
            univariate_outliers      = uni_outs,
            multivariate_outliers    = mult_outs,
            fmm_fits_initial         = fmm_fits,
            cosinor_fits_initial     = cos_fits,
            std_residuals_fmm        = std_res_df,
            fmm_peak_times_initial   = peak_fmm,
            cosinor_peak_times_initial = peak_cos,
            outliers_were_found      = len(dropped) > 0,
            fmm_fits_final           = fmm_fits_fin,
            fmm_peak_times_final     = peak_fmm_fin,
        )

        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Pasos privados
    # ------------------------------------------------------------------

    def _fit_initial(
        self,
        cpca: CPCAResult,
    ) -> tuple[dict, dict, pd.DataFrame, dict, dict]:
        """
        Ajusta Cosinor y FMM en cada gen central + PC1/PC2/PC3.

        Devuelve
        --------
        fmm_fits   : {nombre_senal: FitResult}
        cos_fits   : {nombre_senal: FitResult}
        std_res_df : DataFrame (n_senales × n_muestras) de residuos FMM estandarizados
        peak_fmm   : {nombre_senal: float}  — tiempos de pico FMM
        peak_cos   : {nombre_senal: float}  — acrofases Cosinor
        """
        order       = cpca.sample_order         # sample ordering indices
        time_points = cpca.circular_scale       # escalaPhi8  (sorted phi)
        genes       = cpca.core_genes_found

        # Construir el dict de senales ordenadas (R: datCore)
        # Genes centrales: filas de core_matrix reordenadas por sample_order
        cm = cpca.core_matrix
        cm_vals = cm.values if hasattr(cm, "values") else cm

        signals: dict[str, np.ndarray] = {}
        for i, gene in enumerate(genes):
            signals[gene] = cm_vals[i, order]

        # Autogenes ordenados de la misma forma (R lineas 3978-3980)
        signals["PC1"] = cpca.pc1[order]
        signals["PC2"] = cpca.pc2[order]
        signals["PC3"] = cpca.pc3[order]

        fmm_model = FMMModel(
            length_alpha_grid = self.fmm_length_alpha_grid,
            length_omega_grid = self.fmm_length_omega_grid,
            num_reps          = self.fmm_num_reps,
        )
        cos_model = CosinorModel()

        fmm_fits:  dict[str, FitResult] = {}
        cos_fits:  dict[str, FitResult] = {}
        peak_fmm:  dict[str, float]     = {}
        peak_cos:  dict[str, float]     = {}
        std_res_rows: list[np.ndarray]  = []
        signal_names: list[str]         = []

        n_total = len(signals)
        for idx, (name, data) in enumerate(signals.items(), 1):
            self._log(f"    [{idx}/{n_total}] {name}", end="\r")

            fr = fmm_model.fit(data, time_points)
            cr = cos_model.fit(data, time_points)

            fmm_fits[name] = fr
            cos_fits[name] = cr

            # residuos FMM estandarizados  (R linea 4003: resParStTissue)
            std_res_rows.append(fr.residuals_std)
            signal_names.append(name)

            # tiempo de pico FMM via compUU  (R linea 4009)
            peak_fmm[name] = fmm_peak_time(
                fr.params["alpha"], fr.params["beta"], fr.params["omega"]
            )

            # acrofase Cosinor  (R linea 4006: (-funCos[[5]]) %% (2*pi))
            peak_cos[name] = (-cr.params["phi"]) % (2.0 * np.pi)

        self._log(f"    Ajuste completado de {n_total} senales.           ")

        # Construir DataFrame: filas = senales, columnas = posiciones de muestra 0..n-1
        std_res_df = pd.DataFrame(
            np.vstack(std_res_rows),
            index   = signal_names,
            columns = np.arange(len(time_points)),
        )

        return fmm_fits, cos_fits, std_res_df, peak_fmm, peak_cos

    def _detect_outliers(
        self,
        cpca:       CPCAResult,
        std_res_df: pd.DataFrame,
    ) -> tuple[list[int], list[int]]:
        """
        Aplica los dos criterios de outlier de R a los residuos FMM estandarizados.

        Criterio multivariante (R lineas 4057-4067)
        --------------------------------------------
        Para cada gen central, encontrar posiciones en el array de residuos
        ordenado donde |res_std| > multi_threshold. Una muestra en esa
        posicion es candidata multivariante solo si su indice de muestra
        ORIGINAL tambien esta en la lista de candidatos a outlier CPCA
        (initialTissue[[3]]).
        Limitado a ceil(max_outlier_fraction × n_muestras) total.

        Criterio univariante (R lineas 4069-4074)
        ------------------------------------------
        Para cada gen central, cualquier posicion donde |res_std| > uni_threshold
        es un outlier univariante. Se aplica el mismo limite total.

        Nota: ambos criterios trabajan solo sobre GENES CENTRALES, no autogenes.
        El DataFrame de residuos tiene autogenes como filas extra — aqui
        seleccionamos solo las filas de genes centrales, coincidiendo con el
        bucle de R sobre ``1:length(coreG)`` (lineas 4056, 4069).

        Devuelve
        --------
        uni_outs  : lista de indices de muestra originales (base 0)
        mult_outs : lista de indices de muestra originales (base 0)
        """
        genes       = cpca.core_genes_found
        order       = cpca.sample_order          # position → original index
        # R usa solo los outliers CONFIRMADOS (dentro del umbral radial),
        # no los 8 candidatos.  R: match(..., mOutliers) <= ss8.
        confirmed   = set(cpca.outlier_idx.tolist())
        n_samples   = len(order)
        cap         = int(np.ceil(self.max_outlier_fraction * n_samples))

        # Solo filas de genes centrales (no autogenes)
        core_std_res = std_res_df.loc[genes].values    # (n_genes, n_samples_ordered)

        mult_outs: list[int] = []
        uni_outs:  list[int] = []
        n_flagged = 0

        for i, gene in enumerate(genes):
            gene_res = core_std_res[i]    # (n_samples_ordered,)

            # ── Multivariante: |res| > multi_threshold Y outlier CPCA confirmado
            # R lineas 4057-4067: incrementa nOutsTot por MUESTRA.
            if np.any(np.abs(gene_res) > self.multi_threshold) and n_flagged <= cap:
                positions = np.where(np.abs(gene_res) > self.multi_threshold)[0]
                for pos in positions:
                    orig_idx = int(order[pos])     # original sample index
                    if orig_idx in confirmed:
                        mult_outs.append(orig_idx)
                        n_flagged += 1

            # ── Univariante: |res| > uni_threshold
            # R lineas 4069-4074: agrega TODAS las muestras malas para este gen,
            # pero incrementa nOutsTot en 1 (por gen, no por muestra).
            if np.any(np.abs(gene_res) > self.uni_threshold) and n_flagged <= cap:
                positions = np.where(np.abs(gene_res) > self.uni_threshold)[0]
                for pos in positions:
                    orig_idx = int(order[pos])
                    uni_outs.append(orig_idx)
                n_flagged += 1          # R: un incremento por gen

        # Deduplicar preservando orden
        uni_outs  = list(dict.fromkeys(uni_outs))
        mult_outs = list(dict.fromkeys(mult_outs))

        return uni_outs, mult_outs

    def _rerun_cpca(
        self,
        cpca:    CPCAResult,
        dropped: list[int],
    ) -> tuple[CPCAResult, pd.DataFrame]:
        """
        Elimina muestras outlier, renormaliza la matriz core, re-ejecuta CPCA.

        Equivalente en R: lineas 4084-4090.

        Devuelve
        --------
        cpca_final      : nuevo CPCAResult sobre datos limpios
        core_norm_clean : DataFrame core renormalizado (genes × muestras limpias)
        """
        # Matriz core original (cruda — antes de normalizacion)
        # Almacenamos la version normalizada en cpca.core_matrix.
        # La renormalizacion tras eliminar requiere los valores normalizados
        # porque mTissueCoreG es la matriz core CRUDA en R.
        # Sin embargo, como no almacenamos valores core crudos, renormalizamos
        # desde la matriz ya normalizada — el resultado es el mismo
        # porque normalice() es idempotente cuando los valores ya estan en [-1,1]:
        # eliminar muestras cambia el min/max → la renormalizacion ES necesaria.
        core_matrix = cpca.core_matrix   # DataFrame (genes × samples)

        # Eliminar columnas outlier por indice de muestra original
        all_cols    = np.arange(core_matrix.shape[1])
        keep_mask   = ~np.isin(all_cols, dropped)
        core_clean  = core_matrix.iloc[:, keep_mask]

        # Renormalizar (R linea 4085)
        core_norm_clean = _normalise_matrix(core_clean)

        # Re-ejecutar CPCA (R linea 4087: obtainCPCA13(mTissueCoreGNorm,...))
        cpca_final = CPCA(
            core_genes           = cpca.core_genes_found,
            n_outlier_candidates = self.cpca_n_outlier_candidates,
            tight_radius         = self.cpca_tight_radius,
            loose_radius         = self.cpca_loose_radius,
            verbose              = False,
        ).run(core_norm_clean)

        return cpca_final, core_norm_clean

    def _order_full_matrix(
        self,
        expr_norm:    pd.DataFrame,
        dropped:      list[int],
        sample_order: np.ndarray,
    ) -> pd.DataFrame:
        """
        Elimina muestras outlier de la matriz completa, luego reordena
        por el ordenamiento circular CPCA final.

        Equivalente en R: lineas 4094-4101.

        El resultado se renormaliza gen por gen porque eliminar muestras
        cambia el min/max por gen (R linea 4099: t(apply(..., normalice))).

        Devuelve
        --------
        pd.DataFrame, forma (n_genes, n_muestras_limpias)
            Matriz completa con outliers eliminados, ordenada por fase
            circular, renormalizada.
        """
        # Eliminar columnas outlier
        all_cols  = np.arange(expr_norm.shape[1])
        keep_mask = ~np.isin(all_cols, dropped)
        mat_clean = expr_norm.iloc[:, keep_mask]

        # Reordenar columnas por fase circular (R lineas 4095-4097)
        mat_ordered = mat_clean.iloc[:, sample_order]

        # Renormalizar (R linea 4099)
        mat_norm = _normalise_matrix(mat_ordered)

        return mat_norm

    def _fit_final(
        self,
        expr_norm_final: pd.DataFrame,
        cpca_final:      "CPCAResult",
    ) -> tuple[dict, dict]:
        """
        Ajusta FMM en cada gen central usando el ordenamiento CPCA final.

        Equivalente en R: lineas 4109-4136 de ``giveMatIniNP_v3_cores``
        (``allParAfter``, ``phisFMMAfter``).

        Parametros
        ----------
        expr_norm_final : pd.DataFrame
            Matriz de expresion normalizada completa ya ordenada por el
            ordenamiento CPCA final (salida de ``_order_full_matrix``).
        cpca_final : CPCAResult
            Resultado CPCA final; proporciona ``circular_scale`` (el eje
            temporal) y ``core_genes_found``.

        Devuelve
        --------
        fmm_fits_final : {gen: FitResult}
        fmm_peak_times_final : {gen: float}  — tiempos de pico compUU
        """
        core_genes     = cpca_final.core_genes_found
        circular_scale = cpca_final.circular_scale   # escalaPhi8 (final)

        fmm_model = FMMModel(
            length_alpha_grid = self.fmm_length_alpha_grid,
            length_omega_grid = self.fmm_length_omega_grid,
            num_reps          = self.fmm_num_reps,
        )

        fmm_fits_fin:  dict[str, FitResult] = {}
        peak_times_fin: dict[str, float]    = {}

        n_total = len(core_genes)
        for idx, gene in enumerate(core_genes, 1):
            self._log(f"    [{idx}/{n_total}] {gene}", end="\r")

            if gene not in expr_norm_final.index:
                self._log(f"    AVISO: {gene} no esta en la matriz final, omitiendo.")
                continue

            data = expr_norm_final.loc[gene].values
            fr   = fmm_model.fit(data, circular_scale)

            fmm_fits_fin[gene]   = fr
            peak_times_fin[gene] = fmm_peak_time(
                fr.params["alpha"], fr.params["beta"], fr.params["omega"]
            )

        self._log(f"    Ajuste completado de {len(fmm_fits_fin)} genes centrales (final).      ")
        return fmm_fits_fin, peak_times_fin

    # ------------------------------------------------------------------
    # Utilidad
    # ------------------------------------------------------------------

    def _log(self, message: str, end: str = "\n") -> None:
        if self.verbose:
            print(message, end=end, flush=True)