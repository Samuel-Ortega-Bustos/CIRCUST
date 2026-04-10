"""
circust/preliminary_order.py
============================
Etapa 2: Ordenamiento Circular Preliminar.

Implementa ``basicPreOder_cores`` (Paso 2.1) y ``basicOder_cores``
(Pasos 2.2 y 2.3) de R en ``functionGTEX_cores.R``.

Objetivo
--------
El ordenamiento circular derivado de CPCA es invariante a rotacion y tiene
una direccion arbitraria (sentido horario o antihorario). Esta etapa corrige
ambos problemas usando conocimiento biologico previo sobre el reloj
circadiano mamifero:

  * **ARNTL** tiene su pico cerca del tiempo Zeitgeber 0 (medianoche/amanecer
    subjetivo). Tras la rotacion se situa en π en la escala circular.

  * **DBP** tiene su pico en la fase activa, que deberia caer en [0, π)
    cuando ARNTL esta en π. Si DBP termina en [π, 2π) la direccion del
    ordenamiento es incorrecta y se invierte.

Paso 2.1 — ``basicPreOder_cores``
-----------------------------------
1. Rotar la escala circular para que el tiempo de pico FMM de ARNTL
   quede en π.
2. Determinar direccion: si (DBP_peak − ARNTL_peak + π) mod 2π < π
   entonces DBP esta en la primera mitad → direccion directa; si no,
   invertir.
3. Reparametrizar el ajuste FMM de cada gen central en el nuevo marco
   de coordenadas.
4. Clasificar los genes centrales restantes como *dia* (pico ∈ [0, π))
   o *noche* (pico ∈ [π, 2π)).

Paso 2.2 y 2.3 — ``basicOder_cores``
--------------------------------------
Aplica una verificacion de consistencia biologica refinada. Si alguna de
las siguientes condiciones se cumple (``aviso = True``):
  - DBP esta demasiado cerca de 0 (1 − cos(DBP_peak) ≤ 0.1)
  - DBP esta demasiado cerca de π (1 − cos(DBP_peak − π) ≤ 0.1)
  - Menos de la mitad de los genes sin ARNTL/DBP tienen pico en [0, π)
Y el orden DBP < CRY1 < ARNTL NO se cumple, entonces invertir la direccion.

Lineas fuente en R
-------------------
* Paso 2.1: ``basicPreOder_cores``  (lineas 4192-4263)
* Paso 2.2: ``basicOder_cores``     (lineas 4527-4594)

Posicion en el pipeline
-----------------------
    OutlierRefiner  →  PreliminaryOrderEstimator  →  (Etapa 3)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from math import pi

from circust.outlier import OutlierRefinementResult
from circust.fitting.fmm import FMMModel
from circust.fitting.rhythm_model import FitResult


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class PreliminaryOrderResult:
    """
    Salida combinada de la Etapa 2.1 y Etapas 2.2 y 2.3.

    Atributos
    ---------
    sample_order : np.ndarray de int, forma (n_muestras,)
        Indices finales de muestra en la matriz de expresion limpia.
        Equivalente en R: ``basicOrdRefG2[[1]]``  (= ``indSin``).

    circular_scale : np.ndarray de float, forma (n_muestras,)
        Eje temporal circular en [0, 2π) tras correccion de direccion.
        Equivalente en R: ``basicOrdRefG2[[2]]``  (= ``escSincroRefG``).

    expr_ordered : pd.DataFrame, forma (n_genes, n_muestras)
        Matriz de expresion normalizada completa en el orden circular final.
        Equivalente en R: ``basicOrdRefG2[[3]]``  (= ``matNewNew``).

    peak_times : np.ndarray de float, forma (n_genes_core,)
        Tiempo de pico FMM para cada gen central en el marco de coordenadas final.
        Equivalente en R: ``basicOrdRefG2[[4]]``  (= ``peaksPreNew``).

    r2_fmm : np.ndarray de float, forma (n_genes_core,)
        R² del ajuste FMM para cada gen central (de los ajustes con
        ordenamiento final).
        Equivalente en R: ``basicOrdRefG2[[5]]``.

    fmm_params : np.ndarray de float, forma (n_genes_core, 5)
        Parametros FMM [M, A, α, β, ω] por gen central en el nuevo marco.
        Equivalente en R: ``basicOrdRefG2[[6]]``  (= ``pars``).

    direction_flipped : bool
        True si la direccion fue invertida por el Paso 2.1 o 2.2.
        Equivalente en R: ``basicOrdRefG2[[7]]``  (= ``cambiaOri``).

    within_group_indices : np.ndarray de int, forma (n_muestras,)
        Indices posicionales base 0 dentro del grupo tras posible inversion.
        Equivalente en R: ``basicOrdRefG2[[8]]``  (= ``indNewNew``), pero
        convertido a base 0 (R usa base 1).

    core_genes : list[str]
        Simbolos de genes centrales en el orden utilizado.

    day_genes : list[str]
        Genes centrales (exc. ARNTL, DBP) con pico en [0, π).

    night_genes : list[str]
        Genes centrales (exc. ARNTL, DBP) con pico en [π, 2π).

    pre_sample_order : np.ndarray de int
        Orden de muestras del Paso 2.1 (antes de la verificacion del Paso 2.2).
        Equivalente en R: ``preOrdRefG2[[1]]``.

    pre_circular_scale : np.ndarray de float
        Escala circular del Paso 2.1.
        Equivalente en R: ``preOrdRefG2[[2]]``.

    pre_expr_ordered : pd.DataFrame
        Matriz de expresion ordenada por el Paso 2.1.
        Equivalente en R: ``preOrdRefG2[[3]]``.

    pre_peak_times : np.ndarray de float
        Tiempos de pico FMM del Paso 2.1.
        Equivalente en R: ``preOrdRefG2[[4]]``.

    pre_fmm_params : np.ndarray de float, forma (n_genes_core, 5)
        Parametros FMM del Paso 2.1.
        Equivalente en R: ``preOrdRefG2[[6]]``  (= ``parCore``).

    pre_direction_reversed : bool
        True si el Paso 2.1 invirtio la direccion (DBP en la mitad incorrecta).
        Equivalente en R: ``preOrdRefG2[[13]]``  (= ``reverse``).
    """

    sample_order:           np.ndarray
    circular_scale:         np.ndarray
    expr_ordered:           pd.DataFrame
    peak_times:             np.ndarray
    r2_fmm:                 np.ndarray
    fmm_params:             np.ndarray
    direction_flipped:      bool
    within_group_indices:   np.ndarray
    core_genes:             list

    day_genes:              list       = field(default_factory=list)
    night_genes:            list       = field(default_factory=list)

    pre_sample_order:       np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    pre_circular_scale:     np.ndarray = field(default_factory=lambda: np.array([]))
    pre_expr_ordered:       pd.DataFrame = field(default_factory=pd.DataFrame)
    pre_peak_times:         np.ndarray = field(default_factory=lambda: np.array([]))
    pre_fmm_params:         np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    pre_direction_reversed: bool       = False

    def summary(self) -> str:
        lines = [
            "=== Resumen de Ordenamiento Preliminar ===",
            f"  Direccion invertida  : {self.direction_flipped}",
            f"  Paso 2.1 invertido  : {self.pre_direction_reversed}",
            f"  Genes dia  (0..π)   : {self.day_genes}",
            f"  Genes noche (π..2π) : {self.night_genes}",
            f"  Picos genes core    : {np.round(self.peak_times, 3).tolist()}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PreliminaryOrderEstimator
# ---------------------------------------------------------------------------

class PreliminaryOrderEstimator:
    """
    Implementa la Etapa 2 de CIRCUST: establecer el marco de referencia
    biologico.

    Ejecuta dos sub-pasos reflejando el pipeline de R:
      1. ``basicPreOder_cores`` — rota para que ARNTL quede en π, detecta
         la direccion desde la posicion de DBP.
      2. ``basicOder_cores``   — verificacion de consistencia refinada;
         invierte si la biologia parece inconsistente.

    Parametros
    ----------
    arntl_gene : str
        Gen ancla que se espera tenga su pico cerca de la medianoche
        subjetiva. Por defecto: ``"ARNTL"``.

    dbp_gene : str
        Gen determinante de direccion que se espera tenga su pico en la
        fase activa (primera mitad, [0, π)). Por defecto: ``"DBP"``.

    cry1_gene : str
        Gen usado en la verificacion de consistencia refinada.
        Por defecto: ``"CRY1"``.

    verbose : bool
        Imprimir mensajes de progreso.
    """

    def __init__(
        self,
        arntl_gene: str  = "ARNTL",
        dbp_gene:   str  = "DBP",
        cry1_gene:  str  = "CRY1",
        verbose:    bool = True,
    ) -> None:
        self.arntl_gene = arntl_gene
        self.dbp_gene   = dbp_gene
        self.cry1_gene  = cry1_gene
        self.verbose    = verbose

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(
        self,
        refined:    OutlierRefinementResult,
        core_genes: list[str],
    ) -> PreliminaryOrderResult:
        """
        Ejecuta la Etapa 2 (Pasos 2.1 y 2.2).

        Parametros
        ----------
        refined : OutlierRefinementResult
            Salida de ``OutlierRefiner.run()``. Proporciona
            ``fmm_fits_final``, ``fmm_peak_times_final``,
            ``expr_norm_final`` y ``cpca_final``.

        core_genes : list de str
            Lista ordenada de simbolos de genes reloj centrales (mismo orden
            usado en ``CPCA`` y ``OutlierRefiner``).

        Devuelve
        --------
        PreliminaryOrderResult
        """
        self._log("=== Etapa 2: Ordenamiento Preliminar ===")

        # ── Paso 2.1: basicPreOder_cores ─────────────────────────────────
        self._log("  Paso 2.1 — rotando al marco de referencia ARNTL ...")
        (o_pre, esc_pre, mat_pre, peaks_pre, r2_fmm, par_pre,
         names_day, names_night, reversed_21) = self._pre_order(
            refined, core_genes
        )

        # ── Paso 2.2: basicOder_cores ─────────────────────────────────────
        self._log("  Paso 2.2 — verificando consistencia biologica ...")
        (o_fin, esc_fin, mat_fin, peaks_fin,
         pars_fin, flipped_22, ind_new) = self._basic_order(
            o_pre, esc_pre, mat_pre, peaks_pre, par_pre, core_genes
        )

        # Flag de direccion global (cualquiera de los pasos pudo haber invertido)
        direction_flipped = reversed_21 ^ flipped_22

        result = PreliminaryOrderResult(
            sample_order           = o_fin,
            circular_scale         = esc_fin,
            expr_ordered           = mat_fin,
            peak_times             = peaks_fin,
            r2_fmm                 = r2_fmm,
            fmm_params             = pars_fin,
            direction_flipped      = direction_flipped,
            within_group_indices   = ind_new,
            core_genes             = core_genes,
            day_genes              = names_day,
            night_genes            = names_night,
            pre_sample_order       = o_pre,
            pre_circular_scale     = esc_pre,
            pre_expr_ordered       = mat_pre,
            pre_peak_times         = peaks_pre,
            pre_fmm_params         = par_pre,
            pre_direction_reversed = reversed_21,
        )

        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Paso 2.1 — basicPreOder_cores
    # ------------------------------------------------------------------

    def _pre_order(
        self,
        refined:    OutlierRefinementResult,
        core_genes: list[str],
    ) -> tuple:
        """
        Equivalente en R: ``basicPreOder_cores`` (lineas 4192-4263).

        Devuelve
        --------
        o_new       : np.ndarray  — indices de ordenamiento de muestras (columnas de expr)
        esc_new     : np.ndarray  — escala circular en [0, 2π)
        mat_new     : pd.DataFrame — matriz completa en nuevo orden
        peaks       : np.ndarray  — tiempos de pico FMM por gen central
        r2_fmm      : np.ndarray  — R² por gen central (de ajustes finales)
        par_core    : np.ndarray  forma (n_core, 5) — [M, A, α, β, ω]
        names_day   : list[str]   — genes centrales con pico en [0, π)
        names_night : list[str]   — genes centrales con pico en [π, 2π)
        reversed    : bool        — True si la direccion se invirtio aqui
        """
        circular_scale = refined.cpca_final.circular_scale   # escalaPhi8
        expr_ordered   = refined.expr_norm_final              # mFullTissueNorm
        fmm_fits       = refined.fmm_fits_final               # allParAfter
        peak_times_fin = refined.fmm_peak_times_final         # phisFMMAfter

        n_core = len(core_genes)

        # ── Valores R² de ajustes FMM con ordenamiento final (m3r2[:,0]) ──
        r2_fmm = np.array([
            fmm_fits[g].r2 if g in fmm_fits else float("nan")
            for g in core_genes
        ])

        # ── Tiempos de pico ARNTL y DBP en el marco CPCA original ────────
        arntl_peak = peak_times_fin[self.arntl_gene]  # peakRefEpidermis2
        dbp_peak   = peak_times_fin[self.dbp_gene]    # peakDBPEpidermis2

        # ── Rotar escala circular: ARNTL → π ─────────────────────────────
        # R: escTEpidermis <- order((listaStep1[[2]] - peakRef + pi) %% (2π))
        rotated = (circular_scale - arntl_peak + pi) % (2.0 * pi)
        esc_T   = np.argsort(rotated)          # escTEpidermis
        esc2    = rotated[esc_T]               # esc2

        # ── Direccion: directa o inversa ─────────────────────────────────
        # R: if((peakDBP - peakRef + π) %% (2π) < π) → forward
        dbp_rotated = (dbp_peak - arntl_peak + pi) % (2.0 * pi)
        forward     = dbp_rotated < pi          # True → mantener direccion

        # ── Reparametrizar el FMM de cada gen central en el nuevo marco ───
        # R (per gene i):
        #   newAlpha = (alpha_i - peakRef + π) %% 2π
        #   if forward: peak_i = compUU(newAlpha, beta_i, omega_i)
        #               par_i  = [M, A, newAlpha, beta, omega]
        #   else:       peak_i = compUU(2π-newAlpha, 2π-beta, omega)
        #               par_i  = [M, A, 2π-newAlpha, 2π-beta, omega]
        peaks    = np.zeros(n_core)
        par_core = np.zeros((n_core, 5))

        for i, gene in enumerate(core_genes):
            if gene not in fmm_fits:
                continue
            fr = fmm_fits[gene]
            alpha = fr.params["alpha"]
            beta  = fr.params["beta"]
            omega = fr.params["omega"]
            M     = fr.params["M"]
            A     = fr.params["A"]

            new_alpha = (alpha - arntl_peak + pi) % (2.0 * pi)

            if forward:
                peaks[i]    = FMMModel.peak_time(new_alpha, beta, omega)
                par_core[i] = [M, A, new_alpha, beta, omega]
            else:
                na2 = (2.0 * pi - new_alpha) % (2.0 * pi)
                nb2 = (2.0 * pi - beta)      % (2.0 * pi)
                peaks[i]    = FMMModel.peak_time(na2, nb2, omega)
                par_core[i] = [M, A, na2, nb2, omega]

        # ── Reordenar columnas de la matriz ───────────────────────────────
        if forward:
            o_new   = esc_T
            esc_new = esc2
            mat_new = expr_ordered.iloc[:, esc_T].copy()
        else:
            o_new   = esc_T[::-1].copy()
            esc_new = (2.0 * pi - esc2[::-1]) % (2.0 * pi)
            mat_new = expr_ordered.iloc[:, esc_T[::-1]].copy()

        mat_new.columns = range(mat_new.shape[1])

        # ── Clasificar como dia / noche (excluyendo ARNTL y DBP) ─────────
        names_day   = []
        names_night = []
        for i, gene in enumerate(core_genes):
            if gene == self.arntl_gene or gene == self.dbp_gene:
                continue
            if 0 <= peaks[i] < pi:
                names_day.append(gene)
            else:
                names_night.append(gene)

        self._log(
            f"    ARNTL peak: {arntl_peak:.3f} rad  |  "
            f"DBP rotated: {dbp_rotated:.3f} rad  |  "
            f"Direccion: {'directa' if forward else 'INVERTIDA'}"
        )

        return (o_new, esc_new, mat_new, peaks, r2_fmm, par_core,
                names_day, names_night, not forward)

    # ------------------------------------------------------------------
    # Paso 2.2 — basicOder_cores
    # ------------------------------------------------------------------

    def _basic_order(
        self,
        o:          np.ndarray,
        esc:        np.ndarray,
        mat:        pd.DataFrame,
        peaks:      np.ndarray,
        par:        np.ndarray,
        core_genes: list[str],
    ) -> tuple:
        """
        Equivalente en R: ``basicOder_cores`` (lineas 4527-4594).

        Verifica consistencia biologica e invierte si es necesario.

        Devuelve
        --------
        o_new      : np.ndarray
        esc_new    : np.ndarray
        mat_new    : pd.DataFrame
        peaks_new  : np.ndarray
        pars_new   : np.ndarray  forma (n_core, 5)
        flipped    : bool
        ind_new    : np.ndarray  (indices posicionales base 0)
        """
        arntl_i = core_genes.index(self.arntl_gene) if self.arntl_gene in core_genes else None
        dbp_i   = core_genes.index(self.dbp_gene)   if self.dbp_gene   in core_genes else None
        cry1_i  = core_genes.index(self.cry1_gene)  if self.cry1_gene  in core_genes else None

        if dbp_i is None:
            self._log(
                f"  AVISO: {self.dbp_gene} no encontrado en core_genes; "
                "omitiendo verificacion de direccion."
            )
            n = len(o)
            return o, esc, mat, peaks, par, False, np.arange(n)

        dbp_peak  = peaks[dbp_i]
        n_core    = len(core_genes)

        # Contar genes en [0, π) excluyendo ARNTL y DBP
        peak_d = sum(
            1 for i, g in enumerate(core_genes)
            if g != self.arntl_gene and g != self.dbp_gene
            and 0 < peaks[i] < pi
        )
        mitad = int(np.floor((n_core - 2) / 2))

        # condiciones aviso (R lineas 4531-4537)
        p6am  = 1.0 - np.cos(dbp_peak)          # 1-cos(DBP) — distance from 0
        p6pm  = 1.0 - np.cos(dbp_peak - pi)     # 1-cos(DBP-π) — distance from π
        aviso = (p6am <= 0.1) or (p6pm <= 0.1) or (peak_d < mitad)

        # Exclusion: no invertir si DBP < CRY1 < ARNTL (todo ascendente)
        if aviso and cry1_i is not None and arntl_i is not None:
            excl  = (peaks[dbp_i] < peaks[cry1_i] < peaks[arntl_i])
            flip  = not excl
        elif aviso:
            flip  = True
        else:
            flip  = False

        n = len(o)

        if flip:
            self._log(
                f"    Invirtiendo direccion  "
                f"(aviso=True, DBP_peak={dbp_peak:.3f}, "
                f"peak_d={peak_d}/{mitad})"
            )
            peaks_new = (2.0 * pi - peaks) % (2.0 * pi)
            o_new     = o[::-1].copy()
            esc_new   = (2.0 * pi - esc[::-1]) % (2.0 * pi)
            mat_new   = mat.iloc[:, ::-1].copy()
            mat_new.columns = range(mat_new.shape[1])
            pars_new        = par.copy()
            pars_new[:, 2]  = (2.0 * pi - par[:, 2]) % (2.0 * pi)  # α
            pars_new[:, 3]  = (2.0 * pi - par[:, 3]) % (2.0 * pi)  # β
            ind_new   = np.arange(n - 1, -1, -1)
        else:
            self._log("    La direccion es consistente — no se necesita inversion.")
            peaks_new = peaks.copy()
            o_new     = o.copy()
            esc_new   = esc.copy()
            mat_new   = mat.copy()
            pars_new  = par.copy()
            ind_new   = np.arange(n)

        return o_new, esc_new, mat_new, peaks_new, pars_new, flip, ind_new

    # ------------------------------------------------------------------
    # Utilidad
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
