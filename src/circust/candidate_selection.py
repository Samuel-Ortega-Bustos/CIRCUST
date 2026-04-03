"""
circust/candidate_selection.py
==============================
Etapa 3.2: Seleccion del conjunto de genes candidatos mediante filtrado
por sectores circulares.

Selecciona genes cuyos perfiles de expresion muestran un comportamiento
ritmico fuerte segun criterios no parametricos (R² NP) y parametricos (FMM).
Los genes se dividen en 8 sectores de 45° cada uno sobre el eje de fase
circular, y dentro de cada sector se retienen los mejores candidatos.

Algoritmo
---------
1. Filtrar genes con R² NP por encima de la mediana.
2. Calcular la acrofase Cosinor para cada gen superviviente.
3. Definir 8 sectores equiespaciados de 45° centrados en la mediana
   circular de todas las acrofases.
4. Dentro de cada sector, iterar por genes ordenados por R² NP (descendente):
   - Ajustar FMM y verificar ω > 0.1 (no demasiado puntiagudo) y
     R²_par > umbral.
   - Aceptar hasta *tam* genes por sector, continuando mas alla de *tam*
     solo mientras el R² NP del gen permanezca por debajo del percentil 75
     de la distribucion completa.
5. Recopilar todos los genes aceptados de todos los sectores.

Equivalente en R
-----------------
``minSetCand3_v2_cores()`` (lineas 1242–1692 de ``functionGTEX_cores.R``).

Posicion en el pipeline
-----------------------
    NonParametricScorer  →  **CandidateSelector**  →  ReferenceSetBuilder
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from math import pi

from circust.fitting.cosinor import CosinorModel
from circust.fitting.fmm import FMMModel, fmm_peak_time


# ═══════════════════════════════════════════════════════════════════════════
# Funciones auxiliares de estadistica circular
# ═══════════════════════════════════════════════════════════════════════════

def circular_median(angles: np.ndarray) -> float:
    """
    Mediana circular: el angulo θ que minimiza  Σ d(θ, θᵢ)  donde
    *d* es la distancia geodesica (longitud de arco) en el circulo unitario.

    Coincide con la definicion de ``circular::median.circular()`` de R.

    Parametros
    ----------
    angles : array (n,) en [0, 2π).

    Devuelve
    --------
    float — mediana circular en [0, 2π).
    """
    angles = np.asarray(angles, dtype=np.float64) % (2.0 * pi)
    n = len(angles)
    if n == 0:
        return 0.0
    if n == 1:
        return float(angles[0])

    # Evaluar coste en cada angulo observado (O(n²) — adecuado para ≤ 20 000
    # genes; el paquete circular de R hace lo mismo).
    best_cost  = np.inf
    best_angle = angles[0]
    for candidate in angles:
        diffs = np.abs(angles - candidate)
        diffs = np.minimum(diffs, 2.0 * pi - diffs)
        cost  = diffs.sum()
        if cost < best_cost:
            best_cost  = cost
            best_angle = candidate

    return float(best_angle % (2.0 * pi))


def assign_sectors(
    peaks: np.ndarray,
    reference: float,
    n_sectors: int = 8,
) -> np.ndarray:
    """
    Asigna cada angulo de pico a uno de *n_sectors* sectores equiespaciados
    centrados en *reference*.

    Los sectores se numeran 1 … n_sectors. El sector 1 contiene los picos
    mas cercanos a *reference* (en direccion antihoraria); el sector
    *n_sectors* es el mas lejano.

    Vectoriza la estructura de 8 if en cascada de R (lineas 1300–1334).

    Parametros
    ----------
    peaks     : array (n,) de angulos en [0, 2π).
    reference : float — mediana circular usada como origen.
    n_sectors : int — por defecto 8 (45° cada uno).

    Devuelve
    --------
    array int (n,) — etiquetas de sector en {1, 2, …, n_sectors}.
    """
    peaks = np.asarray(peaks, dtype=np.float64)
    # Rotar para que reference ≡ 0, luego mapear a sector.
    rotated = (peaks - reference) % (2.0 * pi)
    # R asigna sector 1 al arco MAS CERCANO a reference (valores rotados
    # mas altos, justo debajo de 2π), sector 8 al arco centrado en
    # reference − 7π/4.  El codigo R comprueba sectores de arriba a abajo:
    #   if rotated > boundary_2  → sector 1
    #   if boundary_3 < rotated < boundary_2 → sector 2
    #   …
    #   if rotated < boundary_8  → sector 8
    # Los limites estan en k·π/4 para k = 1…7.
    # Mapeo: sector = n_sectors − floor(rotated / (2π / n_sectors)).
    sector_width = 2.0 * pi / n_sectors
    sectors = n_sectors - np.floor(rotated / sector_width).astype(int)
    # Limitar a [1, n_sectors] para casos limite de punto flotante.
    sectors = np.clip(sectors, 1, n_sectors)
    return sectors


def sector_boundaries(reference: float, n_sectors: int = 8) -> np.ndarray:
    """
    Devuelve los *n_sectors* puntos centrales de sector (m1…m8 en R).

    Equivalente en R: ``m1 = peakRef;  m2 = (peakRef − π/4) mod 2π; …``
    (lineas 1280–1287).
    """
    return np.array([
        (reference - k * pi / (n_sectors / 2)) % (2.0 * pi)
        for k in range(n_sectors)
    ])


def reduce_sector(
    peaks: np.ndarray,
    r2: np.ndarray,
    names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Elimina iterativamente el gen con menor R² de un sector hasta que todos
    los genes restantes superan el umbral del percentil 75 de R² **o** el
    sector ya tiene ≤ 25 genes.

    Equivalente en R: ``reduceSectors_v2()`` (lineas 140–161).

    Parametros
    ----------
    peaks : array (k,) de angulos de pico.
    r2    : array (k,) de valores de R².
    names : array (k,) de nombres de genes (cadenas).

    Devuelve
    --------
    (peaks, r2, names) — arrays filtrados.
    """
    peaks = np.array(peaks, dtype=np.float64)
    r2    = np.array(r2, dtype=np.float64)
    names = np.array(names, dtype=object)

    if len(r2) <= 1:
        return peaks, r2, names

    q75 = float(np.quantile(r2, 0.75))

    # Condicion R: seguir podando mientras (n > 25 O count_above_Q75 > 25).
    while len(r2) > 0:
        n_above = int(np.sum(r2 > q75))
        if not (len(peaks) > 25 or n_above > 25):
            break
        if r2.min() >= q75:
            break
        drop = int(np.argmin(r2))
        peaks = np.delete(peaks, drop)
        r2    = np.delete(r2, drop)
        names = np.delete(names, drop)

    return peaks, r2, names


# ═══════════════════════════════════════════════════════════════════════════
# Dataclass de resultado
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateResult:
    """
    Salida de :class:`CandidateSelector`.

    Atributos
    ---------
    gene_names : list[str]
        Simbolos de genes candidatos aceptados.
        R: ``namesIn``  → ``outSetCandRefG[[1]]``.

    expr_data : pd.DataFrame, forma (n_aceptados, n_muestras)
        Filas de expresion para genes aceptados.
        R: ``datosParTissue``  → ``outSetCandRefG[[2]]``.

    fmm_fitted : pd.DataFrame
        Valores ajustados FMM.
        R: ``adjParamTissue``  → ``outSetCandRefG[[3]]``.

    fmm_params : np.ndarray, forma (n_aceptados, 5)
        Parametros FMM [M, A, α, β, ω] por gen.
        R: ``paramParamTissue``  → ``outSetCandRefG[[4]]``.

    cos_fitted : pd.DataFrame
        Valores ajustados Cosinor.
        R: ``adjCosTissue``  → ``outSetCandRefG[[5]]``.

    cos_params : np.ndarray, forma (n_aceptados, 3)
        Parametros Cosinor [M, A, φ] por gen.
        R: ``paramCosTissue``  → ``outSetCandRefG[[6]]``.

    fmm_peaks : np.ndarray, forma (n_aceptados,)
        Tiempos de pico FMM (via compUU) por gen.
        R: ``peaksIn``  → ``outSetCandRefG[[7]]``.

    sector_labels : np.ndarray de int, forma (n_aceptados,)
        Asignacion de sector (1–8) por gen.
        R: ``regIn``  → ``outSetCandRefG[[8]]``.

    r2_par : np.ndarray, forma (n_aceptados,)
        R² parametrico por gen.
        R: ``r2ParIn``  → ``outSetCandRefG[[9]]``.

    ranked_r2_np : np.ndarray
        R² NP de todos los genes ordenado descendente (referencia de distribucion).
        R: ``mDistR2Tissue``  → ``outSetCandRefG[[11]]``.

    sector_centres : np.ndarray, forma (8,)
        Los 8 puntos centrales de sector m1…m8.
        R: ``outSetCandRefG[[20]]`` … ``outSetCandRefG[[27]]``.

    discarded_genes : list[str]
        Genes examinados pero rechazados (fallo en ω o R²_par).
        R: ``nameDiscard``  → ``outSetCandRefG[[28]]``.
    """

    gene_names:      list        = field(default_factory=list)
    expr_data:       pd.DataFrame = field(default_factory=pd.DataFrame)
    fmm_fitted:      pd.DataFrame = field(default_factory=pd.DataFrame)
    fmm_params:      np.ndarray  = field(default_factory=lambda: np.zeros((0, 5)))
    cos_fitted:      pd.DataFrame = field(default_factory=pd.DataFrame)
    cos_params:      np.ndarray  = field(default_factory=lambda: np.zeros((0, 3)))
    fmm_peaks:       np.ndarray  = field(default_factory=lambda: np.array([]))
    sector_labels:   np.ndarray  = field(default_factory=lambda: np.array([], dtype=int))
    r2_par:          np.ndarray  = field(default_factory=lambda: np.array([]))
    ranked_r2_np:    np.ndarray  = field(default_factory=lambda: np.array([]))
    sector_centres:  np.ndarray  = field(default_factory=lambda: np.zeros(8))
    discarded_genes: list        = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.gene_names)
        n_sectors_used = len(np.unique(self.sector_labels)) if n > 0 else 0
        lines = [
            "=== Resumen de Seleccion de Candidatos ===",
            f"  Genes aceptados     : {n}",
            f"  Sectores poblados   : {n_sectors_used} / 8",
            f"  Genes descartados   : {len(self.discarded_genes)}",
        ]
        if n > 0:
            for s in range(1, 9):
                cnt = int(np.sum(self.sector_labels == s))
                if cnt > 0:
                    lines.append(f"    Sector {s}: {cnt} gene(s)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CandidateSelector
# ═══════════════════════════════════════════════════════════════════════════

class CandidateSelector:
    """
    Selecciona genes ritmicos candidatos mediante filtrado por 8 sectores
    validado con FMM.

    Equivalente en R: ``minSetCand3_v2_cores()`` (lineas 1242–1692).

    Parametros
    ----------
    tam : int
        Numero objetivo de genes por sector antes de que se active la
        regla de "continuar mas alla del objetivo". Por defecto en R: 50.

    omega_min : float
        ω minimo de FMM para aceptar un gen (rechaza ajustes puntiagudos).
        Por defecto en R: 0.1.

    r2_par_floor : float
        Umbral absoluto minimo para R² parametrico; el umbral efectivo es
        ``min(Q1_core_R2, r2_par_floor)``. Por defecto en R: 0.4.

    fmm_length_alpha_grid : int
        Tamano de la rejilla para FMM α. Por defecto en R: 48.

    fmm_length_omega_grid : int
        Tamano de la rejilla para FMM ω. Por defecto en R: 24.

    fmm_num_reps : int
        Iteraciones de refinamiento FMM. Por defecto en R: 3.

    verbose : bool
        Imprimir mensajes de progreso.
    """

    def __init__(
        self,
        tam:                   int   = 50,
        omega_min:             float = 0.1,
        r2_par_floor:          float = 0.4,
        fmm_length_alpha_grid: int   = 48,
        fmm_length_omega_grid: int   = 24,
        fmm_num_reps:          int   = 3,
        verbose:               bool  = True,
    ) -> None:
        self.tam                   = tam
        self.omega_min             = omega_min
        self.r2_par_floor          = r2_par_floor
        self.fmm_length_alpha_grid = fmm_length_alpha_grid
        self.fmm_length_omega_grid = fmm_length_omega_grid
        self.fmm_num_reps          = fmm_num_reps
        self.verbose               = verbose

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(
        self,
        r2_np:            np.ndarray,
        expr_norm:        pd.DataFrame,
        circular_scale:   np.ndarray,
        r2_par_core:      np.ndarray,
    ) -> CandidateResult:
        """
        Selecciona genes candidatos.

        Parametros
        ----------
        r2_np : array (n_genes,)
            R² no parametrico para cada gen en la matriz de expresion.
            R: ``obtainNPRefG[[5]]``.

        expr_norm : pd.DataFrame, forma (n_genes, n_muestras)
            Matriz de expresion normalizada completa ordenada por el
            tiempo circular preliminar.
            R: ``preOrdRefG2[[3]]``  (= ``mFullTissueNorm``).

        circular_scale : array (n_muestras,)
            Eje temporal circular del ordenamiento preliminar.
            R: ``preOrdRefG2[[2]]``  (= ``escParTissue``).

        r2_par_core : array (n_core,)
            R² parametrico de los genes reloj centrales (de los ajustes
            FMM del ordenamiento preliminar). Se usa para fijar el umbral
            de aceptacion.
            R: ``basicOrdRefG2[[5]]``  (= ``R2ParcoreG``).

        Devuelve
        --------
        CandidateResult
        """
        self._log("=== Etapa 3.2: Seleccion de Genes Candidatos ===")

        all_genes   = expr_norm.index.values
        n_genes     = len(all_genes)
        r2_np       = np.asarray(r2_np, dtype=np.float64)
        circ        = np.asarray(circular_scale, dtype=np.float64)

        # ── 1. Distribucion de R² NP ordenada ─────────────────────────────
        rank_order    = np.argsort(r2_np)[::-1]
        ranked_r2_np  = r2_np[rank_order]
        q3_all        = float(np.quantile(ranked_r2_np, 0.75))

        # ── 2. Filtro: R² NP > mediana ───────────────────────────────────
        r2_median = float(np.median(r2_np))
        self._log(f"  Mediana R² NP = {r2_median:.3f}")

        mask_above_median = r2_np > r2_median
        genes_above       = all_genes[mask_above_median]
        r2_above          = r2_np[mask_above_median]

        self._log(f"  Genes por encima de mediana: {len(genes_above)}")

        # ── 3. Acrofase Cosinor para cada gen superviviente ───────────────
        cos_model  = CosinorModel()
        cos_peaks  = np.empty(len(genes_above), dtype=np.float64)
        for i, gene in enumerate(genes_above):
            row = expr_norm.loc[gene].values.astype(np.float64)
            cr  = cos_model.fit(row, circ)
            cos_peaks[i] = (-cr.params["phi"]) % (2.0 * pi)

        # ── 4. Asignacion de sectores ────────────────────────────────────
        peak_ref = circular_median(cos_peaks)
        centres  = sector_boundaries(peak_ref)
        sectors  = assign_sectors(cos_peaks, peak_ref)

        self._log(f"  Mediana circular de picos Cosinor: {peak_ref:.3f} rad")
        for s in range(1, 9):
            cnt = int(np.sum(sectors == s))
            self._log(f"    Sector {s}: {cnt} gene(s)")

        # ── 5. Filtrado FMM por sector ────────────────────────────────────
        r2_par_threshold = min(
            float(np.quantile(r2_par_core, 0.25)), self.r2_par_floor
        )
        self._log(f"  Umbral de aceptacion R²_par: {r2_par_threshold:.3f}")

        fmm_model = FMMModel(
            length_alpha_grid=self.fmm_length_alpha_grid,
            length_omega_grid=self.fmm_length_omega_grid,
            num_reps=self.fmm_num_reps,
        )

        # Acumuladores (replicando los patrones rbind/c de R).
        acc_names:      list[str]         = []
        acc_expr:       list[np.ndarray]  = []
        acc_fmm_fit:    list[np.ndarray]  = []
        acc_fmm_par:    list[np.ndarray]  = []  # [M, A, α, β, ω]
        acc_cos_fit:    list[np.ndarray]  = []
        acc_cos_par:    list[np.ndarray]  = []  # [M, A, φ]
        acc_peaks:      list[float]       = []
        acc_sectors:    list[int]         = []
        acc_r2par:      list[float]       = []
        acc_discarded:  list[str]         = []

        total_accepted = 0

        for sector_id in range(1, 9):
            sec_mask    = sectors == sector_id
            sec_genes   = genes_above[sec_mask]
            sec_r2_np   = r2_above[sec_mask]

            if len(sec_genes) == 0:
                continue

            # Ordenar por R² NP descendente.
            order       = np.argsort(sec_r2_np)[::-1]
            sec_genes   = sec_genes[order]
            sec_r2_np   = sec_r2_np[order]

            q1_sector   = float(np.quantile(sec_r2_np, 0.25))
            r2_floor_s  = max(q1_sector, self.r2_par_floor)

            in_sector   = 0
            rec         = 0

            # Condicion del while de R (lineas 1345, 1387, etc.):
            # while (sector_has_genes &
            #        (in_sector < tam | (in_sector >= tam & R2_NP < Q3_all)) &
            #        rec <= len(genes) &
            #        R2_NP[rec] > max(Q1_sector, 0.4))
            while rec < len(sec_genes) and sec_r2_np[rec] > r2_floor_s:
                # Puerta de "continuar mas alla del objetivo".
                if in_sector >= self.tam and sec_r2_np[rec] >= q3_all:
                    break

                gene = sec_genes[rec]
                rec += 1

                # Ajustar FMM.
                row = expr_norm.loc[gene].values.astype(np.float64)
                fr  = fmm_model.fit(row, circ)

                omega  = fr.params["omega"]
                r2_par = fr.r2

                self._log(
                    f"    [{total_accepted+1}] {gene}  "
                    f"R2_NP={sec_r2_np[rec-1]:.3f}  "
                    f"R2_par={r2_par:.3f}  ω={omega:.3f}",
                    end="\r",
                )

                if omega > self.omega_min and r2_par > r2_par_threshold:
                    in_sector      += 1
                    total_accepted += 1

                    # Ajuste Cosinor para este gen.
                    cr = cos_model.fit(row, circ)

                    peak_fmm = fmm_peak_time(
                        fr.params["alpha"], fr.params["beta"], fr.params["omega"]
                    )

                    acc_names.append(gene)
                    acc_expr.append(row)
                    acc_fmm_fit.append(fr.fitted)
                    acc_fmm_par.append([
                        fr.params["M"], fr.params["A"],
                        fr.params["alpha"], fr.params["beta"], fr.params["omega"],
                    ])
                    acc_cos_fit.append(cr.fitted)
                    acc_cos_par.append([
                        cr.params["M"], cr.params["A"], cr.params["phi"],
                    ])
                    acc_peaks.append(peak_fmm)
                    acc_sectors.append(sector_id)
                    acc_r2par.append(r2_par)
                else:
                    acc_discarded.append(gene)

            self._log(
                f"    Sector {sector_id}: accepted {in_sector}, "
                f"examined {rec}           "
            )

        self._log(f"  Total aceptados: {total_accepted}")

        # ── 6. Construir resultado ────────────────────────────────────────
        n_acc = len(acc_names)
        cols  = expr_norm.columns

        result = CandidateResult(
            gene_names      = acc_names,
            expr_data       = pd.DataFrame(
                np.vstack(acc_expr) if n_acc else np.empty((0, len(circ))),
                index=acc_names, columns=cols,
            ),
            fmm_fitted      = pd.DataFrame(
                np.vstack(acc_fmm_fit) if n_acc else np.empty((0, len(circ))),
                index=acc_names, columns=cols,
            ),
            fmm_params      = np.array(acc_fmm_par).reshape(n_acc, 5) if n_acc else np.zeros((0, 5)),
            cos_fitted      = pd.DataFrame(
                np.vstack(acc_cos_fit) if n_acc else np.empty((0, len(circ))),
                index=acc_names, columns=cols,
            ),
            cos_params      = np.array(acc_cos_par).reshape(n_acc, 3) if n_acc else np.zeros((0, 3)),
            fmm_peaks       = np.array(acc_peaks),
            sector_labels   = np.array(acc_sectors, dtype=int),
            r2_par          = np.array(acc_r2par),
            ranked_r2_np    = ranked_r2_np,
            sector_centres  = centres,
            discarded_genes = acc_discarded,
        )

        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Utilidad
    # ------------------------------------------------------------------

    def _log(self, message: str, end: str = "\n") -> None:
        if self.verbose:
            print(message, end=end, flush=True)
