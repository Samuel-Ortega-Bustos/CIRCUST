"""
circust/top_genes.py
====================
Etapa 3: Determinacion de genes TOP ritmicos especificos del tejido.

Implementa la seleccion de genes TOP segun la Seccion 3 de CIRCUST:

  1. Ajuste ORI: para cada gen, ajustar regresion isotonica circular
     unimodal (PAVA) y calcular R2_ORI = 1 - MSE_unimodal / MSE_plano.

  2. Filtro ORI: descartar genes con R2_ORI < 0.5 (ritmicidad no
     parametrica insuficiente).

  3. Filtro FMM: ajustar el modelo FMM a los supervivientes y
     seleccionar aquellos que sean:
       i)   no puntiagudos (omega > 0.1)
       ii)  altamente ritmicos (R2_FMM > 0.5)
       iii) picos (t_U) distribuidos en los 4 cuadrantes de [0, 2pi)

  4. Genes semilla forzados: los 12 genes reloj centrales se
     incluyen siempre; si alguno no paso los filtros, se fuerza su
     inclusion.

Posicion en el pipeline
-----------------------
    CircularSynchronizer  ->  **TopGeneSelector**  ->  RandomSelector
                                                   ->  RobustEstimator
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from math import pi

from circust.fitting.ori import ORIModel
from circust.fitting.fmm import FMMModel
from circust.fitting.cosinor import CosinorModel


# ═══════════════════════════════════════════════════════════════════════════
# Funciones auxiliares de estadistica circular
# ═══════════════════════════════════════════════════════════════════════════

def circular_median(angles: np.ndarray) -> float:
    """
    Mediana circular: el angulo theta que minimiza  sum d(theta, theta_i)
    donde d es la distancia geodesica en el circulo unitario.

    Parametros
    ----------
    angles : array (n,) en [0, 2pi).

    Devuelve
    --------
    float — mediana circular en [0, 2pi).
    """
    angles = np.asarray(angles, dtype=np.float64) % (2.0 * pi)
    n = len(angles)
    if n == 0:
        return 0.0
    if n == 1:
        return float(angles[0])

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

    Los sectores se numeran 1 ... n_sectors.

    Parametros
    ----------
    peaks     : array (n,) de angulos en [0, 2pi).
    reference : float — mediana circular usada como origen.
    n_sectors : int — por defecto 8.

    Devuelve
    --------
    array int (n,) — etiquetas de sector en {1, 2, ..., n_sectors}.
    """
    peaks = np.asarray(peaks, dtype=np.float64)
    rotated = (peaks - reference) % (2.0 * pi)
    sector_width = 2.0 * pi / n_sectors
    sectors = n_sectors - np.floor(rotated / sector_width).astype(int)
    sectors = np.clip(sectors, 1, n_sectors)
    return sectors


def sector_boundaries(reference: float, n_sectors: int = 8) -> np.ndarray:
    """
    Devuelve los *n_sectors* puntos centrales de sector.

    Parametros
    ----------
    reference : float — mediana circular usada como origen.
    n_sectors : int — por defecto 8.

    Devuelve
    --------
    array (n_sectors,) — centros de cada sector en [0, 2pi).
    """
    return np.array([
        (reference - k * pi / (n_sectors / 2)) % (2.0 * pi)
        for k in range(n_sectors)
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Workers a nivel de modulo (serializables por joblib)
# ═══════════════════════════════════════════════════════════════════════════

def _fit_ori_one(
    row: np.ndarray,
    circ: np.ndarray,
) -> float:
    """
    Ajusta ORI a un unico gen. Devuelve R2_ORI.

    Funcion pura a nivel de modulo para que ``joblib`` pueda serializarla
    hacia procesos worker.
    """
    fr = ORIModel().fit(row, circ)
    return fr.r2


def _fit_fmm_one(
    row: np.ndarray,
    circ: np.ndarray,
    alpha_grid: int,
    omega_grid: int,
    num_reps: int,
) -> tuple[float, float, float, np.ndarray]:
    """
    Ajusta FMM a un unico gen. Devuelve (peak_time, omega, r2, fitted).

    Funcion pura a nivel de modulo para que ``joblib`` pueda serializarla
    hacia procesos worker.
    """
    model = FMMModel(
        length_alpha_grid=alpha_grid,
        length_omega_grid=omega_grid,
        num_reps=num_reps,
    )
    fr = model.fit(row, circ)
    return fr.peak_time, fr.params["omega"], fr.r2, fr.fitted


# ═══════════════════════════════════════════════════════════════════════════
# Dataclass de resultado
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TopGeneResult:
    """
    Salida de :class:`TopGeneSelector`.

    Los campos principales son compatibles por duck-typing con
    ``ReferenceSetResult``, de modo que ``RandomSelector`` puede
    consumir este resultado directamente.

    Atributos — pipeline
    --------------------
    gene_names : list[str]
        Simbolos de los genes TOP seleccionados.

    fmm_peaks : np.ndarray (n_top,)
        Tiempos de pico FMM (t_U) por gen.

    r2_par : np.ndarray (n_top,)
        R2 parametrico (FMM) por gen.

    sector_labels : np.ndarray int (n_top,)
        Cuadrante asignado (1–4) por gen.

    sector_centres : np.ndarray (n_sectors,)
        Centros de los cuadrantes en [0, 2pi).

    candidate_matrix : pd.DataFrame (n_top, n_muestras)
        Matriz de expresion de los genes TOP (en el orden circular
        preliminar). Equivale a [X_TOP] del paper.

    cosinor_peaks : np.ndarray (n_top,)
        Acrofases Cosinor de cada gen TOP.

    circular_scale : np.ndarray (n_muestras,)
        Eje temporal circular del ordenamiento preliminar.

    added_genes : list[str]
        Genes semilla que fueron forzados (no pasaron los filtros
        por si solos). Se excluyen del pool de muestreo aleatorio
        en ``RandomSelector``.

    sample_order : np.ndarray int (n_muestras,)
        Orden de muestras del sincronizado preliminar.

    Atributos — diagnostico
    -----------------------
    omega : np.ndarray (n_top,)
        Parametro omega de FMM por gen.

    r2_ori : np.ndarray (n_top,)
        R2 ORI (no parametrico) de cada gen TOP.

    n_total_genes : int
        Numero total de genes antes de cualquier filtro.

    n_after_ori : int
        Genes que superaron el filtro R2_ORI >= umbral.

    n_after_fmm : int
        Genes que superaron los filtros FMM (omega + R2_FMM)
        *antes* de forzar genes semilla.
    """

    # Pipeline (compatibles con ReferenceSetResult por duck-typing)
    gene_names:       list          = field(default_factory=list)
    fmm_peaks:        np.ndarray    = field(default_factory=lambda: np.array([]))
    r2_par:           np.ndarray    = field(default_factory=lambda: np.array([]))
    sector_labels:    np.ndarray    = field(default_factory=lambda: np.array([], dtype=int))
    sector_centres:   np.ndarray    = field(default_factory=lambda: np.zeros(4))
    candidate_matrix: pd.DataFrame  = field(default_factory=pd.DataFrame)
    cosinor_peaks:    np.ndarray    = field(default_factory=lambda: np.array([]))
    circular_scale:   np.ndarray    = field(default_factory=lambda: np.array([]))
    added_genes:      list          = field(default_factory=list)
    sample_order:     np.ndarray    = field(default_factory=lambda: np.array([], dtype=int))

    # Diagnostico
    omega:            np.ndarray    = field(default_factory=lambda: np.array([]))
    r2_ori:           np.ndarray    = field(default_factory=lambda: np.array([]))
    n_total_genes:    int           = 0
    n_after_ori:      int           = 0
    n_after_fmm:      int           = 0

    def summary(self) -> str:
        n = len(self.gene_names)
        n_sec = len(self.sector_centres)
        n_covered = len(np.unique(self.sector_labels)) if n > 0 else 0
        lines = [
            "=== Resumen de Genes TOP ===",
            f"  Genes totales            : {self.n_total_genes}",
            f"  Tras filtro R2_ORI>=0.5  : {self.n_after_ori}",
            f"  Tras filtros FMM         : {self.n_after_fmm}",
            f"  Genes semilla forzados   : {self.added_genes or 'ninguno'}",
            f"  Genes TOP finales        : {n}",
            f"  Cuadrantes cubiertos     : {n_covered} / {n_sec}",
        ]
        if n > 0:
            for s in sorted(np.unique(self.sector_labels)):
                cnt = int(np.sum(self.sector_labels == s))
                lines.append(f"    Cuadrante {s}: {cnt} gen(es)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# TopGeneSelector
# ═══════════════════════════════════════════════════════════════════════════

class TopGeneSelector:
    """
    Selecciona los genes TOP ritmicos especificos del tejido.

    Algoritmo (Seccion 3 del paper CIRCUST)
    ----------------------------------------
    1. Ajustar ORI a cada gen y calcular R2_ORI.
    2. Descartar genes con R2_ORI < ``r2_ori_threshold`` (0.5).
    3. Ajustar FMM a los supervivientes.
    4. Filtrar: omega > ``omega_min`` (0.1) Y R2_FMM > ``r2_fmm_threshold`` (0.5).
    5. Asignar picos FMM a ``n_sectors`` cuadrantes y verificar cobertura.
    6. Forzar inclusion de los genes core si no estan presentes.
    7. Construir la matriz TOP ([X_TOP] en el paper).

    Parametros
    ----------
    r2_ori_threshold : float
        Umbral minimo de R2 ORI para el pre-filtro no parametrico.
        Paper: 0.5.

    r2_fmm_threshold : float
        Umbral minimo de R2 FMM para la ritmicidad parametrica.
        Paper: 0.5.

    omega_min : float
        omega minimo de FMM; rechaza ajustes puntiagudos.
        Paper: 0.1.

    n_sectors : int
        Numero de sectores circulares para verificar cobertura de picos.
        Paper: 4 (cuadrantes).

    fmm_length_alpha_grid, fmm_length_omega_grid, fmm_num_reps : int
        Parametros de la rejilla de ajuste FMM.

    n_jobs : int
        Numero de procesos para el ajuste en paralelo (ORI y FMM).
        ``-1`` usa todos los nucleos disponibles.

    verbose : bool
        Imprimir mensajes de progreso.
    """

    def __init__(
        self,
        r2_ori_threshold:      float = 0.5,
        r2_fmm_threshold:      float = 0.5,
        omega_min:             float = 0.1,
        n_sectors:             int   = 4,
        fmm_length_alpha_grid: int   = 48,
        fmm_length_omega_grid: int   = 24,
        fmm_num_reps:          int   = 3,
        n_jobs:                int   = -1,
        verbose:               bool  = True,
    ) -> None:
        self.r2_ori_threshold      = r2_ori_threshold
        self.r2_fmm_threshold      = r2_fmm_threshold
        self.omega_min             = omega_min
        self.n_sectors             = n_sectors
        self.fmm_length_alpha_grid = fmm_length_alpha_grid
        self.fmm_length_omega_grid = fmm_length_omega_grid
        self.fmm_num_reps          = fmm_num_reps
        self.n_jobs                = n_jobs
        self.verbose               = verbose

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(
        self,
        expr_norm:       pd.DataFrame,
        circular_scale:  np.ndarray,
        seed_genes:      list[str],
        sample_order:    np.ndarray | None = None,
    ) -> TopGeneResult:
        """
        Selecciona los genes TOP.

        Internamente ajusta ORI (regresion isotonica circular unimodal)
        a cada gen para calcular R2_ORI, aplica los filtros y construye
        el conjunto TOP.

        Parametros
        ----------
        expr_norm : pd.DataFrame (n_genes, n_muestras)
            Matriz de expresion normalizada completa, ya ordenada
            por el tiempo circular preliminar.

        circular_scale : array (n_muestras,)
            Eje temporal circular del ordenamiento preliminar.

        seed_genes : list[str]
            Genes core del pipeline (los mismos seleccionados por
            ``CoreGeneSelector``). Se fuerzan en el conjunto TOP
            si no superan los filtros.

        sample_order : array int (n_muestras,) o None
            Indices de muestra del ordenamiento preliminar.
            Si es None se asume identidad.

        Devuelve
        --------
        TopGeneResult
        """
        seed_genes = list(seed_genes)
        self._log("=== Etapa 3: Seleccion de Genes TOP ===")

        gene_names = list(expr_norm.index)
        circ       = np.asarray(circular_scale, dtype=np.float64)
        n_total    = len(gene_names)
        values     = expr_norm.values.astype(np.float64)

        if sample_order is None:
            sample_order = np.arange(expr_norm.shape[1], dtype=int)

        # ── 1. Ajuste ORI (regresion isotonica circular unimodal) ──────
        self._log(f"  Genes totales: {n_total}")
        self._log("  Ajustando ORI (PAVA unimodal) ...")

        r2_ori = self._fit_ori_batch(values, circ)

        # ── 2. Filtro R2_ORI ────────────────────────────────────────────
        self._log(
            f"  Filtro R2_ORI >= {self.r2_ori_threshold} ..."
        )

        mask_ori     = r2_ori >= self.r2_ori_threshold
        surv_genes   = [g for g, m in zip(gene_names, mask_ori) if m]
        surv_r2_ori  = r2_ori[mask_ori]
        n_after_ori  = len(surv_genes)

        self._log(f"  Supervivientes R2_ORI: {n_after_ori}")

        # ── 3. Ajustar FMM a los supervivientes (paralelo) ────────────
        self._log("  Ajustando FMM ...")

        surv_rows = np.array(
            [expr_norm.loc[g].values.astype(np.float64) for g in surv_genes]
        )

        fmm_results = self._fit_fmm_batch(surv_rows, circ)

        fmm_peaks_all = np.array([r[0] for r in fmm_results])
        fmm_omega_all = np.array([r[1] for r in fmm_results])
        fmm_r2_all    = np.array([r[2] for r in fmm_results])

        # ── 4. Filtro FMM: omega > min Y R2_FMM > umbral ──────────────
        mask_fmm = (
            (fmm_omega_all > self.omega_min)
            & (fmm_r2_all > self.r2_fmm_threshold)
        )
        n_after_fmm = int(mask_fmm.sum())

        self._log(
            f"  Filtro FMM (omega>{self.omega_min}, "
            f"R2>{self.r2_fmm_threshold}): {n_after_fmm} supervivientes"
        )

        # Acumular genes aceptados
        acc_names:   list[str]        = []
        acc_r2_ori:  list[float]      = []
        acc_r2_fmm:  list[float]      = []
        acc_omega:   list[float]      = []
        acc_peaks:   list[float]      = []
        acc_rows:    list[np.ndarray] = []

        for i, gene in enumerate(surv_genes):
            if mask_fmm[i]:
                acc_names.append(gene)
                acc_r2_ori.append(float(surv_r2_ori[i]))
                acc_r2_fmm.append(float(fmm_r2_all[i]))
                acc_omega.append(float(fmm_omega_all[i]))
                acc_peaks.append(float(fmm_peaks_all[i]))
                acc_rows.append(surv_rows[i])

        # ── 5. Forzar genes core (semilla) ────────────────────────────
        forced: list[str] = []
        accepted_set = set(acc_names)

        for seed in seed_genes:
            if seed in accepted_set:
                continue
            if seed not in expr_norm.index:
                self._log(
                    f"  AVISO: gen semilla {seed} ausente de la matriz."
                )
                continue

            row = expr_norm.loc[seed].values.astype(np.float64)
            pk, om, r2, _ = _fit_fmm_one(
                row, circ,
                self.fmm_length_alpha_grid,
                self.fmm_length_omega_grid,
                self.fmm_num_reps,
            )
            idx_all = (
                gene_names.index(seed) if seed in gene_names else -1
            )
            seed_r2_ori = float(r2_ori[idx_all]) if idx_all >= 0 else 0.0

            acc_names.append(seed)
            acc_r2_ori.append(seed_r2_ori)
            acc_r2_fmm.append(float(r2))
            acc_omega.append(float(om))
            acc_peaks.append(float(pk))
            acc_rows.append(row)
            forced.append(seed)

        if forced:
            self._log(f"  Genes semilla forzados: {forced}")

        n_top = len(acc_names)
        self._log(f"  Genes TOP finales: {n_top}")

        # ── 6. Acrofases Cosinor (para cuadrantes y RandomSelector) ────
        cos_model = CosinorModel()
        cos_peaks = np.empty(n_top, dtype=np.float64)
        for i in range(n_top):
            cr = cos_model.fit(acc_rows[i], circ)
            cos_peaks[i] = cr.peak_time

        # ── 7. Asignar cuadrantes ──────────────────────────────────────
        peaks_arr = np.array(acc_peaks)

        if n_top > 0:
            peak_ref = circular_median(peaks_arr)
            centres  = sector_boundaries(peak_ref, self.n_sectors)
            sectors  = assign_sectors(
                peaks_arr, peak_ref, self.n_sectors,
            )
        else:
            centres = np.zeros(self.n_sectors)
            sectors = np.array([], dtype=int)

        covered = set(sectors.tolist()) if n_top > 0 else set()
        expected = set(range(1, self.n_sectors + 1))
        missing = expected - covered
        if missing:
            self._log(
                f"  AVISO: cuadrantes sin cobertura: {sorted(missing)}"
            )

        # ── 8. Construir matriz TOP ────────────────────────────────────
        if n_top > 0:
            top_matrix = pd.DataFrame(
                np.vstack(acc_rows),
                index=acc_names,
                columns=expr_norm.columns,
            )
        else:
            top_matrix = pd.DataFrame(
                dtype=float, columns=expr_norm.columns,
            )

        result = TopGeneResult(
            gene_names       = acc_names,
            fmm_peaks        = peaks_arr,
            r2_par           = np.array(acc_r2_fmm),
            sector_labels    = sectors,
            sector_centres   = centres,
            candidate_matrix = top_matrix,
            cosinor_peaks    = cos_peaks,
            circular_scale   = circ,
            added_genes      = forced,
            sample_order     = np.asarray(sample_order, dtype=int),
            omega            = np.array(acc_omega),
            r2_ori           = np.array(acc_r2_ori),
            n_total_genes    = n_total,
            n_after_ori      = n_after_ori,
            n_after_fmm      = n_after_fmm,
        )

        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Ajuste ORI en lote (paralelo)
    # ------------------------------------------------------------------

    def _fit_ori_batch(
        self,
        rows: np.ndarray,
        circ: np.ndarray,
    ) -> np.ndarray:
        """Ajusta ORI a multiples genes en paralelo. Devuelve R2_ORI (n_genes,)."""
        n = rows.shape[0]

        if self.n_jobs == 1 or n <= 8:
            r2_vals = [
                _fit_ori_one(rows[i], circ) for i in range(n)
            ]
        else:
            from joblib import Parallel, delayed
            self._log(
                f"  Paralelizando {n} ajustes ORI en {self.n_jobs} jobs ..."
            )
            r2_vals = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_ori_one)(rows[i], circ) for i in range(n)
            )

        return np.array(r2_vals, dtype=np.float64)

    # ------------------------------------------------------------------
    # Ajuste FMM en lote (paralelo)
    # ------------------------------------------------------------------

    def _fit_fmm_batch(
        self,
        rows: np.ndarray,
        circ: np.ndarray,
    ) -> list[tuple[float, float, float, np.ndarray]]:
        """Ajusta FMM a multiples genes en paralelo con joblib."""
        n  = rows.shape[0]
        ag = self.fmm_length_alpha_grid
        og = self.fmm_length_omega_grid
        nr = self.fmm_num_reps

        if self.n_jobs == 1 or n <= 8:
            return [
                _fit_fmm_one(rows[i], circ, ag, og, nr)
                for i in range(n)
            ]

        from joblib import Parallel, delayed
        self._log(
            f"  Paralelizando {n} ajustes FMM en {self.n_jobs} jobs ..."
        )
        return Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_fmm_one)(rows[i], circ, ag, og, nr)
            for i in range(n)
        )

    # ------------------------------------------------------------------
    # Utilidad
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)
