"""
circust/random_selection.py
===========================
Etapa 3.5 / 3.6: Seleccion aleatoria controlada de subconjuntos del
conjunto de referencia.

Dado el conjunto de referencia minimo producido por
:class:`ReferenceSetBuilder`, esta etapa extrae K subconjuntos
aleatorios sin reemplazo de tamano fijo. Cada extraccion debe
satisfacer una serie de restricciones de cobertura circular y de
calidad numerica para evitar configuraciones degeneradas:

  1. **Cobertura por cuadrantes**: cada extraccion debe representar
     al menos 3 de los 4 cuadrantes circulares (sectores agrupados
     de a 2). Si solo hay 2 cuadrantes presentes en el conjunto de
     referencia, basta con cubrir 2.

  2. **Calidad parametrica**: el R² parametrico minimo de la
     extraccion debe superar 0.5. Los anclas ARNTL y DBP estan
     exentos de este criterio.

  3. **Sin colapsos por cuadrante**: dentro de cada cuadrante las
     distancias circulares ``1 − cos(θᵢ − θⱼ)`` entre picos cosinor
     consecutivos no deben ser practicamente cero (se tolera hasta
     un 10 % de pares colisionando).

  4. **Cuasi-equiespaciado**: si la escala circular original es
     fuertemente no equiespaciada (existe un hueco mayor que
     1.5 · IQR), se ejecuta un mini-CPCA sobre el subconjunto y se
     rechaza la extraccion si el hueco maximo en su nueva escala
     supera al hueco maximo original.

Si tras 5000 intentos no se consiguen K extracciones validas, se
quedan las K mejores por la metrica del hueco maximo del mini-CPCA.

Equivalente en R
-----------------
``compRandomSel_v3()`` (lineas 364–548 de ``functionGTEX_cores.R``).

Posicion en el pipeline
-----------------------
    ReferenceSetBuilder  →  **RandomSelector**  →  RobustEstimator
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from math import pi

from circust.reference_set import ReferenceSetResult


# ---------------------------------------------------------------------------
# Mini-CPCA local (sin acoplamiento a la clase CPCA completa)
# ---------------------------------------------------------------------------

def _mini_cpca_phi(sub: np.ndarray) -> np.ndarray:
    """
    Calcula la escala circular phi para una submatriz de genes.

    Replica los pasos esenciales de :class:`CPCA._run` (centrar por filas,
    escalar por columnas, SVD, proyectar sobre el circulo unitario).

    Parametros
    ----------
    sub : np.ndarray, forma (n_genes, n_muestras)

    Devuelve
    --------
    np.ndarray (n_muestras,) — phi ordenado ascendente en [0, 2π).
    """
    n_genes = sub.shape[0]
    if n_genes < 2:
        return np.linspace(0, 2 * pi, sub.shape[1], endpoint=False)

    centred = sub - sub.mean(axis=1, keepdims=True)
    col_rms = np.sqrt(np.sum(centred ** 2, axis=0) / max(n_genes - 1, 1))
    col_rms[col_rms == 0] = 1.0
    scaled  = centred / col_rms

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
    """
    Mayor hueco entre muestras consecutivas (con cierre circular).

    El eje circular se reescala a unidades de "muestras equivalentes"
    (R: ``timEuc = esc / (2π) * length(esc)``).
    """
    n   = len(circ_sorted)
    if n < 2:
        return 0.0
    t   = circ_sorted / (2.0 * pi) * n
    gaps        = np.diff(t)
    wrap_gap    = n - t[-1] + t[0]
    return float(max(gaps.max(), wrap_gap))


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class RandomSelectionResult:
    """
    Salida de :class:`RandomSelector`.

    Atributos
    ---------
    selection_indices : np.ndarray int, forma (K, tam)
        Indices base 0 dentro del conjunto de referencia para cada
        extraccion. R: ``mSel``  → ``selRefG[[1]]``.

    selection_names : np.ndarray object, forma (K, tam)
        Nombres de gen por extraccion. R: ``mName``  → ``selRefG[[2]]``.

    selection_fmm_peaks : np.ndarray float, forma (K, tam)
        Picos FMM por extraccion. R: ``mPeak``  → ``selRefG[[3]]``.

    selection_r2 : np.ndarray float, forma (K, tam)
        R² parametrico por extraccion. R: ``mR2``  → ``selRefG[[4]]``.

    selection_cos_peaks : np.ndarray float, forma (K, tam)
        Picos Cosinor por extraccion. R: ``mCosPeak``  → ``selRefG[[5]]``.

    only_two_quadrants : bool
        True si el conjunto de referencia solo cubria 2 cuadrantes
        (la regla de cobertura se relaja en consecuencia).
        R: ``dosSect``  → ``selRefG[[6]]``.

    n_attempts : int
        Numero total de extracciones aleatorias realizadas hasta
        completar las K validas (o agotar el limite).

    fallback_used : bool
        True si se agoto el limite de 5000 intentos y se completaron
        las extracciones con las mejores por la metrica del hueco
        maximo.
    """

    selection_indices:    np.ndarray
    selection_names:      np.ndarray
    selection_fmm_peaks:  np.ndarray
    selection_r2:         np.ndarray
    selection_cos_peaks:  np.ndarray
    only_two_quadrants:   bool
    n_attempts:           int
    fallback_used:        bool          = False

    def summary(self) -> str:
        K, tam = self.selection_indices.shape
        lines = [
            "=== Resumen de Seleccion Aleatoria ===",
            f"  Numero de extracciones K  : {K}",
            f"  Tamano por extraccion     : {tam}",
            f"  Cuadrantes cubiertos      : {2 if self.only_two_quadrants else '≥3'}",
            f"  Intentos totales          : {self.n_attempts}",
            f"  Fallback activado         : {self.fallback_used}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RandomSelector
# ---------------------------------------------------------------------------

class RandomSelector:
    """
    Etapa 3.5 / 3.6 de CIRCUST: extraccion aleatoria controlada.

    Parametros
    ----------
    n_reps : int
        Numero K de extracciones aleatorias requeridas. Por defecto en
        R: 5.

    sample_size_fraction : float
        Tamano de cada extraccion como fraccion del conjunto de
        referencia. Por defecto en R: 2/3.

    r2_min : float
        R² parametrico minimo permitido en la extraccion (con
        excepcion de ARNTL/DBP). Por defecto: 0.5.

    max_attempts : int
        Limite de intentos antes de activar el fallback "mejores por
        hueco maximo". Por defecto en R: 5000.

    arntl_gene, dbp_gene : str
        Anclas exentas del umbral de R².

    seed : int o None
        Semilla del generador aleatorio para reproducibilidad.

    verbose : bool
        Imprimir mensajes de progreso.
    """

    def __init__(
        self,
        n_reps:               int   = 5,
        sample_size_fraction: float = 2.0 / 3.0,
        r2_min:               float = 0.5,
        max_attempts:         int   = 5000,
        arntl_gene:           str   = "ARNTL",
        dbp_gene:             str   = "DBP",
        seed:                 int | None = None,
        verbose:              bool  = True,
    ) -> None:
        self.n_reps               = n_reps
        self.sample_size_fraction = sample_size_fraction
        self.r2_min               = r2_min
        self.max_attempts         = max_attempts
        self.arntl_gene           = arntl_gene
        self.dbp_gene             = dbp_gene
        self.seed                 = seed
        self.verbose              = verbose

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(
        self,
        reference: ReferenceSetResult,
        expr_full_norm: pd.DataFrame,
    ) -> RandomSelectionResult:
        """
        Genera las K extracciones aleatorias.

        Parametros
        ----------
        reference : ReferenceSetResult
            Salida de ``ReferenceSetBuilder.run()``.

        expr_full_norm : pd.DataFrame
            Matriz normalizada completa en el orden circular preliminar.
            Necesaria para el mini-CPCA del criterio (4).

        Devuelve
        --------
        RandomSelectionResult
        """
        self._log("=== Etapa 3.5 / 3.6: Seleccion Aleatoria Controlada ===")

        rng = np.random.default_rng(self.seed)

        # ── Filtrar anclas anadidos del pool de muestreo ─────────────────
        names    = np.asarray(reference.gene_names, dtype=object)
        peaks    = np.asarray(reference.fmm_peaks,  dtype=np.float64)
        r2       = np.asarray(reference.r2_par,     dtype=np.float64)
        sectors8 = np.asarray(reference.sector_labels, dtype=int)
        cos_pk   = np.asarray(reference.cosinor_peaks, dtype=np.float64)

        # cosinor_peaks es de la matriz aumentada (incluye ARNTL/DBP) y
        # esta en el mismo orden que reference.candidate_matrix.index.
        # Necesitamos los picos cosinor *de los genes en gene_names*.
        cand_matrix_index = list(reference.candidate_matrix.index)
        cos_pk_by_name = {
            cand_matrix_index[i]: cos_pk[i]
            for i in range(len(cand_matrix_index))
        }
        cos_peaks_ref = np.array(
            [cos_pk_by_name.get(g, 0.0) for g in names],
            dtype=np.float64,
        )

        # Eliminar anclas anadidos del pool (no del nombre del set)
        remove_mask = np.zeros(len(names), dtype=bool)
        for anchor in reference.added_genes:
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

        self._log(
            f"  Pool de muestreo: {n_pool} genes  |  tam = {tam}  "
            f"|  K = {self.n_reps}"
        )

        # Mapeo de sectores 1..8 → cuadrantes 1..4
        quadrants = ((sectors8_pool - 1) // 2) + 1
        unique_q  = np.unique(quadrants)
        only_two  = len(unique_q) <= 2
        min_q     = 2 if only_two else 3

        # Hueco maximo en la escala circular original (criterio 4)
        circ_pre = np.sort(np.asarray(reference.circular_scale, dtype=np.float64))
        max_gap_orig = _max_consecutive_gap(circ_pre)
        gaps_pre = np.diff(circ_pre)
        iqr_pre  = float(np.subtract(*np.percentile(gaps_pre, [75, 25]))) if len(gaps_pre) else 0.0
        require_cpca_check = max_gap_orig > 1.5 * iqr_pre and iqr_pre > 0
        self._log(
            f"  Hueco max original = {max_gap_orig:.3f}  "
            f"(IQR={iqr_pre:.3f})  "
            f"check CPCA = {require_cpca_check}"
        )

        # Lookup de filas de la matriz completa para el mini-CPCA
        full_index = expr_full_norm.index

        # Acumuladores
        sel_idx     = np.zeros((self.n_reps, tam), dtype=int)
        sel_names   = np.empty((self.n_reps, tam), dtype=object)
        sel_peaks   = np.zeros((self.n_reps, tam), dtype=np.float64)
        sel_cos     = np.zeros((self.n_reps, tam), dtype=np.float64)
        sel_r2      = np.zeros((self.n_reps, tam), dtype=np.float64)

        store_sel:    list[np.ndarray] = []
        store_metric: list[float]      = []

        k = 0
        attempts = 0
        fallback = False

        while k < self.n_reps:
            attempts += 1
            sel = rng.choice(n_pool, size=tam, replace=False)

            # ── Criterio 1: cobertura de cuadrantes ──────────────────────
            sel_quads = np.unique(quadrants[sel])
            if len(sel_quads) < min_q:
                if attempts >= self.max_attempts:
                    fallback = True
                    break
                continue

            # ── Criterio 2: R² minimo ────────────────────────────────────
            r2_sel    = r2_pool[sel]
            min_idx   = int(np.argmin(r2_sel))
            high_r2   = r2_sel[min_idx] > self.r2_min
            if not high_r2:
                worst = names_pool[sel[min_idx]]
                if worst in (self.arntl_gene, self.dbp_gene):
                    high_r2 = True
            if not high_r2:
                if attempts >= self.max_attempts:
                    fallback = True
                    break
                continue

            # ── Criterio 3: distancias intra-cuadrante ───────────────────
            cos_sel    = cos_pool[sel]
            order_cos  = np.argsort(cos_sel)
            cos_sorted = cos_sel[order_cos]
            quad_sorted = quadrants[sel][order_cos]

            # 1 − cos(diff) entre vecinos consecutivos (con cierre circular)
            diffs = np.empty(len(sel))
            diffs[:-1] = 1.0 - np.cos(cos_sorted[:-1] - cos_sorted[1:])
            diffs[-1]  = 1.0 - np.cos(cos_sorted[-1]  - cos_sorted[0])

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

            # ── Criterio 4: chequeo CPCA del subconjunto (opcional) ──────
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

            # ── Aceptar extraccion ───────────────────────────────────────
            sel_idx[k]   = sel
            sel_names[k] = names_pool[sel]
            sel_peaks[k] = peaks_pool[sel]
            sel_cos[k]   = cos_pool[sel]
            sel_r2[k]    = r2_pool[sel]
            k += 1

            self._log(
                f"  Extraccion {k}/{self.n_reps} aceptada "
                f"tras {attempts} intento(s)"
            )

        # ── Fallback: completar con las mejores por hueco max ────────────
        if fallback and k < self.n_reps:
            self._log(
                f"  Fallback activado: {self.n_reps - k} extracciones "
                "completadas con las mejores por hueco maximo CPCA."
            )
            order_metric = np.argsort(store_metric)
            for j, idx_sorted in enumerate(order_metric[: self.n_reps - k]):
                sel = store_sel[idx_sorted]
                pos = k + j
                sel_idx[pos]   = sel
                sel_names[pos] = names_pool[sel]
                sel_peaks[pos] = peaks_pool[sel]
                sel_cos[pos]   = cos_pool[sel]
                sel_r2[pos]    = r2_pool[sel]

        result = RandomSelectionResult(
            selection_indices    = sel_idx,
            selection_names      = sel_names,
            selection_fmm_peaks  = sel_peaks,
            selection_r2         = sel_r2,
            selection_cos_peaks  = sel_cos,
            only_two_quadrants   = only_two,
            n_attempts           = attempts,
            fallback_used        = fallback,
        )
        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Utilidad
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
