"""
circust/reference_set.py
========================
Etapa 3.3 / 3.4: Construccion del Conjunto de Referencia.

Refina el conjunto candidato producido por :class:`CandidateSelector`
volviendo a sectorizarlo, esta vez en el marco circular biologico
sincronizado (anclado en ARNTL/DBP). Despues poda cada sector con
``reduce_sector`` para dejar como mucho ~25 genes de alta calidad por
sector. El resultado es el "minimum reference set" que la Etapa 3.5
muestrea aleatoriamente.

Algoritmo
---------
1. Asegurar que ARNTL y DBP estan presentes en la matriz candidata
   (si faltan se anaden desde la matriz normalizada completa).

2. Calcular la acrofase Cosinor de cada gen del conjunto candidato
   en el orden circular preliminar.

3. Filtrar todos los genes de la matriz completa cuyo R²_NP supera
   la mediana global, y calcular sus acrofases Cosinor. La mediana
   circular de estas acrofases define el centro de los 8 sectores
   ``m1 … m8``.

4. Reasignar los candidatos a uno de los 8 sectores definidos en
   torno a esa mediana circular.

5. Aplicar ``reduce_sector`` a cada sector para podar genes de
   bajo R² parametrico.

6. Concatenar los 8 sectores reducidos → conjunto de referencia.

Equivalente en R
-----------------
``computeMinSet_v3_cores()`` (lineas 2357–2632 de ``functionGTEX_cores.R``).
Engloba el trabajo realizado por sus helpers
``reajustarPlotsNewReferences_vInicial_v3`` y
``robustSincro2DBP_v3_cores`` (con ``nReps = 1``), que en la
practica solo recalculan los ajustes Cosinor/FMM ya existentes y los
reordenan en el mismo eje circular preliminar.

Posicion en el pipeline
-----------------------
    CandidateSelector  →  **ReferenceSetBuilder**  →  RandomSelector
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from math import pi

from circust.fitting.cosinor import CosinorModel
from circust.candidate_selection import (
    CandidateResult,
    circular_median,
    sector_boundaries,
    assign_sectors,
    reduce_sector,
)


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class ReferenceSetResult:
    """
    Salida de :class:`ReferenceSetBuilder`.

    Atributos
    ---------
    gene_names : list[str]
        Simbolos de genes del conjunto de referencia (concatenados por
        sector). R: ``namesRedMuscle``  → ``refSetRefG[[1]]``.

    fmm_peaks : np.ndarray, forma (n_ref,)
        Tiempos de pico FMM por gen.
        R: ``peakParRedMuscle``  → ``refSetRefG[[2]]``.

    r2_par : np.ndarray, forma (n_ref,)
        R² parametrico (FMM) por gen.
        R: ``R2RedMuscle``  → ``refSetRefG[[3]]``.

    sector_labels : np.ndarray de int, forma (n_ref,)
        Etiqueta de sector (1 … 8) por gen.
        R: ``sectorBelongingMuscle``  → ``refSetRefG[[4]]``.

    sector_centres : np.ndarray, forma (8,)
        Los 8 centros de sector ``m1 … m8`` (en [0, 2π)).
        R: ``refSetRefG[[5..8]]`` y ``refSetRefG[[14..17]]``.

    candidate_matrix : pd.DataFrame, forma (n_aug_cand, n_muestras)
        Matriz candidata aumentada con ARNTL/DBP, en el orden de muestras
        circular preliminar.
        R: ``mMinCandNormOrd``  → ``refSetRefG[[11]]``.

    sample_order : np.ndarray de int
        Orden de muestras circular preliminar (mismo que el de entrada).
        R: ``indMinCandOrd``  → ``refSetRefG[[12]]``.

    circular_scale : np.ndarray
        Escala circular preliminar (mismo que el de entrada).
        R: ``escMinCandOrd``  → ``refSetRefG[[13]]``.

    cosinor_peaks : np.ndarray, forma (n_aug_cand,)
        Acrofases Cosinor de la matriz candidata aumentada.
        R: ``peakCosMucle``  → ``refSetRefG[[18]]``.

    added_genes : list[str]
        Genes "artificiales" anadidos a la matriz candidata (subconjunto
        de {ARNTL, DBP} que no estaban ya presentes).
        R: ``genesArt``  → ``refSetRefG[[19]]``.
    """

    gene_names:       list
    fmm_peaks:        np.ndarray
    r2_par:           np.ndarray
    sector_labels:    np.ndarray
    sector_centres:   np.ndarray
    candidate_matrix: pd.DataFrame
    sample_order:     np.ndarray
    circular_scale:   np.ndarray
    cosinor_peaks:    np.ndarray
    added_genes:      list           = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.gene_names)
        lines = [
            "=== Resumen del Conjunto de Referencia ===",
            f"  Genes en conjunto referencia : {n}",
            f"  Genes anadidos (ARNTL/DBP)   : {self.added_genes}",
        ]
        for s in range(1, 9):
            cnt = int(np.sum(self.sector_labels == s))
            if cnt > 0:
                lines.append(f"    Sector {s}: {cnt} gene(s)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReferenceSetBuilder
# ---------------------------------------------------------------------------

class ReferenceSetBuilder:
    """
    Construye el conjunto de referencia minimo (Etapa 3.3 / 3.4) a partir
    de la salida de :class:`CandidateSelector`.

    Parametros
    ----------
    arntl_gene : str
        Gen ancla circadiano. Por defecto ``"ARNTL"``.

    dbp_gene : str
        Gen secundario para sincronizacion direccional. Por defecto ``"DBP"``.

    verbose : bool
        Imprimir mensajes de progreso.
    """

    def __init__(
        self,
        arntl_gene: str  = "ARNTL",
        dbp_gene:   str  = "DBP",
        verbose:    bool = True,
    ) -> None:
        self.arntl_gene = arntl_gene
        self.dbp_gene   = dbp_gene
        self.verbose    = verbose

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(
        self,
        candidate:      CandidateResult,
        expr_full_norm: pd.DataFrame,
        circular_scale: np.ndarray,
        sample_order:   np.ndarray,
        r2_np_all:      np.ndarray,
        gene_names_all: list[str],
    ) -> ReferenceSetResult:
        """
        Construye el conjunto de referencia.

        Parametros
        ----------
        candidate : CandidateResult
            Salida de ``CandidateSelector.run()``.

        expr_full_norm : pd.DataFrame, forma (n_genes, n_muestras)
            Matriz de expresion normalizada **completa**, ya en el orden
            circular preliminar.
            R: ``mFullN``  ← ``preOrdRefG2[[3]]``.

        circular_scale : np.ndarray, forma (n_muestras,)
            Escala circular preliminar.
            R: ``escIn``  ← ``preOrdRefG2[[2]]``.

        sample_order : np.ndarray de int, forma (n_muestras,)
            Indices de muestra del ordenamiento preliminar.
            R: ``orIn``  ← ``preOrdRefG2[[1]]``.

        r2_np_all : np.ndarray, forma (n_genes,)
            R² no parametrico para **todos** los genes.
            R: ``R2NPTissue``  ← ``obtainNPRefG[[5]]``.

        gene_names_all : list[str]
            Nombres de genes correspondientes a ``r2_np_all`` y a las
            filas de ``expr_full_norm``.

        Devuelve
        --------
        ReferenceSetResult
        """
        self._log("=== Etapa 3.3 / 3.4: Construyendo Conjunto de Referencia ===")

        circ          = np.asarray(circular_scale, dtype=np.float64)
        r2_np_all_arr = np.asarray(r2_np_all, dtype=np.float64)
        gene_names_all = list(gene_names_all)

        # ── 1. Aumentar la matriz candidata con ARNTL / DBP si faltan ────
        cand_names = list(candidate.gene_names)
        cand_expr  = candidate.expr_data.copy()
        cand_r2par = list(candidate.r2_par)
        cand_peaks = list(candidate.fmm_peaks)

        added_genes: list[str] = []
        for anchor in (self.arntl_gene, self.dbp_gene):
            if anchor in cand_names:
                continue
            if anchor not in expr_full_norm.index:
                self._log(
                    f"  AVISO: gen ancla {anchor} ausente de la matriz "
                    "completa; no se puede anadir."
                )
                continue
            row = expr_full_norm.loc[anchor].values.astype(np.float64)
            cand_expr = pd.concat(
                [pd.DataFrame([row], index=[anchor], columns=cand_expr.columns),
                 cand_expr],
                axis=0,
            )
            cand_names.insert(0, anchor)
            # R² parametrico y peak FMM ya no estan disponibles para los
            # anclas anadidos; se rellenan con un placeholder de alto R²
            # para evitar que ``reduce_sector`` los descarte (los anclas
            # SIEMPRE deben permanecer en el conjunto de referencia).
            cand_r2par.insert(0, 1.0)
            cand_peaks.insert(0, float("nan"))
            added_genes.append(anchor)

        cand_r2par_arr = np.asarray(cand_r2par, dtype=np.float64)
        cand_peaks_arr = np.asarray(cand_peaks, dtype=np.float64)

        self._log(
            f"  Matriz candidata aumentada: {len(cand_names)} genes "
            f"(anadidos: {added_genes or 'ninguno'})"
        )

        # ── 2. Acrofase Cosinor de cada candidato (en orden preliminar) ──
        cos_model  = CosinorModel()
        cos_peaks  = np.empty(len(cand_names), dtype=np.float64)
        for i, gene in enumerate(cand_names):
            row = cand_expr.loc[gene].values.astype(np.float64)
            cr  = cos_model.fit(row, circ)
            cos_peaks[i] = cr.peak_time

        # ── 3. Mediana circular de las acrofases del top global ─────────
        # Filtrar genes de la matriz completa con R² NP > mediana global,
        # luego calcular sus acrofases Cosinor para definir m1.
        r2_median_global = float(np.median(r2_np_all_arr))
        mask_above       = r2_np_all_arr > r2_median_global
        top_genes        = [g for g, m in zip(gene_names_all, mask_above) if m]

        self._log(
            f"  Genes globales con R²_NP > mediana ({r2_median_global:.3f}): "
            f"{len(top_genes)}"
        )

        cos_peaks_top = np.empty(len(top_genes), dtype=np.float64)
        for i, gene in enumerate(top_genes):
            row = expr_full_norm.loc[gene].values.astype(np.float64)
            cr  = cos_model.fit(row, circ)
            cos_peaks_top[i] = cr.peak_time

        m1 = circular_median(cos_peaks_top) if len(cos_peaks_top) > 0 else 0.0
        centres = sector_boundaries(m1)

        self._log(
            f"  Mediana circular global (m1) = {m1:.3f} rad  "
            f"→ centros de sector m1…m8 listos"
        )

        # ── 4. Asignar candidatos a sectores en torno a m1 ───────────────
        sectors = assign_sectors(cos_peaks, m1)

        # ── 5. Podar cada sector con reduce_sector ───────────────────────
        out_names:   list[str]   = []
        out_peaks:   list[float] = []
        out_r2:      list[float] = []
        out_sectors: list[int]   = []

        names_arr = np.asarray(cand_names, dtype=object)

        for sector_id in range(1, 9):
            sec_mask  = sectors == sector_id
            if not sec_mask.any():
                continue

            sec_peaks = cos_peaks[sec_mask]
            sec_r2    = cand_r2par_arr[sec_mask]
            sec_names = names_arr[sec_mask]
            sec_fmm_p = cand_peaks_arr[sec_mask]

            kept_peaks, kept_r2, kept_names = reduce_sector(
                sec_peaks, sec_r2, sec_names
            )

            # Recuperar los picos FMM de los nombres conservados
            kept_fmm_peaks = np.array([
                sec_fmm_p[int(np.where(sec_names == g)[0][0])]
                for g in kept_names
            ])

            out_names.extend(kept_names.tolist())
            out_peaks.extend(kept_fmm_peaks.tolist())
            out_r2.extend(kept_r2.tolist())
            out_sectors.extend([sector_id] * len(kept_names))

            self._log(
                f"    Sector {sector_id}: {len(sec_names)} → {len(kept_names)} "
                "tras reduce_sector"
            )

        result = ReferenceSetResult(
            gene_names       = out_names,
            fmm_peaks        = np.asarray(out_peaks, dtype=np.float64),
            r2_par           = np.asarray(out_r2,    dtype=np.float64),
            sector_labels    = np.asarray(out_sectors, dtype=int),
            sector_centres   = centres,
            candidate_matrix = cand_expr,
            sample_order     = np.asarray(sample_order, dtype=int),
            circular_scale   = circ,
            cosinor_peaks    = cos_peaks,
            added_genes      = added_genes,
        )

        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Utilidad
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
