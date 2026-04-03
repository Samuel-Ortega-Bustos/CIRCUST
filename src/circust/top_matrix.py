"""
circust/top_matrix.py
=====================
Etapa 3.7: Construccion de la matriz de expresion TOP.

Combina los 12 genes reloj centrales con los genes ritmicos candidatos
seleccionados en las etapas anteriores, produciendo la matriz final
utilizada por la Etapa 4 (Estimacion Robusta).

Los genes centrales se toman de la matriz normalizada completa (reordenada
segun el orden sincronizado de muestras), y los genes candidatos provienen
de la matriz de candidatos sincronizada. Los genes artificiales/anadidos
se excluyen de la lista de candidatos para evitar duplicacion.

Equivalente en R
-----------------
``obtainTop_v2_cores()`` (lineas 113–138 de ``functionGTEX_cores.R``).

Posicion en el pipeline
-----------------------
    CandidateSelector → ReferenceSetBuilder → RandomSelector →
    **TopMatrixBuilder** → RobustEstimator
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_top_matrix(
    core_genes:         list[str],
    candidate_names:    list[str],
    expr_full_norm:     pd.DataFrame,
    candidate_norm_ord: pd.DataFrame,
    sample_order:       np.ndarray,
    genes_to_exclude:   list[str] | None = None,
) -> pd.DataFrame:
    """
    Construye la matriz TOP de genes apilando los genes centrales sobre
    el conjunto de candidatos.

    Parametros
    ----------
    core_genes : list de str
        Los 12 simbolos de genes reloj centrales (ej. ARNTL, DBP, PER1, …).

    candidate_names : list de str
        Simbolos de genes seleccionados por el pipeline de conjunto de
        referencia / seleccion aleatoria.
        R: ``mName``  → ``refSetRefG[[1]]``.

    expr_full_norm : pd.DataFrame, forma (n_todos_genes, n_muestras)
        Matriz de expresion normalizada completa (genes × muestras).
        R: ``mFullNorm``  → ``preOrdRefG2[[3]]``.

    candidate_norm_ord : pd.DataFrame, forma (n_candidatos, n_muestras)
        Matriz de expresion de candidatos sincronizada y normalizada.
        R: ``mMinCandNormOrd``  → ``refSetRefG[[11]]``.

    sample_order : array int (n_muestras,)
        Indices de columna que reordenan ``expr_full_norm`` en el
        orden sincronizado de muestras.
        R: ``indOrd``  → ``refSetRefG[[12]]``.

    genes_to_exclude : list de str o None
        Genes artificiales/anadidos a eliminar de la lista de candidatos
        antes de apilar (evita duplicados con genes centrales).
        R: ``genAdd``  → ``refSetRefG[[19]]``.

    Devuelve
    --------
    pd.DataFrame, forma (n_core + n_cand_limpios, n_muestras)
        Matriz apilada por filas: genes centrales primero, luego candidatos.
        Nombres de filas = simbolos de genes.
        Equivalente en R: ``top``  → ``topRefG``.
    """
    # ── Eliminar genes excluidos de la lista de candidatos ─────────────
    cand = list(candidate_names)
    if genes_to_exclude:
        for g in genes_to_exclude:
            if g in cand:
                cand.remove(g)

    # ── Eliminar genes centrales que tambien aparecen en candidatos ────
    # R: si todos los genes centrales estan ausentes de mName → apilar simple.
    #    si no → eliminar genes centrales solapados de mName.
    overlap = [g for g in core_genes if g in cand]
    for g in overlap:
        cand.remove(g)

    # ── Filas de genes centrales: de la matriz completa, reordenadas ───
    core_rows = expr_full_norm.loc[
        [g for g in core_genes if g in expr_full_norm.index]
    ].iloc[:, sample_order]
    core_rows.columns = range(core_rows.shape[1])

    # ── Filas de genes candidatos: de la matriz sincronizada ────────────
    cand_present = [g for g in cand if g in candidate_norm_ord.index]
    if len(cand_present) > 0:
        cand_rows = candidate_norm_ord.loc[cand_present].copy()
        cand_rows.columns = range(cand_rows.shape[1])
    else:
        cand_rows = pd.DataFrame(
            dtype=float, columns=range(core_rows.shape[1])
        )

    # ── Apilar: centrales arriba, candidatos abajo ─────────────────────
    top = pd.concat([core_rows, cand_rows], axis=0)
    top.index = list(core_rows.index) + list(cand_rows.index)

    return top
