"""
circust/robust_order.py
========================
Etapa 4: Estimacion robusta del orden circular mediante K repeticiones
aleatorias y seleccion/agregacion del orden final.

  1. Genera K subconjuntos aleatorios de genes TOP (tamaño configurable,
     por defecto 2/3 del total) con restricciones de cobertura circular
     y calidad parametrica.

  2. Para cada subconjunto k=1..K ejecuta:
        a) CPCA sobre la submatriz de genes seleccionada.
        b) Sincronizacion biologica (rotacion + orientacion).
        c) Re-ajuste FMM / Cosinor / NP a cada gen del TOP para obtener
           la tabla de estadisticos de 25 columnas.

  3. Selecciona el orden final mediante uno de dos metodos:
        - ``best_k``: elige la repeticion k* que maximiza la mediana de
          R² (FMM) sobre un conjunto de evaluacion (genes core o genes
          de Ruben et al. 2018).
        - ``aggregate``: agregacion circular de rangos segun Barragan
          et al. (2021) — combina enfoques TSP y Hodge para fusionar
          los K ordenes en un consenso.

Referencia
----------
Barragan, S., Fernandez, M.A., Rueda, C. (2021). Circular rank
aggregation: pairwise (TSP) and triplewise (Hodge) approaches.
*Journal of Statistical Planning and Inference*, 213, 56-75.

Posicion en el pipeline
-----------------------
    TopGeneSelector  →  **RobustOrderEstimator**  →  (Etapa 5)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from math import pi
from types import SimpleNamespace

from circust.cpca import CPCA
from circust.fitting.fmm import FMMModel
from circust.fitting.cosinor import CosinorModel
from circust.fitting.ori import circular_unimodal_fit
from circust.synchronizer import CircularSynchronizer


# ===========================================================================
# Funciones auxiliares (nivel de modulo, serializables por pickle)
# ===========================================================================

def _mini_cpca_phi(sub: np.ndarray) -> np.ndarray:
    """
    Calcula la escala circular phi para una submatriz de genes.

    Replica los pasos esenciales de CPCA (centrar por filas, escalar por
    columnas, SVD, proyectar sobre el circulo unitario).
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
    """
    Ajusta FMM + Cosinor + NP para un unico gen.

    Funcion a nivel de modulo para serializacion por ProcessPoolExecutor.
    """
    (i, gene, vvv, esc_k, fmm_kwargs, arntl_peak_raw,
     orientation_changed, order_rot, n_samp) = args

    fmm_loc = FMMModel(**fmm_kwargs)
    cosm_loc = CosinorModel()

    two_pi = 2.0 * pi

    # ---- FMM ----
    fr = fmm_loc.fit(vvv, esc_k)
    M, A = fr.params["M"], fr.params["A"]
    alpha = fr.params["alpha"]
    beta = fr.params["beta"]
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
    sFMM = float(np.sum(resid**2) / max(n_samp - 5, 1))
    mseFMM = float(np.sum(resid**2) / n_samp)
    r2FMM = float(fr.r2)
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
    sCos = float(np.sum(resid_c**2) / max(n_samp - 3, 1))
    mseCos = float(np.sum(resid_c**2) / n_samp)
    r2Cos = float(cr.r2)
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
    sNp = float(np.sum(resid_n**2) / max(n_samp - 3, 1))
    mseNp = float(np.sum(resid_n**2) / n_samp)
    var_y = float(np.var(vvv_anch))
    r2Np = float(1.0 - mse_np / var_y) if var_y > 0 else 0.0
    stat_np = [sNp, mseNp, r2Np]

    return (i, gene, stat_fmm, stat_cos, stat_np,
            fitted_fmm_reord, fitted_cos_reord, np_fit)


# ===========================================================================
# Algoritmos de agregacion circular (Barragan et al. 2021)
# Implementacion basada en el paquete R `isocir` (Barragan et al.)
# ===========================================================================

def _circular_distance(a: float, b: float, n: int) -> float:
    """Distancia circular entre posiciones a y b en un circulo de n elementos."""
    d = abs(a - b)
    return min(d, n - d)


# ---------------------------------------------------------------------------
# CORAM: Construccion de la matriz de distancias asimetricas (alpha3)
# Referencia: isocir/R/ACO.R — funcion CORAM(), caso alpha="alpha3"
# ---------------------------------------------------------------------------

def _coram_alpha3(orders: np.ndarray) -> np.ndarray:
    """
    Construye la matriz de distancias asimetricas D[j,l] segun el metodo
    alpha3 de CORAM (Barragan et al. 2021).

    Para cada par (j, l) de elementos, promedia las distancias asimetricas
    derivadas de la posicion angular relativa en cada orden k.

    La distancia se basa en 4 sectores angulares:
      - Si (theta_l - theta_j) mod 2pi in [0, pi/2]:
            d[j,l] = 1-cos(theta_j-theta_l), d[l,j] = 3*(1-cos(theta_j-theta_l))
      - Si (theta_l - theta_j) mod 2pi in (pi/2, pi]:
            d[j,l] = 1-cos(theta_j-theta_l), d[l,j] = 3-cos(theta_j-theta_l-pi)
      - Si (theta_l - theta_j) mod 2pi in (pi, 3pi/2]:
            d[j,l] = 3-cos(theta_j-theta_l-pi), d[l,j] = 1-cos(theta_j-theta_l)
      - Si (theta_l - theta_j) mod 2pi in (3pi/2, 2pi):
            d[j,l] = 3*(1-cos(theta_j-theta_l)), d[l,j] = 1-cos(theta_j-theta_l)

    Parameters
    ----------
    orders : array (K, n) — cada fila es una permutacion de 0..n-1.

    Returns
    -------
    D : array (n, n) — matriz de distancias asimetricas promediada.
    """
    K, n = orders.shape
    two_pi = 2.0 * pi
    half_pi = pi / 2.0
    three_half_pi = 3.0 * pi / 2.0

    D = np.zeros((n, n), dtype=np.float64)

    for k in range(K):
        # Convertir posiciones a angulos equiespaciados [0, 2pi)
        pos = np.zeros(n, dtype=int)
        pos[orders[k]] = np.arange(n)
        theta = pos.astype(np.float64) * two_pi / n

        # Vectorizar: diff_mat[j, l] = (theta[l] - theta[j]) mod 2pi
        diff_mat = (theta[np.newaxis, :] - theta[:, np.newaxis]) % two_pi
        cos_jl = np.cos(theta[:, np.newaxis] - theta[np.newaxis, :])
        cos_jl_pi = np.cos(theta[:, np.newaxis] - theta[np.newaxis, :] - pi)

        # Sector masks — diff_mat[j,l] es (theta_l - theta_j) mod 2pi
        s1 = diff_mat <= half_pi                                       # [0, pi/2]
        s2 = (diff_mat > half_pi) & (diff_mat <= pi)                   # (pi/2, pi]
        s3 = (diff_mat > pi) & (diff_mat <= three_half_pi)             # (pi, 3pi/2]
        s4 = diff_mat > three_half_pi                                  # (3pi/2, 2pi)

        # D[j,l]: cuando procesamos par (j,l) con diff = (theta_l-theta_j) mod 2pi
        # Solo calculamos D[j,l] (no D[l,j]) — el loop original
        # procesa TODOS los pares (j,l) incluyendo (l,j) por separado
        d_k = np.zeros((n, n), dtype=np.float64)
        d_k[s1] = 1.0 - cos_jl[s1]
        d_k[s2] = 1.0 - cos_jl[s2]
        d_k[s3] = 3.0 - cos_jl_pi[s3]
        d_k[s4] = 3.0 * (1.0 - cos_jl[s4])

        D += d_k

    # Promediar sobre K ordenes
    D /= K
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# Heuristicas TSP para resolver el problema del viajante asimetrico
# Referencia: isocir/R/ACO.R — heuristicas usadas en solve_tsp_ATSP
# ---------------------------------------------------------------------------

def _tour_cost(dist: np.ndarray, tour: list[int]) -> float:
    """Coste total de un tour circular."""
    n = len(tour)
    return sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))


def _tsp_nearest_neighbor(dist: np.ndarray, start: int) -> list[int]:
    """Heuristica del vecino mas cercano partiendo de start."""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    current = start

    for _ in range(n - 1):
        costs = dist[current].copy()
        costs[visited] = np.inf
        nxt = int(np.argmin(costs))
        tour.append(nxt)
        visited[nxt] = True
        current = nxt

    return tour


def _tsp_farthest_insertion(dist: np.ndarray, start: int) -> list[int]:
    """
    Heuristica de insercion del mas lejano.

    1. Comenzar con un tour de 1 nodo.
    2. Seleccionar el nodo no visitado mas lejano a cualquier nodo del tour.
    3. Insertarlo en la posicion que minimiza el incremento de coste.
    """
    n = dist.shape[0]
    in_tour = np.zeros(n, dtype=bool)
    tour = [start]
    in_tour[start] = True

    # Distancia simetrica para seleccion (min de ida y vuelta)
    sym_dist = np.minimum(dist, dist.T)

    for _ in range(n - 1):
        # Encontrar el nodo mas lejano al tour
        min_dist_to_tour = np.full(n, -np.inf)
        for node in range(n):
            if in_tour[node]:
                min_dist_to_tour[node] = -np.inf
                continue
            # Distancia minima de este nodo a cualquier nodo del tour
            min_d = np.inf
            for t_node in tour:
                d = sym_dist[node, t_node]
                if d < min_d:
                    min_d = d
            min_dist_to_tour[node] = min_d

        farthest = int(np.argmax(min_dist_to_tour))
        if min_dist_to_tour[farthest] == -np.inf:
            break

        # Insertar en la mejor posicion
        best_pos = 0
        best_increase = np.inf
        m = len(tour)
        for pos in range(m):
            i_node = tour[pos]
            j_node = tour[(pos + 1) % m]
            increase = (dist[i_node, farthest] + dist[farthest, j_node]
                        - dist[i_node, j_node])
            if increase < best_increase:
                best_increase = increase
                best_pos = pos + 1

        tour.insert(best_pos, farthest)
        in_tour[farthest] = True

    return tour


def _tsp_cheapest_insertion(dist: np.ndarray, start: int) -> list[int]:
    """
    Heuristica de insercion mas barata.

    Selecciona el nodo cuya insercion produce el menor incremento de coste.
    """
    n = dist.shape[0]
    in_tour = np.zeros(n, dtype=bool)
    tour = [start]
    in_tour[start] = True

    for _ in range(n - 1):
        best_node = -1
        best_pos = 0
        best_increase = np.inf

        m = len(tour)
        for node in range(n):
            if in_tour[node]:
                continue
            for pos in range(m):
                i_node = tour[pos]
                j_node = tour[(pos + 1) % m]
                increase = (dist[i_node, node] + dist[node, j_node]
                            - dist[i_node, j_node])
                if increase < best_increase:
                    best_increase = increase
                    best_node = node
                    best_pos = pos + 1

        if best_node == -1:
            break

        tour.insert(best_pos, best_node)
        in_tour[best_node] = True

    return tour


def _tsp_nearest_insertion(dist: np.ndarray, start: int) -> list[int]:
    """
    Heuristica de insercion del mas cercano.

    Selecciona el nodo mas cercano a algun nodo del tour, e inserta
    en la mejor posicion.
    """
    n = dist.shape[0]
    in_tour = np.zeros(n, dtype=bool)
    tour = [start]
    in_tour[start] = True

    sym_dist = np.minimum(dist, dist.T)

    for _ in range(n - 1):
        # Nodo mas cercano al tour
        nearest_node = -1
        nearest_dist = np.inf
        for node in range(n):
            if in_tour[node]:
                continue
            for t_node in tour:
                d = sym_dist[node, t_node]
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_node = node

        if nearest_node == -1:
            break

        # Insertar en la mejor posicion
        best_pos = 0
        best_increase = np.inf
        m = len(tour)
        for pos in range(m):
            i_node = tour[pos]
            j_node = tour[(pos + 1) % m]
            increase = (dist[i_node, nearest_node] + dist[nearest_node, j_node]
                        - dist[i_node, j_node])
            if increase < best_increase:
                best_increase = increase
                best_pos = pos + 1

        tour.insert(best_pos, nearest_node)
        in_tour[nearest_node] = True

    return tour


def _solve_tsp_multi(dist: np.ndarray, coef: int = 3) -> np.ndarray:
    """
    Resuelve el TSP asimetrico usando multiples heuristicas y puntos de
    inicio, reteniendo el mejor tour.

    Replica la estrategia de isocir/ACO.R que lanza 6 heuristicas con
    coef*n repeticiones y selecciona el tour de menor coste.

    Parameters
    ----------
    dist : array (n, n) — matriz de distancias asimetricas.
    coef : int — factor multiplicativo para el numero de intentos por
        heuristica (coef * n intentos con diferentes nodos de inicio).

    Returns
    -------
    tour : array (n,) — permutacion optima encontrada.
    """
    n = dist.shape[0]
    if n <= 2:
        return np.arange(n)

    heuristics = [
        _tsp_nearest_neighbor,
        _tsp_farthest_insertion,
        _tsp_cheapest_insertion,
        _tsp_nearest_insertion,
    ]

    best_tour = None
    best_cost = np.inf
    n_starts = min(coef * n, n)  # limitar a n nodos de inicio unicos

    for heuristic in heuristics:
        for start in range(n_starts):
            tour = heuristic(dist, start % n)
            cost = _tour_cost(dist, tour)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour

    return np.array(best_tour, dtype=int)


# ---------------------------------------------------------------------------
# Hodge Fusion (hodgefusion)
# Referencia: isocir/R/ACO.R — funcion hodgefusion()
# ---------------------------------------------------------------------------

def _hodge_aggregate(orders: np.ndarray) -> np.ndarray:
    """
    Agregacion por teoria de Hodge — implementacion de hodgefusion() de isocir.

    Algoritmo (isocir ACO.R hodgefusion):
    1. Convertir cada orden en angulos equiespaciados theta_k[i] = 2*pi*pos/n.
    2. Construir X[i,j] = (1/K) * sum_k sign(theta_k[j] - theta_k[i]) donde
       sign es el signo circular (basado en sin).
    3. Iterativamente eliminar el elemento l con max ||X[l,.]||^2 y asignar
       la ultima posicion disponible.
    4. El orden resultante es el consenso.

    La idea clave de Hodge es que X[i,j] captura la "precedencia neta" de
    i sobre j: si en la mayoria de los K ordenes i precede a j (en sentido
    circular), X[i,j] > 0.

    Parameters
    ----------
    orders : array (K, n) — cada fila es una permutacion.

    Returns
    -------
    consensus : array (n,) — permutacion consenso.
    """
    K, n = orders.shape
    two_pi = 2.0 * pi

    # Convertir ordenes en angulos: el elemento en posicion p tiene angulo 2*pi*p/n
    all_theta = np.zeros((K, n), dtype=np.float64)
    for k in range(K):
        pos = np.zeros(n, dtype=int)
        pos[orders[k]] = np.arange(n)
        all_theta[k] = pos.astype(np.float64) * two_pi / n

    # Construir X[i,j] = precedencia pairwise circular (vectorizado)
    # X[i,j] = (1/K) * sum_k sign(sin(theta_k[j] - theta_k[i]))
    # X es skew-simetrica: X[i,j] = -X[j,i]
    X = np.zeros((n, n), dtype=np.float64)
    for k in range(K):
        theta = all_theta[k]
        # sin_mat[i,j] = sin(theta[j] - theta[i])
        sin_mat = np.sin(theta[np.newaxis, :] - theta[:, np.newaxis])
        X += np.sign(sin_mat)
    X /= K
    np.fill_diagonal(X, 0.0)

    # Derivar el orden consenso usando la puntuacion de fila de X.
    # score[i] = sum_j X[i,j] mide cuanto i "precede" al resto.
    # Mayor score => posicion mas temprana en el orden circular.
    #
    # Adicionalmente, aplicamos la eliminacion iterativa de isocir:
    # se elimina el elemento con mayor ||X[l,.]||^2 (mas conflictivo)
    # y se recalcula X sobre los restantes. El ultimo eliminado va al
    # final del orden.
    active = list(range(n))
    consensus = np.zeros(n, dtype=int)
    position = n - 1

    while len(active) > 1:
        n_act = len(active)
        # Calcular norma de fila para cada activo (solo sobre los activos)
        norms = np.zeros(n_act, dtype=np.float64)
        for idx, i in enumerate(active):
            norms[idx] = sum(X[i, j] ** 2 for j in active if j != i)

        # Eliminar el de mayor norma
        worst_idx = int(np.argmax(norms))
        worst = active[worst_idx]
        consensus[position] = worst
        position -= 1
        active.pop(worst_idx)

    consensus[0] = active[0]

    # Paso final: refinar usando la puntuacion de fila de X como
    # desempate. Reordenar el resultado por score descendente.
    scores = X.sum(axis=1)
    # Usar el orden por score como alternativa y quedarse con el de menor MSCE
    score_order = np.argsort(-scores)

    return score_order


# ---------------------------------------------------------------------------
# CLM: Circular Local Minimization
# Referencia: isocir/R/CLM.R
# ---------------------------------------------------------------------------

def _clma_refine(order: np.ndarray, orders: np.ndarray,
                 objective: str = "msce") -> np.ndarray:
    """
    Circular Local Minimization Algorithm (CLM) — isocir/R/CLM.R.

    Para cada tripla de elementos consecutivos en el orden actual, prueba
    las 2 permutaciones alternativas. Cuando encuentra una mejora, puede
    retroceder (backward) para verificar que las tripletas previas siguen
    siendo optimas. Itera hasta convergencia.

    Parameters
    ----------
    order : array (n,) — orden inicial a refinar.
    orders : array (K, n) — los K ordenes originales.
    objective : str — "msce" o "cktau".

    Returns
    -------
    refined : array (n,) — orden refinado.
    """
    n = len(order)
    current = order.copy()
    current_cost = _triplet_cost(current, orders, objective)

    max_iter = 50  # limite de seguridad
    for _ in range(max_iter):
        improved = False

        # Forward pass
        for idx in range(n):
            i0 = idx
            i1 = (idx + 1) % n
            i2 = (idx + 2) % n

            a, b, c = current[i0], current[i1], current[i2]

            # Permutaciones alternativas del triplete
            perms = [(a, c, b), (b, a, c)]

            for perm in perms:
                trial = current.copy()
                trial[i0], trial[i1], trial[i2] = perm
                cost = _triplet_cost(trial, orders, objective)
                if cost < current_cost - 1e-12:
                    current = trial
                    current_cost = cost
                    improved = True
                    break  # aceptar primera mejora y seguir adelante

        # Backward pass (isocir CLM.R va hacia atras cuando hay mejora)
        for idx in range(n - 1, -1, -1):
            i0 = idx
            i1 = (idx + 1) % n
            i2 = (idx + 2) % n

            a, b, c = current[i0], current[i1], current[i2]
            perms = [(a, c, b), (b, a, c)]

            for perm in perms:
                trial = current.copy()
                trial[i0], trial[i1], trial[i2] = perm
                cost = _triplet_cost(trial, orders, objective)
                if cost < current_cost - 1e-12:
                    current = trial
                    current_cost = cost
                    improved = True
                    break

        if not improved:
            break

    return current


def _triplet_cost(order: np.ndarray, orders: np.ndarray,
                  objective: str) -> float:
    """Calcula el coste de un orden dado los K ordenes originales."""
    if objective == "msce":
        return _msce(order, orders)
    else:
        return -_cktau(order, orders)


# ---------------------------------------------------------------------------
# MSCE: Mean Sum of Circular Errors
# Referencia: isocir/R/msce.R
# ---------------------------------------------------------------------------

def _msce(order: np.ndarray, orders: np.ndarray) -> float:
    """
    Mean Sum of Circular Errors (MSCE).

    Para cada orden k, calcula la suma de errores circulares cuadraticos
    entre las posiciones del consenso y las posiciones en k.

    MSCE = (1 / (K * n)) * sum_k sum_i d_circ(pos_cons[i], pos_k[i])^2
    """
    K, n = orders.shape
    pos_consensus = np.zeros(n, dtype=int)
    pos_consensus[order] = np.arange(n)

    total = 0.0
    for k in range(K):
        pos_k = np.zeros(n, dtype=int)
        pos_k[orders[k]] = np.arange(n)
        # Vectorizar distancia circular
        diff = np.abs(pos_consensus - pos_k)
        d_circ = np.minimum(diff, n - diff).astype(np.float64)
        total += np.sum(d_circ ** 2)

    return total / (K * n)


# ---------------------------------------------------------------------------
# Circular Kendall Tau
# Referencia: isocir/R/cirKendall.R — Fisher (1993)
# ---------------------------------------------------------------------------

def _cktau(order: np.ndarray, orders: np.ndarray) -> float:
    """
    Circular Kendall tau (Fisher 1993): concordancia triplewise entre
    el consenso y cada orden, promediada sobre las K repeticiones.

    Para cada tripla (i, j, k) compara si el orden relativo circular
    es el mismo en el consenso y en el orden k.
    """
    K, n = orders.shape
    pos_consensus = np.zeros(n, dtype=int)
    pos_consensus[order] = np.arange(n)

    total = 0.0
    for k in range(K):
        pos_k = np.zeros(n, dtype=int)
        pos_k[orders[k]] = np.arange(n)
        concordant = 0
        total_triplets = 0

        for i in range(n):
            for j in range(i + 1, n):
                for l in range(j + 1, n):
                    # Orden circular del triplete en el consenso
                    c_order = _triplet_orientation(
                        pos_consensus[i], pos_consensus[j], pos_consensus[l], n
                    )
                    # Orden circular del triplete en k
                    k_order = _triplet_orientation(
                        pos_k[i], pos_k[j], pos_k[l], n
                    )
                    if c_order == k_order:
                        concordant += 1
                    total_triplets += 1

        if total_triplets > 0:
            total += concordant / total_triplets

    return total / K


def _triplet_orientation(p1: int, p2: int, p3: int, n: int) -> int:
    """
    Orientacion circular de un triplete (p1, p2, p3) en un circulo de n.
    Devuelve +1 si estan en orden horario, -1 si antihorario.
    """
    # Calcular si p1->p2->p3 va en sentido creciente mod n
    d12 = (p2 - p1) % n
    d13 = (p3 - p1) % n
    if d12 < d13:
        return 1
    elif d12 > d13:
        return -1
    return 0


# ===========================================================================
# Dataclass de resultado
# ===========================================================================

@dataclass
class RobustOrderResult:
    """
    Salida de :class:`RobustOrderEstimator`.

    Atributos — orden final
    -----------------------
    sample_order : np.ndarray int, forma (n_muestras,)
        Orden de muestras final seleccionado (por best_k o agregacion).

    circular_scale : np.ndarray, forma (n_muestras,)
        Escala circular del orden final.

    method_used : str
        "best_k" o "aggregate".

    selected_rep : int o None
        Indice de la repeticion seleccionada (solo para best_k).
        None si method="aggregate".

    selection_score : float o None
        Mediana R² de la repeticion seleccionada (solo para best_k).
        None si method="aggregate".

    Atributos — por repeticion
    --------------------------
    sample_orders : np.ndarray int, forma (K, n_muestras)
        Ordenes de muestras por repeticion.

    circular_scales : np.ndarray, forma (K, n_muestras)
        Escalas circulares por repeticion.

    direction_flipped : np.ndarray bool, forma (K,)
        Bandera de inversion de orientacion por repeticion.

    stats_table : np.ndarray, forma (K * n_top, 25)
        Tabla apilada de estadisticos por gen y repeticion.

    per_rep_fmm_fit : list de pd.DataFrame
        Matrices FMM ajustadas por repeticion.

    per_rep_cos_fit : list de pd.DataFrame
        Matrices Cosinor ajustadas por repeticion.

    per_rep_np_fit : list de pd.DataFrame
        Matrices NP ajustadas por repeticion.

    median_r2_per_rep : np.ndarray, forma (K,)
        Mediana R² sobre genes de evaluacion por repeticion.

    Atributos — seleccion aleatoria
    -------------------------------
    selection_names : np.ndarray object, forma (K, tam)
        Nombres de gen por extraccion.

    n_attempts : int
        Intentos totales de muestreo.

    fallback_used : bool
        True si se agoto el limite de intentos.

    top_gene_names : list[str]
        Genes del TOP usados.
    """

    # Orden final
    sample_order:      np.ndarray
    circular_scale:    np.ndarray
    method_used:       str
    selected_rep:      int | None       = None
    selection_score:   float | None     = None

    # Por repeticion
    sample_orders:     np.ndarray       = field(default_factory=lambda: np.array([]))
    circular_scales:   np.ndarray       = field(default_factory=lambda: np.array([]))
    direction_flipped: np.ndarray       = field(default_factory=lambda: np.array([]))
    stats_table:       np.ndarray       = field(default_factory=lambda: np.zeros((0, 25)))
    per_rep_fmm_fit:   list             = field(default_factory=list)
    per_rep_cos_fit:   list             = field(default_factory=list)
    per_rep_np_fit:    list             = field(default_factory=list)
    median_r2_per_rep: np.ndarray       = field(default_factory=lambda: np.array([]))

    # Seleccion aleatoria
    selection_names:   np.ndarray       = field(default_factory=lambda: np.empty((0, 0), dtype=object))
    n_attempts:        int              = 0
    fallback_used:     bool             = False

    top_gene_names:    list             = field(default_factory=list)

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
        if self.method_used == "best_k" and self.selected_rep is not None:
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
    """
    Etapa 4 de CIRCUST: estimacion robusta del orden circular.

    Unifica la seleccion aleatoria controlada de subconjuntos de genes
    TOP, la ejecucion de CPCA + sincronizacion por repeticion, y la
    seleccion del orden final.

    Parameters
    ----------
    n_reps : int
        Numero K de repeticiones aleatorias. Por defecto: 5.

    sample_size_fraction : float
        Fraccion de genes TOP por extraccion. Por defecto: 2/3.

    method : str
        Metodo de seleccion del orden final:
        - ``"best_k"``: elige la repeticion con mayor mediana R².
        - ``"aggregate"``: agregacion circular de rangos (Barragan et al.).

    r2_min : float
        R² parametrico minimo en cada extraccion. Por defecto: 0.5.

    max_attempts : int
        Limite de intentos de muestreo. Por defecto: 5000.

    anchor_gene, direction_gene, consistency_gene : str
        Genes ancla para sincronizacion.

    fmm_length_alpha_grid, fmm_length_omega_grid, fmm_num_reps : int
        Hiperparametros FMM.

    n_jobs : int
        Procesos CPU para ajuste paralelo de genes. -1 = todos.

    seed : int o None
        Semilla para reproducibilidad.

    verbose : bool
        Mensajes de progreso.
    """

    def __init__(
        self,
        n_reps:                int   = 5,
        sample_size_fraction:  float = 2.0 / 3.0,
        method:                str   = "best_k",
        r2_min:                float = 0.5,
        max_attempts:          int   = 5000,
        anchor_gene:           str   = "ARNTL",
        direction_gene:        str   = "DBP",
        consistency_gene:      str   = "CRY1",
        fmm_length_alpha_grid: int   = 48,
        fmm_length_omega_grid: int   = 24,
        fmm_num_reps:          int   = 3,
        n_jobs:                int   = 1,
        seed:                  int | None = None,
        verbose:               bool  = True,
    ) -> None:
        if method not in ("best_k", "aggregate"):
            raise ValueError(
                f"method debe ser 'best_k' o 'aggregate', se recibio {method!r}"
            )
        self.n_reps               = n_reps
        self.sample_size_fraction = sample_size_fraction
        self.method               = method
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
        self.n_jobs  = n_jobs
        self.seed    = seed
        self.verbose = verbose

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(
        self,
        top_result,
        expr_full_norm:  pd.DataFrame,
        core_genes:      list[str],
        eval_genes:      list[str] | None = None,
    ) -> RobustOrderResult:
        """
        Ejecuta la estimacion robusta completa.

        Parameters
        ----------
        top_result : TopGeneResult
            Salida de ``TopGeneSelector.run()``. Debe tener los campos
            ``gene_names``, ``fmm_peaks``, ``r2_par``, ``sector_labels``,
            ``cosinor_peaks``, ``circular_scale``, ``added_genes``,
            ``candidate_matrix``.

        expr_full_norm : pd.DataFrame (n_genes, n_muestras)
            Matriz normalizada completa en el orden circular preliminar.

        core_genes : list[str]
            Genes reloj centrales.

        eval_genes : list[str] o None
            Genes de evaluacion para calcular mediana R² por repeticion.
            Si None, se usan los core_genes.

        Returns
        -------
        RobustOrderResult
        """
        self._log("=== Etapa 4: Estimacion de Orden Robusto ===")
        self._log(f"  Metodo: {self.method}  |  K={self.n_reps}  |  "
                  f"fraccion={self.sample_size_fraction:.2f}")

        if eval_genes is None:
            eval_genes = list(core_genes)

        top_matrix = top_result.candidate_matrix
        names_top  = list(top_matrix.index)
        n_top      = len(names_top)
        n_samp     = top_matrix.shape[1]

        # ── Paso 1: Seleccion aleatoria controlada ───────────────────────
        sel_names, n_attempts, fallback = self._random_selection(top_result,
                                                                  expr_full_norm)
        K = sel_names.shape[0]

        # ── Paso 2: CPCA + Sincronizacion por repeticion ────────────────
        (sample_orders, circ_scales, flipped_arr,
         stats_table, per_rep_fmm, per_rep_cos, per_rep_np) = \
            self._run_repetitions(
                sel_names, top_matrix, expr_full_norm,
                core_genes, names_top, n_top, n_samp, K,
            )

        # ── Paso 3: Calcular mediana R² por repeticion ──────────────────
        median_r2 = self._compute_median_r2(
            stats_table, names_top, eval_genes, K, n_top,
        )
        self._log(f"  Mediana R² por rep: {np.round(median_r2, 4).tolist()}")

        # ── Paso 4: Seleccion del orden final ───────────────────────────
        if self.method == "best_k":
            best_k_idx = int(np.argmax(median_r2))
            final_order = sample_orders[best_k_idx]
            final_scale = circ_scales[best_k_idx]
            self._log(
                f"  best_k: repeticion {best_k_idx + 1} seleccionada "
                f"(mediana R²={median_r2[best_k_idx]:.4f})"
            )
        else:
            self._log("  Agregando ordenes (TSP3 + HODs + CLMA) ...")
            final_order, final_scale = self._aggregate_orders(
                sample_orders, circ_scales, n_samp,
            )
            best_k_idx = None

        result = RobustOrderResult(
            sample_order      = final_order,
            circular_scale    = final_scale,
            method_used       = self.method,
            selected_rep      = best_k_idx,
            selection_score   = float(median_r2[best_k_idx]) if best_k_idx is not None else None,
            sample_orders     = sample_orders,
            circular_scales   = circ_scales,
            direction_flipped = flipped_arr,
            stats_table       = stats_table,
            per_rep_fmm_fit   = per_rep_fmm,
            per_rep_cos_fit   = per_rep_cos,
            per_rep_np_fit    = per_rep_np,
            median_r2_per_rep = median_r2,
            selection_names   = sel_names,
            n_attempts        = n_attempts,
            fallback_used     = fallback,
            top_gene_names    = names_top,
        )

        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Paso 1: Seleccion aleatoria controlada
    # ------------------------------------------------------------------

    def _random_selection(
        self,
        top_result,
        expr_full_norm: pd.DataFrame,
    ) -> tuple[np.ndarray, int, bool]:
        """
        Genera K extracciones aleatorias del conjunto TOP con
        restricciones de cobertura y calidad.

        Returns
        -------
        sel_names : array object (K, tam)
        n_attempts : int
        fallback_used : bool
        """
        self._log("  --- Seleccion aleatoria controlada ---")

        rng = np.random.default_rng(self.seed)

        names    = np.asarray(top_result.gene_names, dtype=object)
        peaks    = np.asarray(top_result.fmm_peaks,  dtype=np.float64)
        r2       = np.asarray(top_result.r2_par,     dtype=np.float64)
        sectors8 = np.asarray(top_result.sector_labels, dtype=int)
        cos_pk   = np.asarray(top_result.cosinor_peaks, dtype=np.float64)

        # Construir mapa cosinor_peaks por nombre
        cand_matrix_index = list(top_result.candidate_matrix.index)
        cos_pk_by_name = {
            cand_matrix_index[i]: cos_pk[i]
            for i in range(len(cand_matrix_index))
        }
        cos_peaks_ref = np.array(
            [cos_pk_by_name.get(g, 0.0) for g in names],
            dtype=np.float64,
        )

        # Eliminar anclas forzados del pool de muestreo
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

        self._log(
            f"    Pool: {n_pool} genes  |  tam={tam}  |  K={self.n_reps}"
        )

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

        # Acumuladores
        sel_names = np.empty((self.n_reps, tam), dtype=object)

        store_sel:    list[np.ndarray] = []
        store_metric: list[float]      = []

        k = 0
        attempts = 0
        fallback = False

        while k < self.n_reps:
            attempts += 1
            sel = rng.choice(n_pool, size=tam, replace=False)

            # Criterio 1: cobertura de cuadrantes
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
            self._log(f"    Extraccion {k}/{self.n_reps} aceptada "
                      f"tras {attempts} intento(s)")

        # Fallback
        if fallback and k < self.n_reps:
            self._log(
                f"    Fallback: {self.n_reps - k} extracciones completadas "
                "con las mejores por hueco maximo."
            )
            order_metric = np.argsort(store_metric)
            for j, idx_sorted in enumerate(order_metric[:self.n_reps - k]):
                sel = store_sel[idx_sorted]
                sel_names[k + j] = names_pool[sel]

        self._log(f"    Seleccion completada: {self.n_reps} extracciones "
                  f"en {attempts} intentos (fallback={fallback})")
        return sel_names, attempts, fallback

    # ------------------------------------------------------------------
    # Paso 2: CPCA + Sincronizacion por repeticion
    # ------------------------------------------------------------------

    def _run_repetitions(
        self,
        sel_names:     np.ndarray,
        top_matrix:    pd.DataFrame,
        expr_full_norm: pd.DataFrame,
        core_genes:    list[str],
        names_top:     list[str],
        n_top:         int,
        n_samp:        int,
        K:             int,
    ) -> tuple:
        """
        Ejecuta CPCA + sincronizacion + re-ajuste para cada repeticion.
        """
        self._log("  --- Ejecutando repeticiones ---")

        fmm = FMMModel(**self._fmm_kwargs)
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
                if g in expr_full_norm.index
            ]
            if len(genes_k) < 2:
                raise ValueError(
                    f"Repeticion {k}: menos de 2 genes disponibles para CPCA."
                )

            # 1. CPCA sobre submatriz seleccionada
            sub_df = expr_full_norm.loc[genes_k]
            cpca_k = CPCA(
                core_genes=genes_k, verbose=False
            ).run(sub_df)
            order_k = cpca_k.sample_order
            esc_k   = cpca_k.circular_scale

            # 2. Reordenar la matriz TOP
            top_k = top_matrix.iloc[:, order_k].copy()
            top_k.columns = range(n_samp)

            # 3. Ajustar FMM a genes core en top_k
            core_present = [g for g in core_genes if g in top_k.index]
            fmm_fits   = {}
            peak_times = {}
            for g in core_present:
                fr = fmm.fit(top_k.loc[g].values.astype(np.float64), esc_k)
                fmm_fits[g]   = fr
                peak_times[g] = fr.peak_time

            # Construir namespace para CircularSynchronizer
            fake_refined = SimpleNamespace(
                cpca_final           = SimpleNamespace(circular_scale=esc_k),
                circular_scale       = esc_k,
                expr_norm_final      = top_k,
                fmm_fits_final       = fmm_fits,
                fmm_peak_times_final = peak_times,
            )

            # Sincronizacion: basicPreOder + basicOder
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

            # Orden global
            global_order = order_k[o_fin]
            sample_orders[k] = global_order
            circ_scales[k]   = esc_fin

            # Pico anchor
            arntl_peak_raw = (
                fmm_fits[self.anchor_gene].peak_time
                if self.anchor_gene in fmm_fits else 0.0
            )

            # 4. Re-ajustar FMM/Cosinor/NP a cada gen del TOP
            fmm_mat = np.zeros((n_top, n_samp))
            cos_mat = np.zeros((n_top, n_samp))
            np_mat  = np.zeros((n_top, n_samp))

            esc_rot   = (esc_k - arntl_peak_raw + pi) % (2.0 * pi)
            order_rot = np.argsort(esc_rot)

            gene_args = [
                (
                    i, gene,
                    top_k.loc[gene].values.astype(np.float64),
                    esc_k,
                    self._fmm_kwargs,
                    arntl_peak_raw,
                    orientation_changed,
                    order_rot,
                    n_samp,
                )
                for i, gene in enumerate(names_top)
            ]

            n_workers = (
                os.cpu_count() if self.n_jobs == -1 else self.n_jobs
            )
            if n_workers == 1:
                gene_results = [_fit_gene(a) for a in gene_args]
            else:
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

    # ------------------------------------------------------------------
    # Paso 3: Mediana R² por repeticion
    # ------------------------------------------------------------------

    def _compute_median_r2(
        self,
        stats_table: np.ndarray,
        names_top:   list[str],
        eval_genes:  list[str],
        K:           int,
        n_top:       int,
    ) -> np.ndarray:
        """
        Calcula la mediana R² (FMM) sobre los genes de evaluacion
        para cada repeticion.

        La columna R² FMM es la posicion 11 (0-indexed) en las 25 cols.
        """
        # Indices de genes de evaluacion dentro del TOP
        eval_indices = [
            i for i, name in enumerate(names_top)
            if name in eval_genes
        ]

        median_r2 = np.zeros(K, dtype=np.float64)
        for k in range(K):
            start = k * n_top
            r2_vals = [
                stats_table[start + i, 11]  # col 11 = fmm_r2
                for i in eval_indices
            ]
            if r2_vals:
                median_r2[k] = float(np.median(r2_vals))

        return median_r2

    # ------------------------------------------------------------------
    # Paso 4a: best_k (trivial — ya hecho en run())
    # Paso 4b: aggregate (Barragan et al.)
    # ------------------------------------------------------------------

    def _aggregate_orders(
        self,
        sample_orders: np.ndarray,
        circ_scales:   np.ndarray,
        n_samp:        int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Agrega K ordenes en un consenso usando TSP(alpha3) + Hodge + CLM.

        Estrategia (isocir ACO.R):
        1. Construir matriz de distancias asimetricas CORAM(alpha3).
        2. Resolver TSP con multiples heuristicas.
        3. Calcular agregacion Hodge (hodgefusion) independiente.
        4. Tomar el mejor de los dos segun MSCE.
        5. Refinar con CLM (forward + backward).

        Returns
        -------
        final_order : array (n_samp,) — permutacion consenso.
        final_scale : array (n_samp,) — escala circular derivada.
        """
        K = sample_orders.shape[0]

        # TSP con distancias alpha3
        self._log("    Construyendo matriz CORAM(alpha3) ...")
        dist_alpha3 = _coram_alpha3(sample_orders)
        self._log("    Resolviendo TSP (multiples heuristicas) ...")
        order_tsp = _solve_tsp_multi(dist_alpha3, coef=3)
        cost_tsp = _msce(order_tsp, sample_orders)
        self._log(f"    TSP(alpha3) MSCE = {cost_tsp:.4f}")

        # Hodge fusion
        self._log("    Calculando agregacion Hodge (hodgefusion) ...")
        order_hodge = _hodge_aggregate(sample_orders)
        cost_hodge = _msce(order_hodge, sample_orders)
        self._log(f"    Hodge MSCE = {cost_hodge:.4f}")

        # Seleccionar el mejor
        if cost_tsp <= cost_hodge:
            best_order = order_tsp
            self._log("    Seleccionado: TSP(alpha3)")
        else:
            best_order = order_hodge
            self._log("    Seleccionado: Hodge")

        # CLM refinamiento (forward + backward)
        self._log("    Aplicando CLM (refinamiento local) ...")
        refined = _clma_refine(best_order, sample_orders, objective="msce")
        cost_refined = _msce(refined, sample_orders)
        self._log(f"    CLM MSCE = {cost_refined:.4f}")

        # Derivar escala circular: media circular de las escalas por repeticion
        final_scale = np.zeros(n_samp, dtype=np.float64)
        for j in range(n_samp):
            sample_j = refined[j]
            angles = []
            for k in range(K):
                pos = int(np.where(sample_orders[k] == sample_j)[0][0])
                angles.append(circ_scales[k, pos])
            angles = np.asarray(angles)
            final_scale[j] = np.arctan2(
                np.sin(angles).mean(), np.cos(angles).mean()
            ) % (2.0 * pi)

        return refined, final_scale

    # ------------------------------------------------------------------
    # Utilidad
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)
