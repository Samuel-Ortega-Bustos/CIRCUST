"""
circust/circular_aggregation.py
================================
Agregacion de ordenes circulares — paper Barragan, Rueda y Fernandez (2015).

Toma K ordenes circulares observados sobre las mismas n items (cada orden es
un vector de angulos en [0, 2*pi)) y produce un orden consenso. Util para
combinar resultados de distintas repeticiones del pipeline RobustOrder.

Metodos implementados
---------------------
- **TSP3**: TSP con distancia direccional asimetrica con penalizacion alpha=3
  para movimientos en la direccion equivocada. Recomendado por el paper para
  el criterio MSCE (Mean Sum of Circular Errors).

- **HODs** (proximamente): Hodge score-based triplewise. Mas rapido, mejor
  para CKTau. Aun no implementado.

API publica
-----------
- ``aggregate_circular_orders(Theta, weights, method="tsp3")``: devuelve un
  orden consenso (permutacion de items) y diagnostico.

Equivalente en R: paquete ``isocir`` (funcion ``ACO``), referencia exacta
para esta implementacion.
"""
from __future__ import annotations

from typing import Optional
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 1 — Helpers numericos
# ═══════════════════════════════════════════════════════════════════════════

def _circular_dist(a: float, b: float) -> float:
    """Distancia angular minima entre dos angulos (resultado en ``[0, pi]``).

    Equivalente a la metrica `min(d, 2π−d)` con d=|a−b| mod 2π.
    """
    d = abs(a - b) % (2.0 * np.pi)
    return float(min(d, 2.0 * np.pi - d))


def _circular_dist_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Version vectorizada de ``_circular_dist`` (broadcasting)."""
    d = np.abs(a - b) % (2.0 * np.pi)
    return np.minimum(d, 2.0 * np.pi - d)


def _msce_simple(
    Theta:        np.ndarray,
    target_order: np.ndarray,
    weights:      Optional[np.ndarray] = None,
) -> float:
    """Mean Sum of Circular Errors aproximado (sin CIRE).

    Para cada repeticion k:
      1. Asigna angulos uniformes ``phi_i = 2*pi * pos(i)/n`` a cada item
         segun ``target_order``.
      2. Calcula la rotacion global optima ``delta_k`` que minimiza
         ``sum (1 - cos(theta_{ki} - phi_i - delta_k))``: cierre analitico
         ``delta_k = atan2(sum sin(theta-phi), sum cos(theta-phi))``.
      3. MSCE_k = mean(1 - cos(theta_{ki} - phi_i - delta_k)).

    El paso 2 es la simplificacion clave frente al CIRE del paper: como nuestro
    target_order produce angulos uniformes, la mejor "calibracion" entre
    Theta[k] y el target es una rotacion global. Esto hace que ordenes
    circularmente equivalentes (rolled) den el mismo MSCE.

    El MSCE total es la media ponderada de los MSCE_k entre repeticiones.

    Parameters
    ----------
    Theta : np.ndarray, shape (K, n)
        Angulos observados por repeticion.
    target_order : np.ndarray, shape (n,)
        Permutacion de items que define el orden consenso.
    weights : np.ndarray opcional, shape (K,)
        Pesos por repeticion. Default uniforme.
    """
    K, n = Theta.shape
    # Posicion (0..n-1) que ocupa cada item en target_order
    positions = np.empty(n, dtype=int)
    positions[target_order] = np.arange(n)
    phi = (2.0 * np.pi / n) * positions          # angulo objetivo por item

    msces = np.empty(K)
    for k in range(K):
        diff = Theta[k] - phi
        # Rotacion optima: delta = atan2(sum sin diff, sum cos diff)
        delta_k = np.arctan2(np.sin(diff).sum(), np.cos(diff).sum())
        msces[k] = float(np.mean(1.0 - np.cos(diff - delta_k)))

    if weights is None:
        return float(msces.mean())
    return float(np.average(msces, weights=weights))


def _cktau_pairwise(theta1: np.ndarray, theta2: np.ndarray) -> float:
    """Circular Kendall Tau (CKTau) entre dos vectores angulares.

    Definido en Fisher (1993): para cada terna (i, h, k) de items distintos,
    se mide la concordancia de orientacion circular entre theta1 y theta2.
    Resultado en [-1, 1]: 1 = orden circular identico, -1 = invertido.
    """
    n = theta1.shape[0]
    if n < 3:
        return 0.0

    # Triple para todos los i<h<k
    idx = np.arange(n)
    i, h, k = np.meshgrid(idx, idx, idx, indexing="ij")
    mask = (i < h) & (h < k)

    def _orientation(t):
        d_ih = np.sign(np.sin(t[h] - t[i]))
        d_hk = np.sign(np.sin(t[k] - t[h]))
        d_ki = np.sign(np.sin(t[i] - t[k]))
        return d_ih * d_hk * d_ki

    o1 = _orientation(theta1)[mask]
    o2 = _orientation(theta2)[mask]
    nz = (o1 != 0) & (o2 != 0)
    if not nz.any():
        return 0.0
    return float((o1[nz] * o2[nz]).mean())


def _cktau_vs_order(Theta: np.ndarray, target_order: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> float:
    """CKTau medio entre cada rep en ``Theta`` y el orden ``target_order``.

    Convierte ``target_order`` a angulos uniformes y calcula CKTau con cada
    fila de ``Theta``, luego promedia (ponderado opcionalmente).
    """
    K, n = Theta.shape
    positions = np.empty(n, dtype=int)
    positions[target_order] = np.arange(n)
    phi = (2.0 * np.pi / n) * positions

    taus = np.array([_cktau_pairwise(Theta[k], phi) for k in range(K)])
    if weights is None:
        return float(taus.mean())
    return float(np.average(taus, weights=weights))


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 2 — TSP3 (alpha=3, recomendado para MSCE)
# ═══════════════════════════════════════════════════════════════════════════

def _distance_matrix_tsp3_single(theta: np.ndarray) -> np.ndarray:
    """Matriz de distancias TSP3 ``(n, n)`` para una sola repeticion.

    Para cada par (h, k):

      d_R(h, k) = 1 - cos(theta_k - theta_h)             si  diff <= pi
                  3 - cos(theta_k - theta_h - pi)        si  diff > pi
      d_C(h, k) = d_R(k, h)
      E[h, k]   = min(d_R(h, k), 3 * d_C(h, k))

    donde ``diff = (theta_k - theta_h) mod 2*pi``. La matriz ``E`` es
    asimetrica: capturar la direccion circular es la clave del enfoque TSP3.
    """
    theta = np.asarray(theta, dtype=float)
    diff  = (theta[None, :] - theta[:, None]) % (2.0 * np.pi)   # (n, n)
    # d_R: distancia en direccion "directa" (corto arco) o penalizada si hay que dar la vuelta
    d_R = np.where(
        diff <= np.pi,
        1.0 - np.cos(diff),
        3.0 - np.cos(diff - np.pi),
    )
    d_C = d_R.T                                                  # d_C(h,k) = d_R(k,h)
    E = np.minimum(d_R, 3.0 * d_C)
    np.fill_diagonal(E, 0.0)
    return E


def _build_distance_matrix_tsp3(
    Theta:   np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Matriz de distancias TSP3 agregada sobre las K repeticiones.

    ``E[h, k] = sum_j w_j * E^j[h, k]`` donde ``E^j`` es la matriz TSP3 de la
    repeticion j (formula del paper, eq. tras Sec. 3.1).
    """
    Theta = np.asarray(Theta, dtype=float)
    K, n = Theta.shape

    if weights is None:
        weights = np.ones(K) / K
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != K:
            raise ValueError(f"weights debe tener longitud {K}, got {weights.shape[0]}")
        s = weights.sum()
        if s <= 0:
            raise ValueError("Suma de weights debe ser > 0.")
        weights = weights / s

    E = np.zeros((n, n), dtype=float)
    for j in range(K):
        E += weights[j] * _distance_matrix_tsp3_single(Theta[j])
    return E


def _solve_tsp_nearest_neighbor(D: np.ndarray, start: int) -> tuple[np.ndarray, float]:
    """TSP nearest-neighbor desde un nodo de inicio.

    Heuristica simple pero robusta: empezar en ``start`` y, en cada paso,
    moverse al nodo no visitado mas cercano. Devuelve ``(tour, length)``.
    """
    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour    = np.empty(n, dtype=int)

    tour[0] = start
    visited[start] = True

    for i in range(1, n):
        last  = tour[i - 1]
        dists = D[last].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour[i] = next_node
        visited[next_node] = True

    # Longitud incluyendo el cierre del tour (vuelta al inicio)
    length = float(sum(D[tour[i], tour[(i + 1) % n]] for i in range(n)))
    return tour, length


def _solve_tsp_multi_start(
    D:        np.ndarray,
    n_starts: Optional[int] = None,
) -> tuple[np.ndarray, float]:
    """Multi-start nearest-neighbor: prueba varios puntos de inicio.

    El paper menciona que ningun heuristico TSP es siempre mejor; multi-start
    NN da resultados estables. Devuelve el tour de menor longitud.
    """
    n = D.shape[0]
    if n_starts is None:
        n_starts = min(n, 30)               # 30 starts es suficiente en practica
    n_starts = max(1, min(n_starts, n))

    starts = np.linspace(0, n - 1, n_starts, dtype=int)
    best_tour, best_length = None, np.inf
    for s in starts:
        tour, length = _solve_tsp_nearest_neighbor(D, start=int(s))
        if length < best_length:
            best_length = length
            best_tour   = tour
    return best_tour, best_length


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 3 — API publica
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_circular_orders(
    Theta:    np.ndarray,
    weights:  Optional[np.ndarray] = None,
    method:   str = "tsp3",
    n_starts: Optional[int] = None,
) -> dict:
    """Agrega K ordenes circulares en un orden consenso.

    Parameters
    ----------
    Theta : np.ndarray, shape (K, n)
        ``Theta[k, i]`` es el angulo en ``[0, 2*pi)`` asignado al item ``i``
        en la repeticion ``k``.
    weights : np.ndarray opcional, shape (K,)
        Pesos por repeticion. Por defecto uniforme.
    method : str
        - ``"tsp3"`` (default): TSP con alpha=3, recomendado para MSCE.
        - ``"hods"``: pendiente de implementar.
    n_starts : int opcional
        Numero de puntos de inicio para multi-start NN (solo TSP3).

    Returns
    -------
    dict con:
        - ``order``: np.ndarray (n,), permutacion de items en orden consenso
        - ``method``: str, metodo usado
        - ``tour_length``: float, suma de distancias TSP (solo TSP3)
        - ``msce``: float, MSCE simple del orden consenso
        - ``cktau``: float, CKTau medio del orden consenso
    """
    Theta = np.asarray(Theta, dtype=float) % (2.0 * np.pi)
    K, n = Theta.shape
    if K < 1:
        raise ValueError("Se requiere al menos 1 repeticion en Theta.")
    if n < 3:
        raise ValueError("Se requieren al menos 3 items para agregacion circular.")

    method = method.lower()

    if method == "tsp3":
        D = _build_distance_matrix_tsp3(Theta, weights)
        order, tour_length = _solve_tsp_multi_start(D, n_starts=n_starts)
    elif method == "hods":
        raise NotImplementedError("HODs aun no implementado, usa method='tsp3'.")
    else:
        raise ValueError(f"method desconocido: {method!r}")

    # Diagnostico: MSCE y CKTau del orden consenso
    msce  = _msce_simple(Theta, order, weights=weights)
    cktau = _cktau_vs_order(Theta, order, weights=weights)

    return {
        "order":       order,
        "method":      method,
        "tour_length": float(tour_length) if method == "tsp3" else None,
        "msce":        msce,
        "cktau":       cktau,
    }
