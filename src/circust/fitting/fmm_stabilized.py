"""
circust/fitting/fmm_stabilized.py
==================================
Modelo FMM estabilizado — version mejorada del ajuste FMM.

A diferencia del FMM base (``fmm.py``), este enfoque:

1. Usa una formulacion **numericamente estable** de la fase Mobius,
   sin singularidades en ``t = alpha +/- pi`` donde se tiende a inf.
2. Aplica **pesos por densidad temporal** (kernels KDE von Mises) que compensan
   distribuciones no uniformes de las muestras en el circulo.
3. Reemplaza la minimizacion del RSS por un **criterio de potencia
   penalizado**: J(alpha, omega) = P(alpha, omega) - lambda * g(omega) * R(alpha, omega).
4. Refina (alpha, omega) con busqueda local 2D + WLS cerrado para los
   parametros lineales, en vez de Nelder-Mead 5D.
5. Calibra ``lambda`` automaticamente bajo H0 mediante Monte Carlo + codo.

El resultado es un ajuste mas robusto al ruido y mas eficiente para
datasets grandes (la rejilla y el cache se precomputan una sola vez y
se reutilizan para todos los genes).

Producirse como ``FitResult`` con las mismas claves que ``FMMModel``
(``M, A, alpha, beta, omega``), de modo que es **drop-in** en cualquier
sitio del pipeline donde se use el FMM base.

"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import i0 as _bessel_i0

from circust.fitting.rhythm_model import FitResult, RhythmModel

# ── Aceleracion opcional con Numba (mismo patron que fmm.py) ───────────────
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):                                  # type: ignore
        def deco(fn): return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return deco


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 1 — Helpers numericos (privados)
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_weights_to_n(w: np.ndarray) -> np.ndarray:
    """Reescala pesos para que ``sum(w) == len(w)``.

    Usado en multiples sitios (preparacion de pesos de densidad, calibracion
    bajo H0). Mantener como funcion separada porque se llama desde varios
    contextos con misma logica de validacion.
    """
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("Pesos no finitos.")
    if np.any(w < 0):
        raise ValueError("Los pesos deben ser >= 0.")
    sw = float(w.sum())
    if sw <= 0.0:
        raise ValueError("Suma de pesos debe ser > 0.")
    return w.shape[0] * w / sw


def _time_density_weights(
    times:      np.ndarray,
    kappa:      Optional[float]      = None,
    kappa_grid: Optional[np.ndarray] = None,
    eps:        float                = 1e-12,
) -> np.ndarray:
    """Pesos de densidad temporal von Mises:  ``w_i ~ 1 / f_hat(t_i)``,  ``sum(w) = n``.

    Construye una KDE von Mises sobre los ``times`` y devuelve pesos
    inversamente proporcionales a la densidad estimada — las muestras en
    zonas temporales dispersas reciben mas peso.

    Si ``kappa`` no se especifica, se selecciona automaticamente por
    Leave-One-Out cross-validation sobre ``kappa_grid``.

    Optimizacion frente al codigo R: ``cos(delta)`` se precomputa una sola
    vez y se reutiliza para todos los kappa candidatos del LOO-CV (en R se
    recalcula la matriz completa de kernels en cada iteracion).
    """
    times = np.asarray(times, dtype=float) % (2.0 * np.pi)
    n = times.shape[0]
    if n < 2:
        raise ValueError("Se necesitan al menos 2 muestras.")

    # Matriz n x n de cos(delta) — compartida por todos los candidatos kappa
    cos_delta = np.cos(times[:, None] - times[None, :])

    # Seleccion automatica de kappa por LOO-CV
    if kappa is None:
        if kappa_grid is None:
            kappa_grid = np.exp(np.linspace(np.log(1.0), np.log(64.0), 10))
        kappa_grid = np.asarray(kappa_grid, dtype=float)
        if np.any(~np.isfinite(kappa_grid)) or np.any(kappa_grid <= 0):
            raise ValueError("kappa_grid debe ser finito y > 0.")

        lcv = np.empty(kappa_grid.shape[0])
        for j, k in enumerate(kappa_grid):
            # K[i,i] = exp(k*1)/(2π·I0) es constante; restamos diagonal del row sum
            row_sums = np.exp(k * cos_delta).sum(axis=1)
            diag_val = math.exp(k)                          # cos(0) = 1
            f_loo = (row_sums - diag_val) / ((n - 1) * 2.0 * math.pi * float(_bessel_i0(k)))
            lcv[j] = np.log(np.maximum(f_loo, eps)).sum()

        kappa = float(kappa_grid[int(np.argmax(lcv))])

    if not (np.isfinite(kappa) and kappa > 0):
        raise ValueError("kappa debe ser > 0.")

    # Densidad final (con todos los puntos, sin LOO)
    fhat = np.exp(kappa * cos_delta).sum(axis=1) / (n * 2.0 * math.pi * float(_bessel_i0(kappa)))
    return _normalize_weights_to_n(1.0 / np.maximum(fhat, eps))


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 2 — Fase Mobius estable
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _mobius_cos_sin_jit(
    t: np.ndarray,
    alpha: float,
    omega: float,
    den_eps: float,
) -> tuple:
    """
    Calcula ``cos(phi)`` y ``sin(phi)`` para la fase Mobius sin singularidades.

    phi(t; alpha, omega) = 2 * atan(omega * tan((t - alpha) / 2))

    Pero en vez de calcular ``phi`` explicitamente (que diverge cuando
    ``t - alpha = pi``), se calculan ``cos(phi)`` y ``sin(phi)`` directamente
    via identidades del angulo doble:

        u = (t - alpha) / 2
        den = cos^2(u) + omega^2 * sin^2(u)
        C = (cos^2(u) - omega^2 * sin^2(u)) / den
        S = (2 * omega * sin(u) * cos(u)) / den

    El denominador es siempre positivo (suma de cuadrados), nunca diverge.

    Renormaliza ``C^2 + S^2 = 1`` por seguridad numerica.
    """
    n = t.shape[0]
    C = np.empty(n)
    S = np.empty(n)

    if not np.isfinite(omega) or omega <= 0.0:
        # Limite exacto: phi = 0  =>  cos = 1, sin = 0
        for i in range(n):
            C[i] = 1.0
            S[i] = 0.0
        return C, S

    # Acotar omega a [0, 1]
    if omega > 1.0:
        omega = 1.0

    omega2 = omega * omega
    two_pi = 2.0 * math.pi

    for i in range(n):
        # Envolver t y alpha
        ti = t[i] % two_pi
        ai = alpha % two_pi
        u = 0.5 * (ti - ai)
        s = math.sin(u)
        c = math.cos(u)

        den = c * c + omega2 * s * s + den_eps
        C_i = (c * c - omega2 * s * s) / den
        S_i = (2.0 * omega * s * c) / den

        # Renormalizar a modulo 1
        r = math.sqrt(C_i * C_i + S_i * S_i) + den_eps
        C[i] = C_i / r
        S[i] = S_i / r

    return C, S


def _mobius_cos_sin(
    t: np.ndarray,
    alpha: float,
    omega: float,
    den_eps: float = np.finfo(float).eps,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper publico de ``_mobius_cos_sin_jit``.

    Devuelve ``(C, S)`` con ``C[i] = cos(phi(t[i]; alpha, omega))`` y analogamente ``S``.
    """
    t_arr = np.ascontiguousarray(t, dtype=np.float64)
    return _mobius_cos_sin_jit(t_arr, float(alpha), float(omega), float(den_eps))


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 3 — Estructuras de datos publicas
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FMMGrid:
    """
    Rejilla precomputada de cantidades trigonometricas y de rugosidad.

    Se construye una sola vez por dataset (via ``precompute_grid``) y se
    reutiliza para el ajuste de todos los genes y para la calibracion de
    lambda.

    Attributes
    ----------
    times : np.ndarray, shape (n,)
        Puntos temporales originales (en radianes, [0, 2pi)).

    alpha_grid : np.ndarray, shape (nA,)
    omega_grid : np.ndarray, shape (nW,)
        Rejillas de valores para los parametros no lineales.

    alpha_vec : np.ndarray, shape (nG,)
    omega_vec : np.ndarray, shape (nG,)
        Aplanados con ``nG = nA * nW``: valor (alpha, omega) en cada punto.

    C, S : np.ndarray, shape (nG, n)
        Cantidades ``cos(phi)`` y ``sin(phi)`` para cada punto de rejilla y
        cada muestra. Calculadas via ``_mobius_cos_sin``.

    C2, S2, CS : np.ndarray, shape (nG, n)
        Productos elementales ``C*C``, ``S*S``, ``C*S``. Precomputados para
        evitar recalcularlos en cada ajuste.

    roughness : np.ndarray, shape (nG,)
        Rugosidad de curvatura ``Sum(ddC^2) + Sum(ddS^2)`` por punto de rejilla,
        sobre los tiempos observados.

    dense_roughness : np.ndarray or None, shape (nG,)
        Rugosidad calculada en una rejilla temporal densa uniforme.
        None si no se ha llamado a ``add_dense_roughness``.

    n_dense : int or None
        Tamano de la rejilla densa, o None.

    order : np.ndarray, shape (n,)
        Permutacion que ordena los tiempos circularmente.

    dt : np.ndarray, shape (n,)
        Gaps circulares entre muestras consecutivas (la ultima envuelve).
    """
    times:           np.ndarray
    alpha_grid:      np.ndarray
    omega_grid:      np.ndarray
    alpha_vec:       np.ndarray
    omega_vec:       np.ndarray
    C:               np.ndarray
    S:               np.ndarray
    C2:              np.ndarray
    S2:              np.ndarray
    CS:              np.ndarray
    roughness:       np.ndarray
    order:           np.ndarray
    dt:              np.ndarray
    dense_roughness: Optional[np.ndarray] = None
    n_dense:         Optional[int]        = None


@dataclass
class FMMCache:
    """
    Cache WLS dependiente de los pesos.

    Para cada punto de la rejilla, almacena la inversa ``(X'WX)^{-1}`` con
    ``X = [1, C, S]`` y los pesos fijos. Esto permite resolver la WLS exacta
    en cualquier punto de la rejilla sin recalcular Cholesky cada vez.

    Attributes
    ----------
    weights : np.ndarray, shape (n,)
        Pesos normalizados ``sum(w) = n``.

    inv_XWX : np.ndarray, shape (nG, 3, 3)
        Inversa de la matriz normal ponderada para cada punto de rejilla.
    """
    weights: np.ndarray
    inv_XWX: np.ndarray


@dataclass
class FMMPreparedDataset:
    """
    Bundle completo para ajuste estabilizado.

    Se crea **una sola vez** por dataset (via ``prepare_dataset``) y se
    reutiliza para todos los ajustes posteriores de genes.

    Attributes
    ----------
    times : np.ndarray, shape (n,)
        Puntos temporales (radianes en [0, 2pi)).

    weights : np.ndarray, shape (n,)
        Pesos normalizados a ``sum(w) = n``. Pueden venir del usuario o
        haberse calculado por densidad von Mises.

    grid : FMMGrid
        Rejilla precomputada.

    cache : FMMCache
        Cache WLS para los pesos.

    lambda_star : float
        Fuerza de la penalizacion de rugosidad. Calibrada por Monte Carlo
        bajo H0, o proporcionada por el usuario.

    omega0 : float
        Parametro del factor de penalizacion ``g(omega) = (omega0/(omega+omega0))^p``.

    penalty_power : float
        Exponente ``p`` del factor de penalizacion.
    """
    times:         np.ndarray
    weights:       np.ndarray
    grid:          FMMGrid
    cache:         FMMCache
    lambda_star:   float
    omega0:        float = 0.10
    penalty_power: float = 3.0


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 4 — Precomputos (rejilla + cache)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _fill_basis_jit(
    times:     np.ndarray,
    alpha_vec: np.ndarray,
    omega_vec: np.ndarray,
    den_eps:   float,
):
    """Rellena la base ``C, S`` de forma ``(nG, n)`` para todos los puntos de la rejilla.

    Bucle plano JIT sobre ``(g, i)``. Equivale a llamar ``_mobius_cos_sin_jit``
    nG veces, pero sin overhead de funcion y permitiendo SIMD/loop fusion.
    """
    nG = alpha_vec.shape[0]
    n  = times.shape[0]
    C = np.empty((nG, n))
    S = np.empty((nG, n))
    two_pi = 2.0 * math.pi

    for g in range(nG):
        alpha = alpha_vec[g]
        omega = omega_vec[g]

        if not np.isfinite(omega) or omega <= 0.0:
            for i in range(n):
                C[g, i] = 1.0
                S[g, i] = 0.0
            continue
        if omega > 1.0:
            omega = 1.0

        omega2 = omega * omega
        ai = alpha % two_pi

        for i in range(n):
            ti = times[i] % two_pi
            u = 0.5 * (ti - ai)
            s = math.sin(u)
            c = math.cos(u)

            den = c * c + omega2 * s * s + den_eps
            C_i = (c * c - omega2 * s * s) / den
            S_i = (2.0 * omega * s * c) / den

            r = math.sqrt(C_i * C_i + S_i * S_i) + den_eps
            C[g, i] = C_i / r
            S[g, i] = S_i / r

    return C, S


def _curvature_roughness(
    C:     np.ndarray,
    S:     np.ndarray,
    dt:    np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Rugosidad de curvatura por punto de rejilla:  ``Sum(ddC^2) + Sum(ddS^2)``.

    Discretiza ``int |C''(t)|^2 dt + int |S''(t)|^2 dt`` mediante diferencias
    finitas circulares: pendiente local entre vecinos consecutivos, y luego
    diferencia de pendientes (segunda derivada discreta).

    Vectorizado con ``np.roll`` para los desplazamientos circulares.
    """
    C_ord = C[:, order]
    S_ord = S[:, order]
    dt_b  = dt[np.newaxis, :]

    slope_C = (np.roll(C_ord, -1, axis=1) - C_ord) / dt_b
    slope_S = (np.roll(S_ord, -1, axis=1) - S_ord) / dt_b

    ddC = slope_C - np.roll(slope_C, 1, axis=1)
    ddS = slope_S - np.roll(slope_S, 1, axis=1)

    return (ddC * ddC).sum(axis=1) + (ddS * ddS).sum(axis=1)


def precompute_grid(
    times:      np.ndarray,
    alpha_grid: Optional[np.ndarray] = None,
    omega_grid: Optional[np.ndarray] = None,
) -> FMMGrid:
    """Precomputa la base trigonometrica y la rugosidad para la rejilla.

    Para cada par ``(alpha_g, omega_g)`` de la rejilla:

    - ``C[g, i], S[g, i]``: coseno y seno de la fase Mobius en cada muestra.
    - ``C2, S2, CS``: productos elementales (precomputados para WLS rapida).
    - ``roughness[g]``: rugosidad de curvatura sobre los tiempos observados.

    Tambien se almacenan ``order`` y ``dt`` (orden circular y gaps entre
    muestras consecutivas) para reutilizar en la calibracion de lambda.

    Parameters
    ----------
    times : np.ndarray, shape (n,)
        Puntos temporales en radianes (se envuelve a ``[0, 2*pi)``).
    alpha_grid : np.ndarray, opcional
        Rejilla de alpha. Por defecto: 48 valores en ``[0, 2*pi)``.
    omega_grid : np.ndarray, opcional
        Rejilla de omega. Por defecto: 30 valores log-espaciados en ``[1e-4, 1]``.
    """
    times = np.ascontiguousarray(times, dtype=np.float64) % (2.0 * np.pi)
    n = times.shape[0]
    if n < 4:
        raise ValueError("Se necesitan al menos 4 muestras para la rugosidad.")

    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    if omega_grid is None:
        omega_grid = np.exp(np.linspace(np.log(1e-4), np.log(1.0), 30))

    alpha_grid = np.asarray(alpha_grid, dtype=np.float64) % (2.0 * np.pi)
    omega_grid = np.clip(np.asarray(omega_grid, dtype=np.float64), 0.0, 1.0)
    if np.any(omega_grid <= 0):
        raise ValueError("omega_grid debe contener valores en (0, 1].")

    # Aplanado de la rejilla a vectores de longitud nG = nA * nW
    A_mesh, W_mesh = np.meshgrid(alpha_grid, omega_grid, indexing="ij")
    alpha_vec = A_mesh.ravel()
    omega_vec = W_mesh.ravel()

    # Base trigonometrica para todos los puntos (kernel JIT)
    C, S = _fill_basis_jit(times, alpha_vec, omega_vec, float(np.finfo(float).eps))

    # Productos elementales (se reusan para la WLS y para la potencia)
    C2 = C * C
    S2 = S * S
    CS = C * S

    # Orden circular y gaps entre muestras consecutivas
    order = np.argsort(times)
    t_ord = times[order]
    dt = np.concatenate([np.diff(t_ord),
                         np.array([(t_ord[0] + 2.0 * np.pi) - t_ord[-1]])])
    dt = np.maximum(dt, np.finfo(float).eps)

    # Rugosidad por punto de rejilla (sobre tiempos observados)
    roughness = _curvature_roughness(C, S, dt, order)

    return FMMGrid(
        times      = times,
        alpha_grid = alpha_grid,
        omega_grid = omega_grid,
        alpha_vec  = alpha_vec,
        omega_vec  = omega_vec,
        C          = C,
        S          = S,
        C2         = C2,
        S2         = S2,
        CS         = CS,
        roughness  = roughness,
        order      = order,
        dt         = dt,
    )


def add_dense_roughness(grid: FMMGrid, n_dense: int = 512) -> FMMGrid:
    """Anade rugosidad calculada en rejilla temporal densa uniforme al ``grid``.

    Cuando los tiempos observados son escasos o irregulares, la rugosidad
    sobre tiempos observados es una aproximacion ruidosa. Sobre una rejilla
    uniforme densa de ``n_dense`` puntos, la rugosidad refleja mejor la
    suavidad intrinseca de la base ``(C, S)``.

    El campo ``grid.dense_roughness`` queda relleno; ``rough_mode='dense'``
    en ``fit_gene`` lo usara automaticamente.
    """
    if n_dense < 16:
        raise ValueError("n_dense debe ser >= 16.")

    # Rejilla temporal uniforme y dt constantes
    t_dense = np.linspace(0.0, 2.0 * np.pi, n_dense, endpoint=False)
    dt = np.full(n_dense, 2.0 * np.pi / n_dense)
    order = np.arange(n_dense)

    C_dense, S_dense = _fill_basis_jit(
        t_dense, grid.alpha_vec, grid.omega_vec, float(np.finfo(float).eps),
    )
    grid.dense_roughness = _curvature_roughness(C_dense, S_dense, dt, order)
    grid.n_dense = int(n_dense)
    return grid


def prepare_cache(
    weights: np.ndarray,
    grid:    FMMGrid,
    ridge:   float = 1e-8,
) -> FMMCache:
    """Calcula la inversa ``(X'WX)^{-1}`` para cada punto de rejilla.

    Construye, para cada ``g``, la matriz normal ponderada::

        X = [1 | C[g] | S[g]]   (n x 3)
        X'WX = [ sum(w)        sum(w*C)      sum(w*S)     ]
               [ sum(w*C)      sum(w*C^2)    sum(w*C*S)   ]
               [ sum(w*S)      sum(w*C*S)    sum(w*S^2)   ]

    e invierte con ``np.linalg.inv`` el tensor ``(nG, 3, 3)`` completo de
    una sola pasada. Una pequena regularizacion ``ridge`` se anade a la
    diagonal para estabilidad numerica.

    Parameters
    ----------
    weights : np.ndarray, shape (n,)
        Pesos (se normalizan internamente a ``sum(w) = n``).
    grid : FMMGrid
        Rejilla precomputada.
    ridge : float
        Regularizacion diagonal para estabilidad.
    """
    w = _normalize_weights_to_n(weights)
    if w.shape[0] != grid.times.shape[0]:
        raise ValueError("weights debe tener la misma longitud que grid.times.")

    nG = grid.C.shape[0]

    # Productos elementales con los pesos -> sumas escalares y por punto de rejilla
    a11 = float(w.sum())
    a12 = grid.C  @ w     # (nG,)
    a13 = grid.S  @ w     # (nG,)
    a22 = grid.C2 @ w     # (nG,)
    a23 = grid.CS @ w     # (nG,)
    a33 = grid.S2 @ w     # (nG,)

    # Tensor (nG, 3, 3) de matrices normales ponderadas + ridge
    A = np.empty((nG, 3, 3), dtype=np.float64)
    A[:, 0, 0] = a11 + ridge
    A[:, 0, 1] = a12; A[:, 1, 0] = a12
    A[:, 0, 2] = a13; A[:, 2, 0] = a13
    A[:, 1, 1] = a22 + ridge
    A[:, 1, 2] = a23; A[:, 2, 1] = a23
    A[:, 2, 2] = a33 + ridge

    # Invertir todo el tensor de una vez (NumPy/LAPACK vectorizado)
    inv_XWX = np.linalg.inv(A)

    return FMMCache(weights=w, inv_XWX=inv_XWX)


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 5 — Ajuste por gen (criterio de potencia + penalizacion)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_power_scores(
    data:          np.ndarray,
    grid:          FMMGrid,
    weights:       np.ndarray,
    lambda_:       float,
    omega0:        float,
    penalty_power: float,
    rough_mode:    str,
) -> np.ndarray:
    """Calcula el criterio  ``J(alpha, omega) = P - lambda*g(omega)*R``  para toda la rejilla.

    - ``P(alpha, omega) = (u^2 + v^2) / den``   (potencia normalizada Lomb-Scargle)
      con  ``u = sum(w*x_centered*C)``,  ``v = sum(w*x_centered*S)``,
      ``den = sum(w*C^2) + sum(w*S^2)``.
    - ``g(omega) = (omega0 / (omega + omega0))^p``
    - ``R(alpha, omega)`` = rugosidad de curvatura (``grid.roughness`` u
      ``grid.dense_roughness`` segun ``rough_mode``).

    Operaciones todas vectorizadas matriz-vector — un solo gen cuesta O(nG*n).
    """
    # Centrar los datos por la media ponderada (asi el intercepto desaparece de la ecuacion)
    mu_w = float((weights * data).sum() / weights.sum())
    wx0 = weights * (data - mu_w)

    # Potencia: u, v para todos los puntos a la vez
    u = grid.C @ wx0          # (nG,)
    v = grid.S @ wx0          # (nG,)
    pow_ = u * u + v * v

    # Normalizacion Lomb-Scargle: dividir por la norma de la base ponderada
    den = grid.C2 @ weights + grid.S2 @ weights
    den = np.where(den > 0, den, np.nan)
    pow_ = pow_ / den

    if lambda_ <= 0:
        return pow_

    # Penalizacion
    if rough_mode == "dense":
        if grid.dense_roughness is None:
            raise ValueError("rough_mode='dense' pero grid.dense_roughness es None.")
        roughness = grid.dense_roughness
    elif rough_mode == "observed":
        roughness = grid.roughness
    else:
        raise ValueError(f"rough_mode desconocido: {rough_mode}")

    g_omega = (omega0 / (grid.omega_vec + omega0)) ** penalty_power
    return pow_ - lambda_ * g_omega * roughness


def _wls_fit_at(
    data:    np.ndarray,
    alpha:   float,
    omega:   float,
    weights: np.ndarray,
    times:   np.ndarray,
    ridge:   float = 1e-10,
) -> dict:
    """Resuelve la WLS exacta 3x3 en un punto ``(alpha, omega)`` arbitrario.

    Construye la base ``[1, C, S]``, resuelve  ``(X'WX) coef = X'Wx``, y devuelve
    los parametros del FMM con  ``A = sqrt(b^2 + g^2)``,  ``beta = atan2(-g, b)``.

    Devuelve dict con ``M, A, alpha, beta, omega, fitted, rss``.
    """
    C, S = _mobius_cos_sin(times, alpha, omega)

    # Sistema normal ponderado 3x3
    a11 = weights.sum() + ridge
    a12 = (weights * C).sum()
    a13 = (weights * S).sum()
    a22 = (weights * C * C).sum() + ridge
    a23 = (weights * C * S).sum()
    a33 = (weights * S * S).sum() + ridge

    A = np.array([[a11, a12, a13],
                  [a12, a22, a23],
                  [a13, a23, a33]])
    b = np.array([(weights * data).sum(),
                  (weights * data * C).sum(),
                  (weights * data * S).sum()])

    coef = np.linalg.solve(A, b)
    M, b_coef, g_coef = coef

    fitted = M + b_coef * C + g_coef * S
    rss = float((weights * (data - fitted) ** 2).sum())

    A_amp = float(np.sqrt(b_coef ** 2 + g_coef ** 2))
    beta  = float(np.arctan2(-g_coef, b_coef) % (2.0 * np.pi))

    return dict(
        M = float(M), A = A_amp, alpha = float(alpha) % (2.0 * np.pi),
        beta = beta, omega = float(omega),
        fitted = fitted, rss = rss,
    )


def _local_refine(
    alpha:           float,
    omega:           float,
    score_init:      float,
    data:            np.ndarray,
    grid:            FMMGrid,
    weights:         np.ndarray,
    lambda_:         float,
    omega0:          float,
    penalty_power:   float,
    rough_mode:      str,
    n_steps:         int   = 3,
    alpha_step:      Optional[float] = None,
    omega_step_mult: float = 1.25,
    omega_min:       float = 1e-4,
    omega_max:       float = 1.0,
) -> tuple[float, float]:
    """Refinamiento local en 8 direcciones con halveo de paso (alpha, omega).

    En cada iteracion evalua 8 candidatos ``(alpha +/- da, omega * o_mult^{0,+1,-1})``
    y escoge el de mayor score. Si ninguno mejora, halvea ``da`` y reduce ``o_mult``.

    El score se evalua off-grid recomputando ``C, S, R`` con ``_mobius_cos_sin``.
    """
    if alpha_step is None:
        alpha_step = (2.0 * np.pi) / max(48, len(grid.alpha_grid))
    da = float(alpha_step)
    om = float(omega_step_mult)

    # Centrado para potencia
    mu_w = float((weights * data).sum() / weights.sum())
    wx0 = weights * (data - mu_w)

    # Roughness en rejilla densa: precalculada en grid; off-grid la recalculamos al vuelo
    if rough_mode == "dense":
        # Rejilla temporal densa cacheada en el grid (si add_dense_roughness lo creo)
        if grid.n_dense is None:
            raise ValueError("rough_mode='dense' requiere add_dense_roughness aplicado.")
        n_d = grid.n_dense
        t_dense = np.linspace(0.0, 2.0 * np.pi, n_d, endpoint=False)
        dt_d = np.full(n_d, 2.0 * np.pi / n_d)
        order_d = np.arange(n_d)
        rough_times = t_dense
        rough_dt = dt_d
        rough_order = order_d
    else:
        rough_times = grid.times
        rough_dt = grid.dt
        rough_order = grid.order

    def score_at(a: float, o: float) -> float:
        a = a % (2.0 * np.pi)
        o = max(min(o, omega_max), omega_min)
        # Potencia
        C, S = _mobius_cos_sin(grid.times, a, o)
        u = float((wx0 * C).sum())
        v = float((wx0 * S).sum())
        den = float((weights * C * C).sum() + (weights * S * S).sum())
        if den <= 0:
            return -np.inf
        score = (u * u + v * v) / den
        if lambda_ <= 0:
            return score
        # Rugosidad
        Cr, Sr = _mobius_cos_sin(rough_times, a, o)
        Cr_o = Cr[rough_order]; Sr_o = Sr[rough_order]
        slope_C = (np.roll(Cr_o, -1) - Cr_o) / rough_dt
        slope_S = (np.roll(Sr_o, -1) - Sr_o) / rough_dt
        ddC = slope_C - np.roll(slope_C, 1)
        ddS = slope_S - np.roll(slope_S, 1)
        R = float((ddC * ddC).sum() + (ddS * ddS).sum())
        g = (omega0 / (o + omega0)) ** penalty_power
        return score - lambda_ * g * R

    score_cur = float(score_init)
    a_cur, o_cur = float(alpha), float(omega)

    for _ in range(n_steps):
        # 8 candidatos: 4 cardinales + 4 diagonales
        cands = [
            (a_cur + da, o_cur),
            (a_cur - da, o_cur),
            (a_cur, o_cur * om),
            (a_cur, o_cur / om),
            (a_cur + da, o_cur * om),
            (a_cur - da, o_cur * om),
            (a_cur + da, o_cur / om),
            (a_cur - da, o_cur / om),
        ]
        scores = np.array([score_at(a, o) for a, o in cands])
        if not np.any(np.isfinite(scores)):
            da *= 0.5; om = math.sqrt(om); continue
        idx = int(np.nanargmax(scores))
        if scores[idx] > score_cur:
            a_cur, o_cur = cands[idx]
            a_cur = a_cur % (2.0 * np.pi)
            o_cur = max(min(o_cur, omega_max), omega_min)
            score_cur = float(scores[idx])
        else:
            da *= 0.5
            om = math.sqrt(om)

    return a_cur, o_cur


def fit_gene(
    data:       np.ndarray,
    prepared:   FMMPreparedDataset,
    refine:     bool = True,
    rough_mode: str  = "dense",
) -> FitResult:
    """Ajuste FMM estabilizado para un gen.

    Algoritmo:
      1. Calcula ``J(alpha, omega) = P - lambda*g(omega)*R`` para toda la rejilla.
      2. Selecciona el ``(alpha*, omega*)`` con maximo score.
      3. (Opcional) Refinamiento local 2D con halveo de paso.
      4. WLS exacta 3x3 en ``(alpha*, omega*)`` para obtener ``M, A, beta``.
      5. Construye ``FitResult`` compatible con ``FMMModel``.

    El ``model_name`` del resultado es ``"fmm_stabilized"`` (vs ``"fmm"`` en el base),
    por si se quiere distinguir mas adelante. Las claves de ``params`` son las
    mismas (M, A, alpha, beta, omega) y la formula de ``peak_time`` es identica.
    """
    data = np.asarray(data, dtype=float)
    if data.shape[0] != prepared.times.shape[0]:
        raise ValueError("data debe tener la misma longitud que prepared.times.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data contiene valores no finitos.")

    grid    = prepared.grid
    weights = prepared.cache.weights
    lambda_ = float(prepared.lambda_star)
    omega0  = float(prepared.omega0)
    p       = float(prepared.penalty_power)

    # 1. Score sobre toda la rejilla
    scores = _compute_power_scores(
        data, grid, weights, lambda_, omega0, p, rough_mode,
    )
    scores_safe = np.where(np.isfinite(scores), scores, -np.inf)
    if not np.any(np.isfinite(scores_safe)):
        raise RuntimeError("Ningun score finito en la rejilla.")

    # 2. Seleccion con desempate por omega mayor entre candidatos cercanos
    sc_max = float(scores_safe.max())
    close = np.where(scores_safe >= sc_max - 1e-2)[0]
    # Entre los cercanos, escoger el de omega mas grande (forma mas suave)
    idx = int(close[int(np.argmax(grid.omega_vec[close]))])
    alpha = float(grid.alpha_vec[idx])
    omega = float(grid.omega_vec[idx])

    # 3. Refinamiento local 2D opcional
    if refine:
        alpha, omega = _local_refine(
            alpha, omega, scores_safe[idx],
            data, grid, weights, lambda_, omega0, p, rough_mode,
        )

    # 4. WLS final exacta en el punto seleccionado
    fit = _wls_fit_at(data, alpha, omega, weights, prepared.times)

    # 5. Construir FitResult (mismo formato que FMMModel)
    fitted = fit["fitted"]
    residuals = data - fitted
    # Ajustamos std residuos con ddof=5 (mismo que FMMModel)
    res_std = residuals.std(ddof=5)
    residuals_std = (residuals - residuals.mean()) / res_std if res_std > 0 \
                    else np.zeros_like(residuals)
    sse = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((data - data.mean()) ** 2))
    r2 = float(1.0 - sse / ss_tot) if ss_tot > 0 else 0.0

    peak_t = float(
        (fit["alpha"] + 2.0 * np.arctan2(
            (1.0 / fit["omega"]) * np.sin(-fit["beta"] / 2.0),
            np.cos(-fit["beta"] / 2.0),
        )) % (2.0 * np.pi)
    )

    return FitResult(
        fitted        = fitted,
        params        = {
            "M":     fit["M"],
            "A":     fit["A"],
            "alpha": fit["alpha"],
            "beta":  fit["beta"],
            "omega": fit["omega"],
        },
        peak_time     = peak_t,
        r2            = r2,
        residuals     = residuals,
        residuals_std = residuals_std,
        sse           = sse,
        model_name    = "fmm_stabilized",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 6 — Calibracion de lambda
# ═══════════════════════════════════════════════════════════════════════════

def _spike_max_jump(y: np.ndarray, order: np.ndarray) -> float:
    """
    Salto maximo entre valores consecutivos en orden circular.

    Diagnostico de sobreajuste para calibrar lambda.
    """
    raise NotImplementedError


def _pick_lambda_elbow(
    lambda_grid: np.ndarray,
    q_spike:     np.ndarray,
    smooth_k:    int  = 3,
    use_log10:   bool = True,
) -> float:
    """
    Detecta el codo de la curva ``q_spike(lambda)`` por distancia a la cuerda.

    """
    raise NotImplementedError


def calibrate_lambda(
    times:         np.ndarray,
    weights:       np.ndarray,
    grid:          FMMGrid,
    cache:         FMMCache,
    lambda_grid:   Optional[np.ndarray] = None,
    n_simulations: int    = 500,
    quantile:      float  = 0.99,
    sigma:         float  = 1.0,
    omega0:        float  = 0.10,
    penalty_power: float  = 3.0,
    rough_mode:    str    = "dense",
    seed:          int    = 1,
    smooth_k:      int    = 3,
    center_null:   bool   = True,
) -> dict:
    """
    Calibracion de lambda por Monte Carlo bajo H0 + codo.
    """
    raise NotImplementedError


def calibrate_lambda_repeat(
    n_repeats: int = 5,
    seed0:     int = 1,
    agg:       str = "median",
    **kwargs,
) -> float:
    """
    Repite la calibracion ``n_repeats`` veces y agrega via mediana/media.
    """
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 7 — Funcion publica de preparacion
# ═══════════════════════════════════════════════════════════════════════════

def prepare_dataset(
    times:             np.ndarray,
    weights:           Optional[np.ndarray] = None,
    lambda_star:       Optional[float]      = None,
    length_alpha_grid: int   = 48,
    length_omega_grid: int   = 30,
    omega_max:         float = 1.0,
    omega0:            float = 0.10,
    penalty_power:     float = 3.0,
    n_dense:           int   = 512,
    n_simulations:     int   = 500,
    n_repeats:         int   = 5,
    quantile:          float = 0.99,
    seed:              int   = 1,
    rough_mode:        str   = "dense",
) -> FMMPreparedDataset:
    """
    Preparacion completa del dataset para ajuste estabilizado.

    Llamar UNA SOLA VEZ por dataset. El ``FMMPreparedDataset`` resultante
    se reutiliza para todos los genes.

    Pasos:

    1. Si ``weights`` es None: calcular pesos de densidad temporal von Mises
       (kappa por LOO-CV).
    2. Construir rejillas alpha/omega (defaults equivalentes a R).
    3. ``precompute_grid`` + ``add_dense_roughness``.
    4. ``prepare_cache`` con los pesos.
    5. Si ``lambda_star`` es None: ``calibrate_lambda_repeat``.
    """
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Seccion 8 — Clase modelo (interfaz RhythmModel)
# ═══════════════════════════════════════════════════════════════════════════

class StabilizedFMMModel(RhythmModel):
    """
    FMM estabilizado con criterio de potencia + penalizacion de rugosidad.

    A diferencia del FMM base (``FMMModel``), este modelo:

    - Usa pesos por densidad temporal von Mises
    - Selecciona (alpha, omega) por criterio de potencia penalizado
    - Calibra lambda automaticamente bajo H0
    - Es sustancialmente mas rapido en batch (preparacion compartida)

    El ``FitResult`` resultante es **compatible** con el de ``FMMModel``:
    misma estructura de ``params`` (M, A, alpha, beta, omega) y misma formula
    de ``peak_time``. Es drop-in en el resto del pipeline.

    Modos de uso:

    1. **Eficiente (recomendado)**: el usuario crea el ``FMMPreparedDataset``
       una vez y lo pasa al constructor::

           prepared = prepare_dataset(times)
           model = StabilizedFMMModel(prepared)
           for gene in genes:
               result = model.fit(data[gene], times)

    2. **Lazy (drop-in)**: la clase prepara automaticamente al primer fit
       y cachea el resultado mientras los tiempos no cambien::

           model = StabilizedFMMModel()
           for gene in genes:
               result = model.fit(data[gene], times)   # primera prep + cache

    Parameters
    ----------
    prepared : FMMPreparedDataset, opcional
        Si se proporciona, se usa directamente. Si es None, se prepara
        de forma lazy en el primer ``fit()``.
    refine : bool
        Si aplicar refinamiento local 2D tras la seleccion en rejilla.
    rough_mode : str
        ``"dense"`` (recomendado) o ``"observed"``.
    **prep_kwargs
        Parametros extra para ``prepare_dataset`` (solo en modo lazy).
    """

    def __init__(
        self,
        prepared:   Optional[FMMPreparedDataset] = None,
        refine:     bool = True,
        rough_mode: str  = "dense",
        **prep_kwargs,
    ) -> None:
        self.prepared    = prepared
        self.refine      = refine
        self.rough_mode  = rough_mode
        self._prep_kwargs = prep_kwargs
        self._cached_times: Optional[np.ndarray] = None

    @staticmethod
    def peak_time(alpha: float, beta: float, omega: float) -> float:
        """
        Tiempo de pico — formula identica a ``FMMModel.peak_time``.

        t_U = alpha + 2*atan2((1/omega)*sin(-beta/2), cos(-beta/2))  mod 2*pi
        """
        return float(
            (alpha + 2.0 * np.arctan2(
                (1.0 / omega) * np.sin(-beta / 2.0),
                np.cos(-beta / 2.0),
            )) % (2.0 * np.pi)
        )

    def fit(
        self,
        data:        np.ndarray,
        time_points: np.ndarray,
    ) -> FitResult:
        """
        Ajusta el FMM estabilizado a ``data``.

        Si no se ha proporcionado ``prepared`` en el constructor, se prepara
        en la primera llamada y se cachea. Si los ``time_points`` cambian
        respecto a la preparacion previa, se vuelve a preparar.
        """
        # Preparar (lazy) si no hay prepared cacheado o cambian los tiempos
        if self.prepared is None or self._cached_times is None \
                or not np.array_equal(self._cached_times, time_points):
            self.prepared    = prepare_dataset(time_points, **self._prep_kwargs)
            self._cached_times = np.asarray(time_points, dtype=float).copy()

        return fit_gene(
            data       = data,
            prepared   = self.prepared,
            refine     = self.refine,
            rough_mode = self.rough_mode,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory simple para selector base/estabilizado en el pipeline
# ═══════════════════════════════════════════════════════════════════════════

def make_fmm_model(method: str = "base", **kwargs) -> RhythmModel:
    """
    Factory para seleccionar entre FMM base y estabilizado.

    Parameters
    ----------
    method : str
        - ``"base"``       : retorna ``FMMModel`` (fmm.py, RSS + Nelder-Mead)
        - ``"stabilized"`` : retorna ``StabilizedFMMModel`` (fmm_stabilized.py)
    **kwargs
        Argumentos del constructor del modelo elegido.
    """
    if method == "base":
        from circust.fitting.fmm import FMMModel
        return FMMModel(**kwargs)
    elif method == "stabilized":
        return StabilizedFMMModel(**kwargs)
    else:
        raise ValueError(
            f"Unknown fmm_method='{method}'. Use 'base' or 'stabilized'."
        )
