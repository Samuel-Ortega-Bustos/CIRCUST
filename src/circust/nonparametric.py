"""
circust/nonparametric.py
========================
Puntuacion de ritmicidad no parametrica mediante regresion isotonica
circular unimodal.

Etapa 3.1 de CIRCUST: para cada gen, ajusta una regresion isotonica
*subida-bajada* (un pico, un valle) al perfil de expresion ordenado por
el tiempo circular preliminar. Compara contra un modelo plano (media)
para obtener R².

El metodo usa el Algoritmo Pool-Adjacent-Violators (PAVA) para imponer
restricciones de monotonia en segmentos circulares, buscando sobre todos
los pares candidatos valle-pico para encontrar la forma unimodal de
mejor ajuste.

Equivalente en R
-----------------
``computeNP()``  (lineas 13–30 de ``functionGTEX_cores.R``) que llama a
``function1Local_modif()`` → ``busquedaMejor()`` → ``pavaC()`` de
``NucleoComun.R`` y ``upDownUp_NP_Code_*.R``.

Posicion en el pipeline
-----------------------
    CircularSynchronizer  →  **NonParametricScorer**  →  CandidateSelector
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# ── Aceleracion opcional con Numba ─────────────────────────────────────────
# Si Numba esta disponible, compilamos el bucle interno de PAVA y las rutinas
# auxiliares de circular_unimodal_fit a codigo maquina. Esto acelera la
# Etapa 3.1 ~20-50x porque el bucle (L, U) x PAVA es el cuello de botella
# dominante del pipeline (O(n^3) por gen).
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):                          # type: ignore
        def deco(fn): return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return deco


@njit(cache=True, fastmath=True)
def _pava_inc_numba(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Pool-Adjacent-Violators creciente compilado JIT.

    Acepta pesos arbitrarios *w*. Para pesos uniformes el llamador pasa
    ``np.ones(n)``. Minimiza  Σ wᵢ (yᵢ − ŷᵢ)²  s.a.  ŷ₁ ≤ … ≤ ŷₙ.
    """
    n = y.shape[0]
    if n <= 1:
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = y[i]
        return out

    block_sum = np.empty(n, dtype=np.float64)
    block_w   = np.empty(n, dtype=np.float64)
    block_end = np.empty(n, dtype=np.int64)
    nb = 0

    for i in range(n):
        block_sum[nb] = y[i] * w[i]
        block_w[nb]   = w[i]
        block_end[nb] = i
        nb += 1
        while nb >= 2:
            avg_prev = block_sum[nb - 2] / block_w[nb - 2]
            avg_curr = block_sum[nb - 1] / block_w[nb - 1]
            if avg_prev <= avg_curr:
                break
            block_sum[nb - 2] += block_sum[nb - 1]
            block_w[nb - 2]   += block_w[nb - 1]
            block_end[nb - 2]  = block_end[nb - 1]
            nb -= 1

    out = np.empty(n, dtype=np.float64)
    start = 0
    for b in range(nb):
        end = block_end[b] + 1
        avg = block_sum[b] / block_w[b]
        for k in range(start, end):
            out[k] = avg
        start = end
    return out


@njit(cache=True, fastmath=True)
def _pava_inc_range(buf: np.ndarray, out: np.ndarray, m: int) -> None:
    """PAVA creciente sobre buf[:m], escribe resultado en out[:m]. Sin alloc."""
    if m <= 0:
        return
    block_sum = np.empty(m, dtype=np.float64)
    block_w   = np.empty(m, dtype=np.float64)
    block_end = np.empty(m, dtype=np.int64)
    nb = 0
    for i in range(m):
        block_sum[nb] = buf[i]
        block_w[nb]   = 1.0
        block_end[nb] = i
        nb += 1
        while nb >= 2:
            ap = block_sum[nb - 2] / block_w[nb - 2]
            ac = block_sum[nb - 1] / block_w[nb - 1]
            if ap <= ac:
                break
            block_sum[nb - 2] += block_sum[nb - 1]
            block_w[nb - 2]   += block_w[nb - 1]
            block_end[nb - 2]  = block_end[nb - 1]
            nb -= 1
    start = 0
    for b in range(nb):
        end = block_end[b] + 1
        avg = block_sum[b] / block_w[b]
        for k in range(start, end):
            out[k] = avg
        start = end


@njit(cache=True, fastmath=True)
def _circular_unimodal_fit_numba(
    v: np.ndarray, candL: np.ndarray, candU: np.ndarray,
):
    """
    Version JIT del bucle (L, U) x PAVA. Devuelve:
        (fitted, mse, L_opt, U_opt, ok)
    ``ok`` es True si se encontro un ajuste valido.
    """
    n = v.shape[0]
    nL = candL.shape[0]
    nU = candU.shape[0]

    best_mse  = 1e300
    best_L    = 0
    best_U    = 0
    found     = False
    best_fit  = np.empty(n, dtype=np.float64)

    # Buffers reutilizables
    ordered_U = np.empty(nU, dtype=np.int64)
    inc_buf   = np.empty(n + 2, dtype=np.float64)   # interior PAVA input
    inc_out   = np.empty(n + 2, dtype=np.float64)   # interior PAVA output
    full_inc  = np.empty(n + 2, dtype=np.float64)   # [v[L], pava, v[U]]
    dec_buf   = np.empty(n + 2, dtype=np.float64)
    dec_out   = np.empty(n + 2, dtype=np.float64)
    valid_Us      = np.empty(nU, dtype=np.int64)
    valid_inc_len = np.empty(nU, dtype=np.int64)
    valid_inc_mat = np.empty((nU, n + 2), dtype=np.float64)
    fitted        = np.empty(n, dtype=np.float64)

    for li in range(nL):
        indL = candL[li]

        # Ordenar candU: primero >= indL, luego < indL
        k = 0
        for j in range(nU):
            if candU[j] >= indL:
                ordered_U[k] = candU[j]; k += 1
        for j in range(nU):
            if candU[j] < indL:
                ordered_U[k] = candU[j]; k += 1

        n_valid = 0
        stop_L  = False

        for uj in range(nU):
            indU = ordered_U[uj]

            # ── _increasing_segment ──
            if indL == indU:
                # ajuste constante
                mean_v = 0.0
                for t in range(n):
                    mean_v += v[t]
                mean_v /= n
                mse = 0.0
                for t in range(n):
                    d = v[t] - mean_v
                    mse += d * d
                mse /= n
                if mse < best_mse:
                    best_mse = mse
                    best_L = indL
                    best_U = indU
                    for t in range(n):
                        best_fit[t] = mean_v
                    found = True
                continue

            if indL < indU:
                k_interior = indU - indL - 1
                if k_interior > 0:
                    for t in range(k_interior):
                        inc_buf[t] = v[indL + 1 + t]
            else:
                k_interior = (n - indL - 1) + indU
                if k_interior > 0:
                    p = 0
                    for t in range(indL + 1, n):
                        inc_buf[p] = v[t]; p += 1
                    for t in range(0, indU):
                        inc_buf[p] = v[t]; p += 1

            inc_len = k_interior + 2
            if k_interior > 0:
                _pava_inc_range(inc_buf, inc_out, k_interior)
                full_inc[0] = v[indL]
                for t in range(k_interior):
                    full_inc[1 + t] = inc_out[t]
                full_inc[k_interior + 1] = v[indU]
            else:
                full_inc[0] = v[indL]
                full_inc[1] = v[indU]

            # Validez
            if inc_len > 2:
                if full_inc[1] <= v[indL]:
                    stop_L = True
                    break
                is_valid_U = full_inc[inc_len - 2] < v[indU]
            else:
                is_valid_U = True

            if is_valid_U:
                valid_Us[n_valid] = indU
                valid_inc_len[n_valid] = inc_len
                for t in range(inc_len):
                    valid_inc_mat[n_valid, t] = full_inc[t]
                n_valid += 1

        # Pasada hacia atras
        for rev in range(n_valid - 1, -1, -1):
            indU    = valid_Us[rev]
            inc_len = valid_inc_len[rev]

            if indL == indU:
                continue  # ya manejado arriba

            # ── _decreasing_segment ──
            if indU < indL:
                k_int = indL - indU - 1
                if k_int > 0:
                    for t in range(k_int):
                        dec_buf[t] = v[indU + 1 + t]
            else:
                k_int = (n - indU - 1) + indL
                if k_int > 0:
                    p = 0
                    for t in range(indU + 1, n):
                        dec_buf[p] = v[t]; p += 1
                    for t in range(0, indL):
                        dec_buf[p] = v[t]; p += 1

            if k_int == 0:
                dec_ok = True
                is_valid_dec = True
            else:
                # pava decreciente = -pava_inc(-buf)
                for t in range(k_int):
                    dec_buf[t] = -dec_buf[t]
                _pava_inc_range(dec_buf, dec_out, k_int)
                for t in range(k_int):
                    dec_out[t] = -dec_out[t]

                if dec_out[k_int - 1] < v[indL]:
                    break  # no mas U's validos
                is_valid_dec = dec_out[0] <= v[indU]
                dec_ok = True

            if not is_valid_dec:
                continue

            # ── _assemble ──
            # arco creciente L..U
            if indL <= indU:
                for t in range(inc_len):
                    fitted[indL + t] = valid_inc_mat[rev, t]
            else:
                p = 0
                for t in range(indL, n):
                    fitted[t] = valid_inc_mat[rev, p]; p += 1
                for t in range(0, indU + 1):
                    fitted[t] = valid_inc_mat[rev, p]; p += 1

            # arco decreciente U+1..L-1 (interior)
            if k_int > 0:
                if indU < indL:
                    for t in range(k_int):
                        fitted[indU + 1 + t] = dec_out[t]
                else:
                    p = 0
                    for t in range(indU + 1, n):
                        fitted[t] = dec_out[p]; p += 1
                    for t in range(0, indL):
                        fitted[t] = dec_out[p]; p += 1

            # MSE
            mse = 0.0
            for t in range(n):
                d = v[t] - fitted[t]
                mse += d * d
            mse /= n

            if mse < best_mse:
                best_mse = mse
                best_L = indL
                best_U = indU
                for t in range(n):
                    best_fit[t] = fitted[t]
                found = True

        if best_mse == 0.0:
            break

    return best_fit, best_mse, best_L, best_U, found


# ═══════════════════════════════════════════════════════════════════════════
# Algoritmo Pool-Adjacent-Violators  (R: pavaC / Iso::pava)
# ═══════════════════════════════════════════════════════════════════════════

def pava_increasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """
    Regresion isotonica creciente mediante el Algoritmo Pool-Adjacent-Violators.

    Dados observaciones *y* con pesos positivos opcionales *w*, devuelve el
    vector ŷ que minimiza  Σ wᵢ (yᵢ − ŷᵢ)²  sujeto a  ŷ₁ ≤ ŷ₂ ≤ … ≤ ŷₙ.

    Es un port directo de la funcion C

    Complejidad: O(n) tiempo, O(n) espacio.

    Parametros
    ----------
    y : array (n,)
        Valores observados.
    w : array (n,) o None
        Pesos positivos. Si es *None*, se usan pesos uniformes.

    Devuelve
    --------
    array (n,) — valores ajustados monotonicamente no decrecientes.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    n = len(y)
    if n <= 1:
        return y.copy()

    w_arr = (np.ones(n, dtype=np.float64) if w is None
             else np.ascontiguousarray(np.asarray(w, dtype=np.float64)))

    if _HAS_NUMBA:
        return _pava_inc_numba(y, w_arr)

    # Fallback Python puro — algoritmo de fusion de bloques.
    # Cada bloque: (suma ponderada acumulada, peso acumulado, indice derecho).
    block_sum = np.empty(n, dtype=np.float64)
    block_w   = np.empty(n, dtype=np.float64)
    block_end = np.empty(n, dtype=np.intp)
    nb = 0

    for i in range(n):
        block_sum[nb] = y[i] * w_arr[i]
        block_w[nb]   = w_arr[i]
        block_end[nb] = i
        nb += 1

        while nb >= 2:
            avg_prev = block_sum[nb - 2] / block_w[nb - 2]
            avg_curr = block_sum[nb - 1] / block_w[nb - 1]
            if avg_prev <= avg_curr:
                break
            block_sum[nb - 2] += block_sum[nb - 1]
            block_w[nb - 2]   += block_w[nb - 1]
            block_end[nb - 2]  = block_end[nb - 1]
            nb -= 1

    result = np.empty(n, dtype=np.float64)
    start = 0
    for b in range(nb):
        end = block_end[b] + 1
        result[start:end] = block_sum[b] / block_w[b]
        start = end

    return result


def pava_decreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """
    Regresion isotonica decreciente.

    Equivalente a negar *y*, ejecutar ``pava_increasing``, y negar de vuelta.
    """
    return -pava_increasing(-np.asarray(y, dtype=np.float64), w)


# ═══════════════════════════════════════════════════════════════════════════
# Deteccion de extremos locales circulares  (R: extremosLocales en NucleoComun.R)
# ═══════════════════════════════════════════════════════════════════════════

def find_local_extrema(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Encuentra minimos locales (valles) y maximos locales (picos) usando
    vecinos circulares.

    Un punto i es un maximo local si  v[i−1] ≤ v[i]  Y  v[i] ≥ v[i+1],
    con indices envolventes circulares. Los minimos locales se definen
    simetricamente.

    Nota sobre el codigo R: los comentarios etiquetan los maximos como
    "minimos locales" y viceversa (un error de copiar-pegar en los
    comentarios en espanol), pero los nombres de variable ``candU``
    (up = max) y ``candL`` (low = min) son correctos.

    Parametros
    ----------
    v : array (n,)

    Devuelve
    --------
    candL : array int 1-D — indices de minimos locales (valles).
    candU : array int 1-D — indices de maximos locales (picos).
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    prev  = np.roll(v, 1)
    next_ = np.roll(v, -1)

    candU = np.where((prev <= v) & (v >= next_))[0]    # local maxima
    candL = np.where((prev >= v) & (v <= next_))[0]    # local minima

    return candL, candU


# ═══════════════════════════════════════════════════════════════════════════
# Ajuste isotonico circular unimodal  (R: busquedaMejor en NucleoComun.R)
# ═══════════════════════════════════════════════════════════════════════════

def circular_unimodal_fit(
    v: np.ndarray,
) -> tuple[np.ndarray, float, int, int] | None:
    """
    Ajusta una regresion isotonica circular unimodal (subida-bajada).

    Busca sobre todos los pares candidatos (valle, pico) — minimos y
    maximos locales de *v* — para encontrar el par (L, U) que produce
    el menor MSE al ajustar un PAVA creciente de L→U y un PAVA
    decreciente de U→L (circularmente).

    Equivalente en R
    -----------------
    ``busquedaMejor(v, candL, candU)`` en ``NucleoComun.R`` (lineas 83–478).

    Parametros
    ----------
    v : array (n,)
        Valores de expresion genica ordenados por tiempo circular.

    Devuelve
    --------
    fitted : array (n,) — valores ajustados.
    mse    : float      — error cuadratico medio.
    L_opt  : int        — indice del valle optimo (base 0).
    U_opt  : int        — indice del pico optimo (base 0).
    None si no se encuentra un ajuste valido.
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)

    if n == 0:
        return None

    candL, candU = find_local_extrema(v)

    if len(candL) == 0 or len(candU) == 0:
        # Sin extremos — el ajuste constante es lo mejor que podemos hacer.
        fitted = np.full(n, v.mean())
        mse = float(np.mean((v - fitted) ** 2))
        return fitted, mse, 0, 0

    # ── Ruta rapida Numba ────────────────────────────────────────────────
    if _HAS_NUMBA:
        fitted, mse, L_opt, U_opt, ok = _circular_unimodal_fit_numba(
            v,
            np.ascontiguousarray(candL.astype(np.int64)),
            np.ascontiguousarray(candU.astype(np.int64)),
        )
        if ok:
            return fitted, float(mse), int(L_opt), int(U_opt)
        return None

    # Vector duplicado para indexacion circular envolvente.
    v2 = np.concatenate([v, v])

    best_mse    = np.inf
    best_fitted = None
    best_L      = 0
    best_U      = 0

    for indL in candL:
        # Ordenar candidatos U: primero los ≥ indL, luego los < indL.
        # Replica el c(candU[candU>=indL], candU[candU<indL]) de R.
        mask      = candU >= indL
        ordered_U = np.concatenate([candU[mask], candU[~mask]])

        # ── Pasada hacia adelante: identificar U's validos con sus PAVAs crecientes ──
        valid_Us:   list[int]        = []
        valid_incs: list[np.ndarray] = []

        for indU in ordered_U:
            inc_result = _increasing_segment(v, v2, indL, indU, n)

            if inc_result is None:
                # senal de "break": PAVA no sube desde el valle.
                break

            full_inc, is_valid_U = inc_result
            if is_valid_U:
                valid_Us.append(indU)
                valid_incs.append(full_inc)

        # ── Pasada hacia atras: calcular PAVAs decrecientes para U's validos ──
        for idx_rev, indU in enumerate(reversed(valid_Us)):
            j = len(valid_Us) - 1 - idx_rev
            full_inc = valid_incs[j]

            if indL == indU:
                # Ajuste constante.
                fitted = full_inc
                mse = float(np.mean((v - fitted) ** 2))
                if mse < best_mse:
                    best_mse, best_fitted = mse, fitted
                    best_L, best_U = indL, indU
                continue

            dec_result = _decreasing_segment(v, v2, indL, indU, n)

            if dec_result is None:
                # senal de "break": ajuste decreciente no alcanza el valle.
                break

            dec_fit, is_valid = dec_result
            if not is_valid:
                continue

            # Ensamblar ajuste circular completo.
            fitted = _assemble(v, indL, indU, full_inc, dec_fit, n)
            mse = float(np.mean((v - fitted) ** 2))

            if mse < best_mse:
                best_mse, best_fitted = mse, fitted
                best_L, best_U = indL, indU

        if best_mse == 0.0:
            break

    if best_fitted is None:
        return None

    return best_fitted, best_mse, best_L, best_U


# ── Auxiliares para circular_unimodal_fit ─────────────────────────────────

def _increasing_segment(
    v: np.ndarray, v2: np.ndarray,
    indL: int, indU: int, n: int,
) -> tuple[np.ndarray, bool] | None:
    """
    Calcula el PAVA creciente desde *indL* hasta *indU* (circular),
    con ``v[indL]`` y ``v[indU]`` como valores de frontera fijos.

    Devuelve
    --------
    (full_inc, is_valid_U) — valores ajustados de L a U y flag de validez.
    None — senal de "break": no hay mas U's validos para este L.
    """
    if indL == indU:
        return np.full(n, v.mean()), True

    # Puntos interiores entre L y U (excluyendo ambos extremos).
    if indL < indU:
        k_interior = indU - indL - 1
        if k_interior > 0:
            interior_vals = v[indL + 1 : indU]
        else:
            interior_vals = np.array([], dtype=np.float64)
    else:
        # Envolver: L+1, L+2, …, n-1, 0, 1, …, U-1
        k_interior = (n - indL - 1) + indU
        if k_interior > 0:
            interior_vals = v2[indL + 1 : indL + 1 + k_interior]
        else:
            interior_vals = np.array([], dtype=np.float64)

    # Construir ajuste creciente completo: [v[L], pava(interior), v[U]]
    if len(interior_vals) > 0:
        inc_fit  = pava_increasing(interior_vals)
        full_inc = np.empty(len(interior_vals) + 2, dtype=np.float64)
        full_inc[0]    = v[indL]
        full_inc[1:-1] = inc_fit
        full_inc[-1]   = v[indU]
    else:
        full_inc = np.array([v[indL], v[indU]], dtype=np.float64)

    # Verificaciones de validez (coincidiendo con busquedaMejor de R).
    if len(full_inc) > 2:
        # El primer punto interior debe superar al valle.
        if full_inc[1] <= v[indL]:
            return None   # "break" — no hay mas U's validos para este L.
        # El ultimo punto interior debe estar debajo del pico.
        is_valid = full_inc[-2] < v[indU]
    else:
        # Solo 2 puntos (L y U adyacentes): siempre valido.
        is_valid = True

    return full_inc, is_valid


def _decreasing_segment(
    v: np.ndarray, v2: np.ndarray,
    indL: int, indU: int, n: int,
) -> tuple[np.ndarray, bool] | None:
    """
    Calcula el PAVA decreciente en el arco interior de *indU* a *indL*
    (el complemento del arco creciente).

    A diferencia del segmento creciente, los extremos ``v[indU]`` y ``v[indL]``
    **no** se incluyen en la entrada PAVA — solo se ajusta el interior estricto.

    Devuelve
    --------
    (dec_fit, is_valid) — valores interiores ajustados y flag de validez.
    None — senal de "break": no hay mas U's validos.
    """
    # Puntos interiores: indU+1, …, indL-1  (circularmente).
    if indU < indL:
        k_interior = indL - indU - 1
        if k_interior > 0:
            interior_vals = v[indU + 1 : indL]
        else:
            interior_vals = np.array([], dtype=np.float64)
    else:
        # Envolver: indU+1, …, n-1, 0, …, indL-1
        k_interior = (n - indU - 1) + indL
        if k_interior > 0:
            interior_vals = v2[indU + 1 : indU + 1 + k_interior]
        else:
            interior_vals = np.array([], dtype=np.float64)

    if len(interior_vals) == 0:
        return np.array([], dtype=np.float64), True

    dec_fit = pava_decreasing(interior_vals)

    # Validez: ultimo valor ajustado debe ser ≥ valle (puede conectar suavemente).
    if dec_fit[-1] < v[indL]:
        return None  # senal de "break".

    # Primer valor ajustado debe ser ≤ pico (comienza debajo del maximo).
    is_valid = dec_fit[0] <= v[indU]

    return dec_fit, is_valid


def _assemble(
    v: np.ndarray,
    indL: int, indU: int,
    full_inc: np.ndarray,
    dec_fit: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Coloca los valores ajustados crecientes y decrecientes de vuelta en
    un array de longitud *n* alineado con las posiciones originales de *v*.

    ``full_inc`` cubre posiciones L, L+1, …, U (incluyendo extremos).
    ``dec_fit`` cubre posiciones U+1, U+2, …, L-1 (solo interior).
    """
    fitted = np.empty(n, dtype=np.float64)

    # ── Arco creciente: L → U ──
    if indL <= indU:
        inc_pos = np.arange(indL, indU + 1)
    else:
        inc_pos = np.concatenate([np.arange(indL, n), np.arange(0, indU + 1)])

    fitted[inc_pos] = full_inc

    # ── Arco decreciente: U+1 → L-1 ──
    if len(dec_fit) > 0:
        if indU < indL:
            dec_pos = np.arange(indU + 1, indL)
        else:
            dec_pos = np.concatenate([
                np.arange(indU + 1, n), np.arange(0, indL)
            ])
        fitted[dec_pos] = dec_fit

    return fitted


# ═══════════════════════════════════════════════════════════════════════════
# Dataclass de resultado
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NonParametricResult:
    """
    Salida de :class:NonParametricScorer.

    Atributos
    ---------
    fitted : pd.DataFrame, forma (n_genes, n_muestras)
        Valores ajustados de regresion isotonica unimodal por gen.
        Equivalente en R: ``computeNP()[[1]]``  (= ``fitNP``).

    mse_np : np.ndarray, forma (n_genes,)
        MSE del ajuste no parametrico para cada gen.
        Equivalente en R: ``computeNP()[[3]]``  (= ``msseNP``).

    mse_flat : np.ndarray, forma (n_genes,)
        MSE del modelo plano (media) para cada gen.
        Equivalente en R: ``computeNP()[[4]]``  (= ``msseFlat``).

    r2 : np.ndarray, forma (n_genes,)
        R² no parametrico = 1 − MSE_NP / MSE_flat.
        Equivalente en R: ``computeNP()[[5]]``  (= ``R2``).

    gene_names : list[str]
        Simbolos de genes en el mismo orden de filas que los arrays anteriores.
    """

    fitted:     pd.DataFrame
    mse_np:     np.ndarray
    mse_flat:   np.ndarray
    r2:         np.ndarray
    gene_names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.r2)
        above50 = int(np.sum(self.r2 > 0.5))
        above70 = int(np.sum(self.r2 > 0.7))
        lines = [
            "=== Resumen de Ritmicidad No Parametrica ===",
            f"  Genes puntuados     : {n}",
            f"  Mediana R2_NP       : {float(np.median(self.r2)):.3f}",
            f"  R2_NP > 0.5         : {above50} ({100*above50/max(n,1):.1f}%)",
            f"  R2_NP > 0.7         : {above70} ({100*above70/max(n,1):.1f}%)",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# NonParametricScorer
# ═══════════════════════════════════════════════════════════════════════════

class NonParametricScorer:
    """
    Puntua cada gen en una matriz de expresion para ritmicidad no parametrica
    usando regresion isotonica circular unimodal.

    Para cada gen, el algoritmo ajusta la mejor forma subida-bajada (un pico,
    un valle) usando PAVA sobre todos los pares de extremos candidatos. El R²
    contra un modelo plano (media) cuantifica que tan bien el perfil de
    expresion ordenado sigue una oscilacion unimodal — sin asumir ninguna
    forma de onda parametrica.

    Equivalente en R: ``computeNP(datos)`` (lineas 13–30).

    Parametros
    ----------
    verbose : bool
        Imprimir mensajes de progreso.

    Ejemplos
    --------
    >>> from circust.nonparametric import NonParametricScorer
    >>> result = NonParametricScorer().run(expr_matrix)
    >>> print(result.summary())
    """

    def __init__(self, verbose: bool = True, n_jobs: int = -1) -> None:
        """
        Parametros
        ----------
        verbose : bool
            Imprimir mensajes de progreso.
        n_jobs : int
            Numero de procesos worker para paralelizar el bucle por gen.
            ``-1`` usa todos los nucleos disponibles, ``1`` deshabilita el
            paralelismo (util para debugging). Se usa ``joblib`` con el
            backend ``loky``.
        """
        self.verbose = verbose
        self.n_jobs  = n_jobs

    @staticmethod
    def _score_one_gene(
        row: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        """
        Puntua un unico gen. Devuelve (fitted, mse_np, mse_flat, r2).

        Funcion pura (sin ``self``) para poder ser serializada por
        ``joblib`` hacia los procesos worker.
        """
        row_mean  = float(row.mean())
        mse_flat  = float(np.mean((row - row_mean) ** 2))

        result = circular_unimodal_fit(row)
        if result is not None:
            fitted = result[0]
            mse_np = float(result[1])
        else:
            fitted = np.full(row.shape[0], row_mean)
            mse_np = mse_flat

        if mse_flat > 0.0:
            r2 = 1.0 - mse_np / mse_flat
        else:
            r2 = 0.0

        return fitted, mse_np, mse_flat, r2

    def run(self, expr_ordered: pd.DataFrame) -> NonParametricResult:
        """
        Puntua todos los genes.

        Parametros
        ----------
        expr_ordered : pd.DataFrame, forma (n_genes, n_muestras)
            Matriz de expresion normalizada completa ya ordenada por el
            tiempo circular preliminar.

        Devuelve
        --------
        NonParametricResult
        """
        genes  = expr_ordered.index.tolist()
        n_genes, n_samples = expr_ordered.shape
        values = expr_ordered.values.astype(np.float64)

        self._log("=== Etapa 3.1: Puntuacion de Ritmicidad No Parametrica ===")

        fitted_mat = np.zeros((n_genes, n_samples), dtype=np.float64)
        mse_np     = np.zeros(n_genes, dtype=np.float64)
        mse_flat   = np.zeros(n_genes, dtype=np.float64)
        r2         = np.zeros(n_genes, dtype=np.float64)

        # ── Bucle por gen (paralelizable con joblib) ─────────────────────
        # Cada gen es independiente, asi que repartimos las filas entre
        # procesos worker. Usamos el backend 'loky' (por defecto) para
        # evitar la GIL; para n_jobs=1 ejecutamos en el proceso principal
        # sin overhead de serializacion.
        score_fn = NonParametricScorer._score_one_gene

        if self.n_jobs == 1:
            results = [score_fn(values[i]) for i in range(n_genes)]
        else:
            from joblib import Parallel, delayed
            self._log(f"  Paralelizando en {self.n_jobs} jobs ...")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(score_fn)(values[i]) for i in range(n_genes)
            )

        for i, (fit_i, mse_np_i, mse_flat_i, r2_i) in enumerate(results):
            fitted_mat[i] = fit_i
            mse_np[i]     = mse_np_i
            mse_flat[i]   = mse_flat_i
            r2[i]         = r2_i

        self._log(f"  Hecho — {n_genes} genes puntuados.")

        result = NonParametricResult(
            fitted     = pd.DataFrame(fitted_mat, index=genes,
                                      columns=expr_ordered.columns),
            mse_np     = mse_np,
            mse_flat   = mse_flat,
            r2         = r2,
            gene_names = genes,
        )

        self._log(result.summary())
        return result

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
