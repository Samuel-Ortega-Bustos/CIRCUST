# ============================================================
# FMM1 stabilized fitting (power + curvature roughness penalty)
# ============================================================
# This file contains:
#  - Computation of cos(phi) and sin(phi) for the Mobius phase: mobius_cos_sin()
#  - Grid precomputation: fmm_grid_precompute()
#  - Per-(dataset,weights) cache: fmm_cache_prepare()
#  - Fit by power + roughness: fmm_fit_power_rough()
#  - Lambda calibration by null Monte Carlo: fmm_calibrate_lambda()
#  - Time-density weights (von Mises KDE + LOO CV): fmm_select_kappa_lcv(), fmm_time_weights()
#  - Utilities: spike_max_jump(), simulate_H0_whitened()
#
# Notes:
#  - Times t are assumed in [0, 2*pi). If not, they are wrapped with t %% (2*pi).
#  - Weights are normalized to sum(w)=n by default.
#  - The "curvature" roughness uses 3-point discrete second derivative of the basis
#    along circularly ordered times, scaled by dt.
# ============================================================

# -----------------------------
# 0) Mobius cos/sin
# -----------------------------
#' Stable computation of cos(phi) and sin(phi) for the Mobius phase
#'
#' Computes C = cos(phi(t;alpha,omega)) and S = sin(phi(t;alpha,omega))
#' without explicitly computing phi via tan/atan near the asymptote.
#'
#' @param t numeric vector of times (radians)
#' @param alpha scalar in [0,2*pi)
#' @param omega scalar in [0,1]
#' @param den_eps small positive constant to avoid division by zero
#' @return list(C=..., S=...)
mobius_cos_sin <- function(t, alpha, omega, den_eps = .Machine$double.eps) {
  t <- t %% (2*pi)
  alpha <- alpha %% (2*pi)
  
  if (!is.finite(omega) || omega <= 0) {
    # Exact limit: phi == 0 -> cos(phi)=1, sin(phi)=0
    return(list(C = rep(1, length(t)), S = rep(0, length(t))))
  }
  omega <- min(max(omega, 0), 1)
  
  u <- 0.5 * (t - alpha)
  s <- sin(u)
  c <- cos(u)
  
  den <- c*c + (omega*omega) * (s*s)
  den <- den + den_eps
  
  C <- (c*c - (omega*omega) * (s*s)) / den
  S <- (2 * omega * s * c) / den
  
  # Renormalize to enforce C^2 + S^2 = 1 (numerical safety)
  r <- sqrt(C*C + S*S) + den_eps
  list(C = C / r, S = S / r)
}




# -----------------------------
# 1) Grid precomputation
# -----------------------------

#' Precompute FMM grid basis and roughness terms
#'
#' @param t numeric vector of times (radians)
#' @param alpha_grid numeric vector in [0,2*pi)
#' @param omega_grid numeric vector in (0,1]
#' @return list containing:
#'  - C,S and elementwise products C2,S2,CS
#'  - ord: circular order of times
#'  - dt: circular gaps
#'  - ddC2, ddS2: curvature roughness components per grid row
fmm_grid_precompute <- function(t,
                                alpha_grid = seq(0, 2*pi, length.out = 48),
                                omega_grid = exp(seq(log(1e-4), log(1), length.out = 30))) {
  # wrap times
  t <- t %% (2*pi)
  n <- length(t)
  
  if (n < 4) stop("Need at least 4 time points for curvature roughness.")
  if (any(!is.finite(t))) stop("Non-finite t detected.")
  
  alpha_grid <- alpha_grid %% (2*pi)
  omega_grid <- pmin(pmax(omega_grid, 0), 1)
  if (any(omega_grid <= 0)) stop("omega_grid must be in (0,1].")
  
  nA <- length(alpha_grid)
  nO <- length(omega_grid)
  G  <- nA * nO
  
  alpha_vec <- rep(alpha_grid, each = nO)
  omega_vec <- rep(omega_grid, times = nA)
  
  C <- matrix(NA_real_, nrow = G, ncol = n)
  S <- matrix(NA_real_, nrow = G, ncol = n)
  
  row <- 1L
  for (a in alpha_grid) {
    for (o in omega_grid) {
      cs <- mobius_cos_sin(t = t, alpha = a, omega = o)
      C[row, ] <- cs$C
      S[row, ] <- cs$S
      row <- row + 1L
    }
  }
  
  C2 <- C*C
  S2 <- S*S
  CS <- C*S
  
  # Circular order and gaps
  ord <- order(t)
  t_ord <- t[ord]
  dt <- c(diff(t_ord), (t_ord[1] + 2*pi) - t_ord[length(t_ord)])
  dt <- pmax(dt, .Machine$double.eps)
  
  # Put basis in circular order
  C_ord  <- C[, ord, drop = FALSE]
  S_ord  <- S[, ord, drop = FALSE]
  C_next <- C_ord[, c(2:n, 1), drop = FALSE]
  S_next <- S_ord[, c(2:n, 1), drop = FALSE]
  
  dC <- C_next - C_ord
  dS <- S_next - S_ord
  
  dt_mat <- matrix(dt, nrow = nrow(C), ncol = n, byrow = TRUE)
  
  # Curvature roughness: differences of local slopes
  slopeC <- dC / dt_mat
  slopeS <- dS / dt_mat
  
  slopeC_prev <- slopeC[, c(n, 1:(n-1)), drop = FALSE]
  slopeS_prev <- slopeS[, c(n, 1:(n-1)), drop = FALSE]
  
  ddC <- (slopeC - slopeC_prev) 
  ddS <- (slopeS - slopeS_prev) 
  
  ddC2 <- rowSums(ddC * ddC)
  ddS2 <- rowSums(ddS * ddS)
  
  list(
    t = t,
    ord = ord,
    dt = dt,
    alpha_grid = alpha_grid,
    omega_grid = omega_grid,
    alpha_vec = alpha_vec,
    omega_vec = omega_vec,
    C = C, S = S,
    C2 = C2, S2 = S2, CS = CS,
    ddC2 = ddC2, ddS2 = ddS2
  )
}

# -----------------------------
# 2) Cache: inv(X'WX) per gridpoint (no smoothing inside WLS)
# -----------------------------

#' Prepare cache for fast closed-form WLS for all grid points
#'
#' Design matrix per grid point is [1, C, S], weights are fixed.
#'
#' @param weights numeric vector (nonnegative), length n
#' @param grid output of fmm_grid_precompute()
#' @param ridge numeric ridge added to diagonal for numerical stability
#' @param normalize logical; if TRUE, weights are rescaled to sum(w)=n
#' @return list with w and inv components inv11,...,inv33
fmm_cache_prepare <- function(weights, grid, ridge = 1e-10, normalize = TRUE) {
  w <- as.numeric(weights)
  if (length(w) !=  length(grid$t)) stop("weights must have same length as t.")
  if (any(!is.finite(w))) stop("Non-finite weights detected.")
  if (any(w < 0)) stop("Weights must be nonnegative.")
  
  n <- length(w)
  if (normalize) {
    sw <- sum(w)
    if (sw <= 0) stop("Sum of weights must be > 0.")
    w <- n * w / sw
  }
  
  G <- nrow(grid$C)
  
  a11 <- sum(w)
  a12 <- as.numeric(grid$C  %*% w)
  a13 <- as.numeric(grid$S  %*% w)
  a22 <- as.numeric(grid$C2 %*% w)
  a23 <- as.numeric(grid$CS %*% w)
  a33 <- as.numeric(grid$S2 %*% w)
  
  inv11 <- inv12 <- inv13 <- inv22 <- inv23 <- inv33 <- numeric(G)
  
  for (g in seq_len(G)) {
    A <- matrix(c(
      a11,     a12[g], a13[g],
      a12[g],  a22[g], a23[g],
      a13[g],  a23[g], a33[g]
    ), nrow = 3, byrow = TRUE)
    
    A[1,1] <- A[1,1] + ridge
    A[2,2] <- A[2,2] + ridge
    A[3,3] <- A[3,3] + ridge
    
    
    iA <- tryCatch(chol2inv(chol(A)), error = function(e) NULL)
    if (is.null(iA)) {
      A <- A + diag(ridge*10, 3)
      iA <- chol2inv(chol(A))
    }
    
    inv11[g] <- iA[1,1]; inv12[g] <- iA[1,2]; inv13[g] <- iA[1,3]
    inv22[g] <- iA[2,2]; inv23[g] <- iA[2,3]; inv33[g] <- iA[3,3]
  }
  
  list(
    w = w,
    inv11 = inv11, inv12 = inv12, inv13 = inv13,
    inv22 = inv22, inv23 = inv23, inv33 = inv33
  )
}

# -----------------------------
# 3) Fit: Power + curvature penalty, then WLS refit at selected (alpha,omega)
# -----------------------------

#' Fit FMM1 using normalized power criterion with curvature roughness penalty
#'
#' Selection step maximizes:
#'  J(alpha,omega) = P(alpha,omega) - lambda * g(omega) * R(alpha,omega),
#' with g(omega)=(omega0/(omega+omega0))^p and R computed from curvature of C,S.
#'
#' @param x numeric vector length n (observations)
#' @param grid output of fmm_grid_precompute()
#' @param cache output of fmm_cache_prepare() with same weights
#' @param center logical; center x by weighted mean before computing P
#' @param normalize logical; divide power by weighted energy of (C,S)
#' @param lambda numeric >=0 (roughness strength)
#' @param omega0 numeric in (0,1)
#' @param p numeric >=1
#' @param p numeric >=1
#' @param refine logical; if TRUE, refine (alpha,omega) locally after grid selection
#' @param refine_steps integer; number of refinement iterations
#' @param alpha_step numeric; initial alpha step size (radians)
#' @param omega_step_mult numeric; multiplicative omega step (e.g. 1.25)
#' @param omega_min numeric; lower bound for omega in refinement
#' @param omega_max numeric; upper bound for omega in refinement
#' @param ridge_refine numeric; ridge used in the tiny 3x3 WLS during refinement
#' @param keep_trace logical; if TRUE, return refinement trace
#' @param rough_mode
#' @param ndense 
#' @param return_grid
#' @return list with parameters, fitted_values, rss, and diagnostics
fmm_fit_power_rough <- function(x, grid, cache,
                                center = TRUE,
                                normalize = TRUE,
                                lambda = 0,
                                omega0 = 0.10,
                                p = 3,
                                refine = FALSE,
                                refine_steps = 3L,
                                alpha_step = (2*pi)/48,
                                omega_step_mult = 1.25,
                                omega_min = 1e-4,
                                omega_max = 1,
                                ridge_refine = 1e-10,
                                keep_trace = FALSE,
                                rough_mode = c("observed" , "dense"),
                                ndense = 512L,
                                return_grid = FALSE) {
  
  rough_mode <- match.arg(rough_mode)
  x <- as.numeric(x)
  if (length(x) != length(grid$t)) stop("x must have same length as t.")
  if (any(!is.finite(x))) stop("Non-finite x detected.")
  if (lambda < 0) stop("lambda must be >= 0.")
  if (!(omega0 > 0 && omega0 < 1)) stop("omega0 must be in (0,1).")
  if (p < 1) stop("p must be >= 1.")
  
  w <- cache$w
  if (length(w) != length(x)) stop("cache weights length mismatch.")
  
  # ---- POWER ----
  if (center) {
    mu_w <- sum(w * x) / sum(w)
    x0 <- x - mu_w
  } else {
    x0 <- x
  }
  
  wx0 <- w * x0
  u <- as.numeric(grid$C %*% wx0)
  v <- as.numeric(grid$S %*% wx0)
  pow <- u*u + v*v
  
  if (normalize) {
    den <- as.numeric(grid$C2 %*% w) + as.numeric(grid$S2 %*% w)
    den[den <= 0] <- NA_real_
    pow <- pow / den
  }
  
  # ---- Optional: dense roughness ----
  if (rough_mode == "dense" && lambda > 0) {
    if (is.null(grid$rough_dense) || is.null(grid$t_dense) || is.null(grid$ndense) || grid$ndense != ndense) {
      # Build dense uniform grid once (store inside grid)
      t_dense <- seq(0, 2*pi, length.out = ndense + 1L)
      t_dense <- t_dense[-length(t_dense)]
      nd <- length(t_dense)
      dt <- rep(2*pi/nd, nd)
      dt <- pmax(dt, .Machine$double.eps)
      
      R_dense <- numeric(length(grid$alpha_vec))
      for (g in seq_along(R_dense)) {
        cs <- mobius_cos_sin(t = t_dense, alpha = grid$alpha_vec[g], omega = grid$omega_vec[g])
        C <- cs$C; S <- cs$S
        
        C_next <- C[c(2:nd, 1)]
        S_next <- S[c(2:nd, 1)]
        dC <- C_next - C
        dS <- S_next - S
        
        slopeC <- dC / dt
        slopeS <- dS / dt
        slopeC_prev <- slopeC[c(nd, 1:(nd-1))]
        slopeS_prev <- slopeS[c(nd, 1:(nd-1))]
        
        ddC <- slopeC - slopeC_prev
        ddS <- slopeS - slopeS_prev
        
        R_dense[g] <- sum(ddC*ddC) + sum(ddS*ddS)
      }
      
      grid$rough_dense <- R_dense
      grid$t_dense <- t_dense
      grid$ndense <- ndense
    }
    
    rough <- grid$rough_dense
  } else {
    rough <- grid$ddC2 + grid$ddS2
  }
  
  # ---- Curvature roughness (grid-only) ----
  if (lambda > 0) {
    #rough <- rough
    gomega <- (omega0 / (grid$omega_vec + omega0))^p
    score <- pow - lambda * gomega * rough
  } else {
    rough <- rep(0, length(pow))
    gomega <- rep(0, length(pow))
    score <- pow
  }
  
  score2 <- score
  score2[!is.finite(score2)] <- -Inf
  
  # soluciones con el mismo score
  mx <- max(score2, na.rm = TRUE)
  idx_close <- which(score2 >= mx - 1e-2 & is.finite(score2))
  #length(idx_eq)
  # el maximo
  #idx <- which.max(score2)
  #if (length(idx) != 1L || !is.finite(idx)) stop("Failed to select a grid point (idx).")
  if (length(idx_close) == 0L) stop("Failed to select a grid point (idx).")
  
  
  # ---- Helper: closed-form WLS at a given (alpha,omega) using mobius_cos_sin ----
  wls_at <- function(alpha, omega) {
    alpha <- alpha %% (2*pi)
    omega <- min(max(omega, omega_min), omega_max)
    
    cs <- mobius_cos_sin(t = grid$t, alpha = alpha, omega = omega)
    C <- cs$C
    S <- cs$S
    
    # Weighted normal equations for [1, C, S]
    a11 <- sum(w)
    a12 <- sum(w * C)
    a13 <- sum(w * S)
    a22 <- sum(w * C * C)
    a23 <- sum(w * C * S)
    a33 <- sum(w * S * S)
    
    A <- matrix(c(
      a11, a12, a13,
      a12, a22, a23,
      a13, a23, a33
    ), 3, 3, byrow = TRUE)
    
    A[1,1] <- A[1,1] + ridge_refine
    A[2,2] <- A[2,2] + ridge_refine
    A[3,3] <- A[3,3] + ridge_refine
    
    b1 <- sum(w * x)
    b2 <- sum(w * x * C)
    b3 <- sum(w * x * S)
    
    coef <- solve(A, c(b1, b2, b3))
    c1 <- coef[1]; c2 <- coef[2]; c3 <- coef[3]
    
    fitted <- c1 + c2 * C + c3 * S
    rss <- sum(w * (x - fitted)^2)
    
    Ahat <- sqrt(c2^2 + c3^2)
    betahat <- atan2(-c3, c2) %% (2*pi)
    
    list(
      parameters = list(M = c1, A = Ahat, beta = betahat, alpha = alpha, omega = omega),
      fitted_values = fitted,
      rss = rss
    )
  }

  # ---- Helper: evaluate score J(alpha,omega) locally (same definition as in the paper) ----
  eval_score <- function(alpha, omega) {
    alpha <- alpha %% (2*pi)
    omega <- min(max(omega, omega_min), omega_max)
    
    cs <- mobius_cos_sin(t = grid$t, alpha = alpha, omega = omega)
    C <- cs$C
    S <- cs$S
    
    uu <- sum(wx0 * C)
    vv <- sum(wx0 * S)
    Pval <- uu*uu + vv*vv
    
    if (normalize) {
      den <- sum(w * C * C) + sum(w * S * S)
      if (!is.finite(den) || den <= 0) return(-Inf)
      Pval <- Pval / den
    }
    
    if (lambda <= 0) return(Pval)
    
    # ---- Curvature roughness ----
    if (rough_mode == "dense") {
      t_dense <- grid$t_dense
      nd <- length(t_dense)
      dt <- rep(2*pi/nd, nd)
      dt <- pmax(dt, .Machine$double.eps)
      
      csd <- mobius_cos_sin(t = t_dense, alpha = alpha, omega = omega)
      Cd <- csd$C; Sd <- csd$S
      
      Cd_next <- Cd[c(2:nd, 1)]
      Sd_next <- Sd[c(2:nd, 1)]
      dCd <- Cd_next - Cd
      dSd <- Sd_next - Sd
      
      slopeCd <- dCd / dt
      slopeSd <- dSd / dt
      slopeCd_prev <- slopeCd[c(nd, 1:(nd-1))]
      slopeSd_prev <- slopeSd[c(nd, 1:(nd-1))]
      
      ddCd <- slopeCd - slopeCd_prev
      ddSd <- slopeSd - slopeSd_prev
      
      Rval <- sum(ddCd*ddCd) + sum(ddSd*ddSd)
      
    } else {
      # observed-time roughness (your current construction)
      ord <- grid$ord
      n <- length(grid$t)
      t_ord <- grid$t[ord]
      dt <- c(diff(t_ord), (t_ord[1] + 2*pi) - t_ord[n])
      dt <- pmax(dt, .Machine$double.eps)
      
      C_ord <- C[ord]
      S_ord <- S[ord]
      C_next <- C_ord[c(2:n, 1)]
      S_next <- S_ord[c(2:n, 1)]
      
      dC <- C_next - C_ord
      dS <- S_next - S_ord
      slopeC <- dC / dt
      slopeS <- dS / dt
      
      slopeC_prev <- slopeC[c(n, 1:(n-1))]
      slopeS_prev <- slopeS[c(n, 1:(n-1))]
      
      ddC <- slopeC - slopeC_prev
      ddS <- slopeS - slopeS_prev
      
      Rval <- sum(ddC*ddC) + sum(ddS*ddS)
    }
    
    g <- (omega0 / (omega + omega0))^p
    
    Pval - lambda * g * Rval
  }

  # ---- Elijo el mejor entre los candidatos ----
    best <- idx_close[1]
    best_obj <- Inf
    best_omega <- -Inf
    
    for (ii in idx_close) {
      fit <- wls_at(grid$alpha_vec[ii] , grid$omega_vec[ii])
      rss <- fit$rss
      pen <- lambda * gomega[ii] * rough[ii]
      obj <- rss + pen
      
      if (obj < best_obj) {
        best_obj <- obj
        best <- ii
        best_omega <- grid$omega_vec[ii]
      } else if (isTRUE(all.equal(obj, best_obj, tolerance=1e-10))) {
        # tie-break: prefer larger omega (smoother)
        if (grid$omega_vec[ii] > best_omega) {
          best <- ii
          best_omega <- grid$omega_vec[ii]
        }
      }
    }
    idx <- best
  
  # ---- Base (coarse) parameters from grid ----
  alpha_hat <- grid$alpha_vec[idx] %% (2*pi)
  omega_hat <- min(max(grid$omega_vec[idx], omega_min), omega_max)
  
  # Optional: local refinement around (alpha_hat, omega_hat)
  trace <- NULL
  if (refine) {
    score_cur <- eval_score(alpha_hat, omega_hat)
    
    if (keep_trace) {
      trace <- data.frame(step = 0L, alpha = alpha_hat, omega = omega_hat, score = score_cur)
    }
    
    a_step <- alpha_step
    o_mult <- omega_step_mult
    
    for (s in seq_len(refine_steps)) {
      candidates <- list(
        c(alpha_hat + a_step, omega_hat),
        c(alpha_hat - a_step, omega_hat),
        c(alpha_hat, omega_hat * o_mult),
        c(alpha_hat, omega_hat / o_mult),
        c(alpha_hat + a_step, omega_hat * o_mult),
        c(alpha_hat - a_step, omega_hat * o_mult),
        c(alpha_hat + a_step, omega_hat / o_mult),
        c(alpha_hat - a_step, omega_hat / o_mult)
      )
      
      #best_a <- alpha_hat
      #best_o <- omega_hat
      #best_s <- score_cur
      
      #for (cand in candidates) {
      #  sc <- eval_score(cand[1], cand[2])
      #  if (is.finite(sc) && sc > best_s) {
      #    best_s <- sc
      #    best_a <- cand[1] %% (2*pi)
      #    best_o <- min(max(cand[2], omega_min), omega_max)
      #  }
      #}
      # 1) evalúa scores locales
      scs <- vapply(candidates, function(cand) eval_score(cand[1], cand[2]), numeric(1))
      ok <- is.finite(scs)
      if (!any(ok)) {
        # no hay candidatos válidos -> reduce pasos
        a_step <- a_step / 2
        o_mult <- sqrt(o_mult)
        next
      }
      
      mx_loc <- max(scs[ok])
      
      # tolerancia local (relativa al tamaño del score)
      tol_loc <- 1e-4 * max(1, abs(mx_loc))
      cand_idx <- which(ok & scs >= mx_loc - tol_loc)
      
      # 2) entre la meseta local, elige el que minimice rss+pen (o prioriza omega grande)
      best_obj <- Inf
      best_a <- alpha_hat
      best_o <- omega_hat
      best_s <- score_cur
      best_omega <- -Inf
      
      for (j in cand_idx) {
        a <- candidates[[j]][1] %% (2*pi)
        o <- min(max(candidates[[j]][2], omega_min), omega_max)
        
        fitj <- wls_at(a, o)  # te da rss
        rssj <- fitj$rss
        
        # penalización rápida (observed roughness) o dense; si no quieres coste, usa gomega solo
        roughj <- fmm_roughness(a, o, grid, rough_mode = rough_mode, ndense = ndense)
        penj <- lambda * (omega0 / (o + omega0))^p * roughj
        objj <- rssj + penj
        
        if (objj < best_obj) {
          best_obj <- objj
          best_a <- a
          best_o <- o
          best_s <- scs[j]
          best_omega <- o
        } else if (isTRUE(all.equal(objj, best_obj, tolerance = 1e-10))) {
          if (o > best_omega) {
            best_a <- a
            best_o <- o
            best_s <- scs[j]
            best_omega <- o
          }
        }
      }
      
      # accept or shrink steps
      if (best_s > score_cur) {
        alpha_hat <- best_a
        omega_hat <- best_o
        score_cur <- best_s
      } else {
        a_step <- a_step / 2
        o_mult <- sqrt(o_mult)
      }
      
      if (keep_trace) {
        trace <- rbind(trace, data.frame(step = s, alpha = alpha_hat, omega = omega_hat, score = score_cur))
      }
    }
  }
  
  # ---- Final WLS fit at selected (possibly refined) pair ----
  final <- wls_at(alpha_hat, omega_hat)
  
  rough_final  <- fmm_roughness(alpha_hat, omega_hat, grid,
                                rough_mode = rough_mode, ndense = ndense)
  gomega_final <- (omega0 / (omega_hat + omega0))^p
  pen_final    <- lambda * gomega_final * rough_final
  
  
  out <- list(
    parameters = final$parameters,
    fitted_values = final$fitted_values,
    rss = final$rss,
    grid_index = idx,
    power = pow[idx],
    score_grid = score[idx],
    score = score,
    lambda = lambda,
    omega0 = omega0,
    p = p,
    refine = refine,
    rough_sel = rough_final,
    gomega_sel = gomega_final,
    penalty_sel = pen_final
  )
  
  if (keep_trace) out$refine_trace <- trace
  if (return_grid) out$grid <- grid
  
  out
}

wls_at_huber_grid <- function(ii, x, w, grid,
                              ridge = 1e-8,
                              huber_k = 1.345,
                              maxit = 30,
                              tol = 1e-6) {
  C <- as.numeric(grid$C[ii, ])
  S <- as.numeric(grid$S[ii, ])
  x <- as.numeric(x)
  w <- as.numeric(w)
  
  solve_wls <- function(ww) {
    a11 <- sum(ww)
    a12 <- sum(ww * C)
    a13 <- sum(ww * S)
    a22 <- sum(ww * C * C)
    a23 <- sum(ww * C * S)
    a33 <- sum(ww * S * S)
    
    A <- matrix(c(a11,a12,a13,
                  a12,a22,a23,
                  a13,a23,a33), 3,3, byrow=TRUE)
    A[1,1] <- A[1,1] + ridge
    A[2,2] <- A[2,2] + ridge
    A[3,3] <- A[3,3] + ridge
    
    b <- c(sum(ww * x),
           sum(ww * x * C),
           sum(ww * x * S))
    
    coef <- solve(A, b)
    fitted <- coef[1] + coef[2]*C + coef[3]*S
    list(coef=coef, fitted=fitted)
  }
  
  # start: plain WLS
  ww <- w
  fit <- solve_wls(ww)
  rss_old <- sum(w * (x - fit$fitted)^2)
  
  for (it in seq_len(maxit)) {
    r <- x - fit$fitted
    s <- mad(r) + 1e-12      # escala robusta
    z <- r / s
    
    u <- rep(1, length(z))
    idx <- abs(z) > huber_k
    u[idx] <- huber_k / abs(z[idx])
    
    ww <- w * u
    fit <- solve_wls(ww)
    rss_new <- sum(w * (x - fit$fitted)^2)
    
    if (abs(rss_old - rss_new) / (rss_old + 1e-12) < tol) break
    rss_old <- rss_new
  }
  
  c1 <- fit$coef[1]; c2 <- fit$coef[2]; c3 <- fit$coef[3]
  Ahat <- sqrt(c2^2 + c3^2)
  betahat <- atan2(-c3, c2) %% (2*pi)
  
  list(
    parameters = list(M=c1, A=Ahat, beta=betahat,
                      alpha = grid$alpha_vec[ii], omega = grid$omega_vec[ii]),
    fitted_values = fit$fitted,
    rss = sum(w * (x - fit$fitted)^2)
  )
}

# ============================================================
# ---- Helper: evaluate score J(alpha,omega) locally ----
fmm_roughness <- function(alpha, omega, grid,
                          rough_mode = c("observed","dense"),
                          ndense = 512L) {
  rough_mode <- match.arg(rough_mode)
  
  alpha <- alpha %% (2*pi)
  omega <- min(max(omega, 0), 1)
  
  if (rough_mode == "dense") {
    # Use cached dense grid if available and matches ndense; else build it
    if (!is.null(grid$t_dense) && !is.null(grid$ndense) && as.integer(grid$ndense) == as.integer(ndense)) {
      t_dense <- grid$t_dense
    } else {
      t_dense <- seq(0, 2*pi, length.out = ndense + 1L)
      t_dense <- t_dense[-length(t_dense)]
    }
    nd <- length(t_dense)
    dt <- rep(2*pi/nd, nd)
    dt <- pmax(dt, .Machine$double.eps)
    
    cs <- mobius_cos_sin(t = t_dense, alpha = alpha, omega = omega)
    C <- cs$C; S <- cs$S
    
    Cn <- C[c(2:nd, 1)]; Sn <- S[c(2:nd, 1)]
    dC <- Cn - C; dS <- Sn - S
    
    slopeC <- dC / dt; slopeS <- dS / dt
    slopeC_prev <- slopeC[c(nd, 1:(nd-1))]
    slopeS_prev <- slopeS[c(nd, 1:(nd-1))]
    
    ddC <- slopeC - slopeC_prev
    ddS <- slopeS - slopeS_prev
    
    return(sum(ddC*ddC) + sum(ddS*ddS))
  } else {
    # observed-time roughness
    t <- grid$t
    ord <- order(t)
    t_ord <- t[ord]
    n <- length(t_ord)
    
    dt <- c(diff(t_ord), (t_ord[1] + 2*pi) - t_ord[n])
    dt <- pmax(dt, .Machine$double.eps)
    
    cs <- mobius_cos_sin(t = t, alpha = alpha, omega = omega)
    C <- cs$C; S <- cs$S
    
    C_ord <- C[ord]; S_ord <- S[ord]
    Cn <- C_ord[c(2:n, 1)]
    Sn <- S_ord[c(2:n, 1)]
    
    dC <- Cn - C_ord
    dS <- Sn - S_ord
    
    slopeC <- dC / dt
    slopeS <- dS / dt
    slopeC_prev <- slopeC[c(n, 1:(n-1))]
    slopeS_prev <- slopeS[c(n, 1:(n-1))]
    
    ddC <- slopeC - slopeC_prev
    ddS <- slopeS - slopeS_prev
    
    return(sum(ddC*ddC) + sum(ddS*ddS))
  }
}

# ============================================================
# 4) Lambda calibration under H0 + elbow (knee) selection
# ============================================================

#' Circular maximum jump diagnostic
#'
#' @param y numeric vector length n (e.g., fitted values)
#' @param ord integer permutation giving circular time order
#' @return scalar max |y_{i+1}-y_i| in circular order
spike_max_jump <- function(y, ord) {
  y <- as.numeric(y)
  yo <- y[ord]
  diffs <- abs(c(diff(yo), yo[1] - yo[length(yo)]))
  max(diffs, na.rm = TRUE)
}


#' Pick lambda by "elbow" (knee) on a decreasing curve q_spike(lambda)
#'
#' Uses distance-to-line method on x=log10(lambda), with optional moving-average smoothing.
#'
#' @param tab data.frame with columns lambda and q_spike
#' @param smooth_k integer window size for running median (>=1); use 3 for light smoothing
#' @param use_log10 logical; if TRUE, work on log10(lambda)
#' @return list with lambda_star and diagnostic vectors
pick_lambda_elbow <- function(tab, smooth_k = 3L, use_log10 = FALSE) {
  if (!all(c("lambda", "q_spike") %in% names(tab))) stop("tab must have columns: lambda, q_spike")
  lam <- as.numeric(tab$lambda)
  y0  <- as.numeric(tab$q_spike)
  
  if (any(!is.finite(lam)) || any(lam <= 0)) stop("lambda must be finite and > 0.")
  if (any(!is.finite(y0))) stop("q_spike must be finite.")
  
  # order by lambda increasing
  o <- order(lam)
  lam <- lam[o]
  y0  <- y0[o]
  
  x <- if (use_log10) log10(lam) else lam
  y <- runmed_safe(y0, k = smooth_k)
  
  # chord endpoints
  x1 <- x[1]; y1 <- y[1]
  x2 <- x[length(x)]; y2 <- y[length(y)]
  
  # distance from point to chord line
  a <- (y2 - y1)
  b <- -(x2 - x1)
  c <- (x2 - x1) * y1 - (y2 - y1) * x1
  denom <- sqrt(a*a + b*b)
  if (!is.finite(denom) || denom <= 0) stop("Degenerate chord in elbow selection.")
  
  d_chord <- abs(a * x + b * y + c) / denom
  
  # do not allow endpoints as elbow
  d_chord[1] <- -Inf
  d_chord[length(d_chord)] <- -Inf
  
  i_star <- which.max(d_chord)
  
  list(
    lambda_star = lam[i_star],
    index_star  = i_star,
    lambda_order = lam,
    x = x,
    q_spike_raw = y0,
    q_spike_smooth = y,
    dist_to_chord = d_chord
  )
}

#' Calibrate lambda under H0 by Monte Carlo + elbow rule
#'
#' For each lambda in lambda_grid, simulates B null genes under H0, fits the model,
#' computes the spike diagnostic D, and stores q-quantile of D as q_spike(lambda).
#' Then selects lambda_star by the elbow (knee) rule.
#'
#' @param t numeric vector of times (radians)
#' @param weights numeric vector of weights (length n)
#' @param grid output of fmm_grid_precompute()
#' @param cache output of fmm_cache_prepare()
#' @param lambda_grid numeric vector (positive, increasing recommended)
#' @param B integer number of null simulations
#' @param q quantile level for spike diagnostic (e.g., 0.99)
#' @param sigma positive scalar noise sd under H0
#' @param omega0 numeric in (0,1) for g(omega)
#' @param p numeric >=1 for g(omega)
#' @param rough_mode "observed" or "dense"
#' @param ndense integer if rough_mode="dense"
#' @param seed optional integer seed
#' @param smooth_k smoothing window for elbow selection (default 3)
#' center_null = TRUE,
#' reuse_null = TRUE,
#' cache_dense_roughness = TRUE,
#' show_progress = FALSE
#' @return list with lambda_star, table, and settings
fmm_calibrate_lambda_old <- function(t, weights, grid, cache,
                                 lambda_grid = 2^seq(-1, 4, length.out = 24),
                                 B = 500L,
                                 q = 0.99,
                                 sigma = 1,
                                 omega0 = 0.10,
                                 p = 3,
                                 rough_mode = c("observed","dense"),
                                 ndense = 512L,
                                 seed = 1,
                                 smooth_k = 3L,
                                 center_null = TRUE,
                                 reuse_null = TRUE,
                                 cache_dense_roughness = TRUE,
                                 show_progress = FALSE) {
  rough_mode <- match.arg(rough_mode)
  
  # times + weights
  t <- as.numeric(t) %% (2*pi)
  w <- normalize_weights_to_n(as.numeric(weights))
  n <- length(t)
  if (length(w) != n) stop("weights must have same length as t.")
  if (!(is.finite(sigma) && sigma > 0)) stop("sigma must be > 0.")
  if (!(q > 0 && q < 1)) stop("q must be in (0,1).")
  
  B <- as.integer(B)
  if (B < 50) warning("B is small; elbow selection may be noisy.")
  
  # lambda grid
  lambda_grid <- as.numeric(lambda_grid)
  if (any(!is.finite(lambda_grid)) || any(lambda_grid <= 0)) {
    stop("lambda_grid must be finite and > 0.")
  }
  
  # ---- 1) CRN null replicates (fixed across ALL lambdas) ----
  # If reuse_null=FALSE, we still generate CRN but only for current lambda (less useful).
  Xnull <- NULL
  if (!is.null(seed)) set.seed(seed)
  
  if (reuse_null) {
    # CRN: fixed Z -> fixed X for all lambdas
    Z <- matrix(rnorm(B * n), nrow = B, ncol = n)
    Xnull <- sigma * Z
    
    if (center_null) {
      sw <- sum(w)
      if (sw <= 0) stop("sum(weights) must be > 0.")
      muw <- as.numeric(Xnull %*% w) / sw   # length B
      Xnull <- Xnull - muw                 # recycled across columns
    }
  }
  
  # ---- 2) Optionally precompute dense roughness inside grid once ----
  if (rough_mode == "dense" && cache_dense_roughness) {
    x_tmp <- if (reuse_null) Xnull[1, ] else {
      # CRN even when reuse_null=FALSE: draw one replicate deterministically from seed
      Z1 <- rnorm(n)
      x1 <- sigma * Z1
      if (center_null) {
        x1 <- x1 - (sum(w * x1) / sum(w))
      }
      x1
    }
    
    tmp_fit <- fmm_fit_power_rough(
      x = x_tmp, grid = grid, cache = cache,
      center = TRUE, normalize = TRUE,
      lambda = lambda_grid[1], omega0 = omega0, p = p,
      refine = FALSE, keep_trace = FALSE,
      rough_mode = "dense", ndense = ndense,
      return_grid = TRUE
    )
    grid <- tmp_fit$grid
  }
  
  # ---- 3) Evaluate Q(lambda) ----
  qvals <- numeric(length(lambda_grid))
  
  for (L in seq_along(lambda_grid)) {
    lam <- lambda_grid[L]
    if (show_progress) message(sprintf("lambda %d/%d: %.5g", L, length(lambda_grid), lam))
    
    Dvals <- numeric(B)
    
    for (b in seq_len(B)) {
      if (reuse_null) {
        xnull <- Xnull[b, ]
      } else {
        # If you insist on not reusing, still do CRN *within* this lambda:
        # reset seed per lambda so each lambda gets same stream => still CRN across lambdas if you want it.
        # But easiest is just set reuse_null=TRUE.
        if (!is.null(seed)) set.seed(seed + 10000L * L + b)
        xnull <- sigma * rnorm(n)
        if (center_null) xnull <- xnull - (sum(w * xnull) / sum(w))
      }
      
      fit <- fmm_fit_power_rough(
        x = xnull, grid = grid, cache = cache,
        center = TRUE, normalize = TRUE,
        lambda = lam, omega0 = omega0, p = p,
        refine = FALSE, keep_trace = FALSE,
        rough_mode = rough_mode, ndense = ndense,
        return_grid = FALSE
      )
      
      Dvals[b] <- spike_max_jump(fit$fitted_values, ord = grid$ord)
    }
    
    qvals[L] <- as.numeric(stats::quantile(Dvals, probs = q, names = FALSE, na.rm = TRUE))
  }
  
  tab <- data.frame(lambda = lambda_grid, q_spike = qvals)
  
  # ---- 4) Elbow selection ----
  knee <- pick_lambda_elbow(tab, smooth_k = smooth_k, use_log10 = TRUE)
  lam_star <- knee$lambda_star
  
  list(
    lambda_star = lam_star,
    table = tab,
    knee = knee,
    grid = grid,  # return enriched grid if dense roughness cached
    settings = list(
      B = B, q = q, sigma = sigma,
      omega0 = omega0, p = p,
      rough_mode = rough_mode, ndense = ndense,
      lambda_grid = lambda_grid,
      smooth_k = smooth_k,
      seed = seed,
      center_null = center_null,
      reuse_null = reuse_null,
      cache_dense_roughness = cache_dense_roughness
    )
  )
}

fmm_calibrate_lambda_repeat_old <- function(R = 5L,
                                        seed0 = 1L,
                                        agg = c("median","mean"),
                                        ...) {
  agg <- match.arg(agg)
  R <- as.integer(R)
  
  outs <- vector("list", R)
  for (r in seq_len(R)) {
    outs[[r]] <- fmm_calibrate_lambda(seed = seed0 + r - 1L, ...)
  }
  
  # All must share same lambda grid
  lambda <- outs[[1]]$table$lambda
  Qmat <- do.call(cbind, lapply(outs, function(o) o$table$q_spike))
  
  Qhat <- if (agg == "mean") rowMeans(Qmat) else apply(Qmat, 1, stats::median)
  
  # SE across repetitions (for 1-SE elbow or reporting uncertainty)
  Qsd <- apply(Qmat, 1, stats::sd)
  Qse <- Qsd / sqrt(R)
  
  tab_agg <- data.frame(lambda = lambda, q_spike = Qhat, se = Qse)
  
  knee <- pick_lambda_elbow(tab_agg, smooth_k = outs[[1]]$settings$smooth_k,
                            use_log10 = TRUE)
  
  list(
    lambda_star = knee$lambda_star,
    table = tab_agg,
    knee = knee,
    reps = outs,
    settings = outs[[1]]$settings)
}


### Usando el cpp
fmm_calibrate_lambda <- function(t, weights, grid, cache,
                                 lambda_grid = 2^seq(-1, 4, length.out = 24),
                                 B = 500L,
                                 q = 0.99,
                                 sigma = 1,
                                 omega0 = 0.10,
                                 p = 3,
                                 rough_mode = c("observed","dense"),
                                 ndense = 512L,
                                 seed = 1,
                                 smooth_k = 3L,
                                 center_null = TRUE,
                                 reuse_null = TRUE,
                                 cache_dense_roughness = TRUE,
                                 show_progress = FALSE,
                                 parallel_lambda = FALSE,
                                 lambda_workers = NULL,
                                 lambda_block_size = NULL,
                                 cpp_file_lambda = NULL,
                                 cluster = NULL,
                                 backend = c("serial", "parallel"),
                                 grain_size = 10L) {
  
  rough_mode <- match.arg(rough_mode)
  backend <- match.arg(backend)
  
  # ---- checks ----
  t <- as.numeric(t) %% (2*pi)
  w <- normalize_weights_to_n(as.numeric(weights))
  n <- length(t)
  
  if (length(w) != n) stop("weights must have same length as t.")
  if (!(is.finite(sigma) && sigma > 0)) stop("sigma must be > 0.")
  if (!(q > 0 && q < 1)) stop("q must be in (0,1).")
  
  B <- as.integer(B)
  if (B < 50) warning("B is small; elbow selection may be noisy.")
  
  lambda_grid <- as.numeric(lambda_grid)
  if (any(!is.finite(lambda_grid)) || any(lambda_grid <= 0)) {
    stop("lambda_grid must be finite and > 0.")
  }
  
  # ---- 1) CRN null replicates ----
  Xnull <- NULL
  if (!is.null(seed)) set.seed(seed)
  
  if (reuse_null) {
    Z <- matrix(rnorm(B * n), nrow = B, ncol = n)
    Xnull <- sigma * Z
    
    if (center_null) {
      sw <- sum(w)
      if (sw <= 0) stop("sum(weights) must be > 0.")
      muw <- as.numeric(Xnull %*% w) / sw
      Xnull <- Xnull - muw
    }
  }
  
  # ---- 2) optional dense roughness cache inside grid ----
  if (rough_mode == "dense" && cache_dense_roughness) {
    x_tmp <- if (reuse_null) {
      Xnull[1, ]
    } else {
      Z1 <- rnorm(n)
      x1 <- sigma * Z1
      if (center_null) x1 <- x1 - (sum(w * x1) / sum(w))
      x1
    }
    
    tmp_fit <- fmm_fit_power_rough_cpp(
      x = x_tmp, grid = grid, cache = cache,
      backend = backend,
      center = TRUE, normalize = TRUE,
      lambda = lambda_grid[1], omega0 = omega0, p = p,
      refine = FALSE,
      rough_mode = "dense", ndense = ndense,
      return_grid = TRUE,
      grain_size = grain_size
    )
    grid <- tmp_fit$grid
  }
  
  # ---- helper: evaluate one lambda ----
  eval_one_lambda <- function(lam, Xnull, grid, cache, w, q, omega0, p,
                              rough_mode, ndense, B, sigma, center_null, seed, lam_index,
                              backend, grain_size) {
    n <- if (!is.null(Xnull)) ncol(Xnull) else length(w)
    Dvals <- numeric(B)
    
    for (b in seq_len(B)) {
      if (!is.null(Xnull)) {
        xnull <- Xnull[b, ]
      } else {
        if (!is.null(seed)) set.seed(seed + 10000L * lam_index + b)
        xnull <- sigma * rnorm(n)
        if (center_null) xnull <- xnull - (sum(w * xnull) / sum(w))
      }
      
      fit <- fmm_fit_power_rough_cpp(
        x = xnull, grid = grid, cache = cache,
        backend = backend,
        center = TRUE, normalize = TRUE,
        lambda = lam, omega0 = omega0, p = p,
        refine = FALSE,
        rough_mode = rough_mode, ndense = ndense,
        return_grid = FALSE,
        grain_size = grain_size
      )
      
      Dvals[b] <- spike_max_jump(fit$fitted_values, ord = grid$ord)
    }
    
    as.numeric(stats::quantile(Dvals, probs = q, names = FALSE, na.rm = TRUE))
  }
  
  # ---- 3) evaluate q_spike(lambda) ----
  qvals <- numeric(length(lambda_grid))
  
  if (!parallel_lambda || length(lambda_grid) == 1L) {
    
    for (L in seq_along(lambda_grid)) {
      lam <- lambda_grid[L]
      if (show_progress) {
        message(sprintf("lambda %d/%d: %.5g", L, length(lambda_grid), lam))
      }
      
      qvals[L] <- eval_one_lambda(
        lam = lam,
        Xnull = if (reuse_null) Xnull else NULL,
        grid = grid,
        cache = cache,
        w = w,
        q = q,
        omega0 = omega0,
        p = p,
        rough_mode = rough_mode,
        ndense = ndense,
        B = B,
        sigma = sigma,
        center_null = center_null,
        seed = seed,
        lam_index = L,
        backend = backend,
        grain_size = grain_size
      )
    }
    
  } else {
    
    if (is.null(lambda_workers)) {
      lambda_workers <- max(1L, min(4L, parallel::detectCores() - 1L))
    }
    lambda_workers <- max(1L, as.integer(lambda_workers))
    
    created_here <- FALSE
    cl <- cluster
    
    if (is.null(cl)) {
      if (is.null(cpp_file_lambda)) {
        stop("If parallel_lambda=TRUE and cluster=NULL, you must provide cpp_file_lambda.")
      }
      cl <- init_fmm_cluster(
        workers = lambda_workers,
        cpp_file = cpp_file_lambda,
        backend = backend,
        num_threads_cpp = 1L
      )
      created_here <- TRUE
    }
    
    on.exit({
      if (created_here) stop_fmm_cluster(cl)
    }, add = TRUE)
    
    if (is.null(lambda_block_size)) {
      lambda_block_size <- ceiling(length(lambda_grid) / lambda_workers)
      lambda_block_size <- max(1L, as.integer(lambda_block_size))
    } else {
      lambda_block_size <- max(1L, as.integer(lambda_block_size))
    }
    
    lambda_idx <- seq_along(lambda_grid)
    lambda_blocks <- split(lambda_idx, ceiling(lambda_idx / lambda_block_size))
    
    parallel::clusterExport(
      cl,
      varlist = c(
        "lambda_grid", "Xnull", "grid", "cache", "w", "q",
        "omega0", "p", "rough_mode", "ndense", "B", "sigma",
        "center_null", "seed", "eval_one_lambda",
        "spike_max_jump", "fmm_fit_power_rough_cpp",
        "backend", "grain_size"
      ),
      envir = environment()
    )
    
    block_fun <- function(ix) {
      out <- numeric(length(ix))
      for (k in seq_along(ix)) {
        L <- ix[k]
        out[k] <- eval_one_lambda(
          lam = lambda_grid[L],
          Xnull = if (!is.null(Xnull)) Xnull else NULL,
          grid = grid,
          cache = cache,
          w = w,
          q = q,
          omega0 = omega0,
          p = p,
          rough_mode = rough_mode,
          ndense = ndense,
          B = B,
          sigma = sigma,
          center_null = center_null,
          seed = seed,
          lam_index = L,
          backend = backend,
          grain_size = grain_size
        )
      }
      out
    }
    
    qvals_list <- parallel::parLapply(cl, lambda_blocks, block_fun)
    qvals <- unlist(qvals_list, use.names = FALSE)
  }
  
  tab <- data.frame(lambda = lambda_grid, q_spike = qvals)
  
  # ---- 4) elbow selection ----
  knee <- pick_lambda_elbow(tab, smooth_k = smooth_k, use_log10 = TRUE)
  lam_star <- knee$lambda_star
  
  list(
    lambda_star = lam_star,
    table = tab,
    knee = knee,
    grid = grid,
    settings = list(
      B = B,
      q = q,
      sigma = sigma,
      omega0 = omega0,
      p = p,
      rough_mode = rough_mode,
      ndense = ndense,
      lambda_grid = lambda_grid,
      smooth_k = smooth_k,
      seed = seed,
      center_null = center_null,
      reuse_null = reuse_null,
      cache_dense_roughness = cache_dense_roughness,
      parallel_lambda = parallel_lambda,
      lambda_workers = lambda_workers,
      lambda_block_size = lambda_block_size,
      backend = backend,
      grain_size = grain_size
    )
  )
}

fmm_calibrate_lambda_repeat <- function(R = 5L,
                                        seed0 = 1L,
                                        agg = c("median","mean"),
                                        ...,
                                        backend = c("serial", "parallel"),
                                        grain_size = 10L) {
  agg <- match.arg(agg)
  backend <- match.arg(backend)
  R <- as.integer(R)
  
  outs <- vector("list", R)
  for (r in seq_len(R)) {
    outs[[r]] <- fmm_calibrate_lambda(
      seed = seed0 + r - 1L,
      backend = backend,
      grain_size = grain_size,
      ...
    )
  }
  
  lambda <- outs[[1]]$table$lambda
  Qmat <- do.call(cbind, lapply(outs, function(o) o$table$q_spike))
  
  Qhat <- if (agg == "mean") {
    rowMeans(Qmat)
  } else {
    apply(Qmat, 1, stats::median)
  }
  
  Qsd <- apply(Qmat, 1, stats::sd)
  Qse <- Qsd / sqrt(R)
  
  tab_agg <- data.frame(lambda = lambda, q_spike = Qhat, se = Qse)
  
  knee <- pick_lambda_elbow(
    tab_agg,
    smooth_k = outs[[1]]$settings$smooth_k,
    use_log10 = TRUE
  )
  
  list(
    lambda_star = knee$lambda_star,
    table = tab_agg,
    knee = knee,
    reps = outs,
    settings = outs[[1]]$settings
  )
}

# -----------------------------
# 5) Time-density weights via von Mises KDE + LOO CV
# -----------------------------
#' von Mises kernel on the circle
#'
#' K_kappa(delta) = exp(kappa cos(delta)) / (2*pi*I0(kappa))
#'
#' @param delta numeric matrix/vector of angular differences in (-pi,pi]
#' @param kappa positive scalar concentration
#' @return numeric matrix/vector kernel values
vm_kernel <- function(delta, kappa) {
  (1 / (2*pi*besselI(kappa, 0))) * exp(kappa * cos(delta))
}

#' LOO log-likelihood for von Mises KDE on observed times
#'
#' Computes LCV(kappa) = sum_i log(f_{-i,kappa}(t_i)).
#'
#' @param times numeric vector in radians (will be wrapped to [0,2*pi))
#' @param kappa positive scalar
#' @param eps small floor to avoid log(0)
#' @return scalar LOO-CV score
fmm_lcv_kappa <- function(times, kappa, eps = 1e-12) {
  times <- as.numeric(times) %% (2*pi)
  n <- length(times)
  if (n < 2) stop("Need n>=2 for LOO CV.")
  if (!(is.finite(kappa) && kappa > 0)) stop("kappa must be > 0.")
  
  Delta <- outer(times, times, function(a, b) angdiff(a - b))
  K <- vm_kernel(Delta, kappa)
  diag(K) <- 0  # leave-one-out
  f_loo <- rowSums(K) / (n - 1)
  sum(log(pmax(f_loo, eps)))
}

#' Select kappa by LOO-CV on a user-supplied grid
#'
#' @param times numeric vector in radians
#' @param Kgrid numeric vector of positive candidate kappas
#' @param eps small floor for LCV computation
#' @return list with kappa (selected), lcv (scores), Kgrid
fmm_select_kappa_lcv <- function(times,
                                 Kgrid = exp(seq(log(1), log(64), length.out = 10)),
                                 eps = 1e-12) {
  
  Kgrid <- as.numeric(Kgrid)
  if (any(!is.finite(Kgrid)) || any(Kgrid <= 0)) stop("Kgrid must be finite and > 0.")
  lcv <- vapply(Kgrid, function(k) fmm_lcv_kappa(times, kappa = k, eps = eps), numeric(1))
  idx <- which.max(lcv)
  list(kappa = Kgrid[idx], lcv = lcv, Kgrid = Kgrid)
}

#' Compute time-density weights w_i proportional to KDE(t_i)
#'
#' Weights are normalized to sum(w)=n.
#'
#' @param times numeric vector in radians (will be wrapped to [0,2*pi))
#' @param kappa positive scalar concentration
#' @param eps small floor for numerical stability
#' @return list with w_raw and w (normalized to sum(w)=n)
fmm_time_weights <- function(times, kappa, eps = 1e-12) {
  
  times <- as.numeric(times) %% (2*pi)
  n <- length(times)
  if (n < 1) stop("times must have length >= 1.")
  if (!(is.finite(kappa) && kappa > 0)) stop("kappa must be > 0.")
  
  Delta <- outer(times, times, function(a, b) angdiff(a - b))
  K <- vm_kernel(Delta, kappa)
  fhat <- rowSums(K) / n
  fhat_floor <- pmax(fhat, eps)
  w_raw <- 1 / fhat_floor
    
  w <- normalize_weights_to_n(w_raw)
  
  #list(w_raw = fhat, w = w, kappa = kappa)
  list(fhat = fhat, w_raw = w_raw, w = w, kappa = kappa)
 
}
