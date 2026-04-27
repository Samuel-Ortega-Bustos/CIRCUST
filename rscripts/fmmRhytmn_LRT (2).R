#' Prepare dataset
#'
#' This function:
#'  (1) builds (or accepts) time-density weights,
#'  (2) precomputes the (alpha, omega) grid and WLS cache,
#'  (3) calibrates lambda* if not provided (CRN + repetitions),
#'
#' @param times numeric vector of times (radians). Will be wrapped to [0,2*pi).
#' @param w     optional numeric weights (length n). If NULL, weights are computed from kappa.
#' @param lambda_star optional scalar. If NULL, lambda* is calibrated via Monte Carlo.
#' @param Kgrid grid of kappa candidates for fmm_select_kappa_lcv().
#' @param alpha_grid alpha grid for selection.
#' @param omega_grid omega grid for selection.
#'
#' @param R number of calibration repetitions (only if lambda_star is NULL).
#' @param agg aggregation for repeated calibration ("median" or "mean").
#' @param lambda_grid candidate lambdas for calibration.
#' @param q quantile level for the spike diagnostic in calibration.
#' @param sigma_l noise sd under H0 for calibration (usually 1).
#' @param B_l number of null replicates per repetition.
#' @param smooth_k running-median window used in elbow selection.
#'
#'  @return  a list with weights, lambda_star, grid, cache and configuration parameters.
fmm_prepare_lrt_old <- function(times,
                            w = NULL,
                            lambda_star = NULL,
                            Kgrid = exp(seq(log(1), log(64), length.out = 10)),
                            alpha_grid = seq(0, 2*pi, length.out = 48),
                            omega_grid = exp(seq(log(1e-4), log(1), length.out = 30)),
                            # lambda calibration (used only if lambda_star is NULL)
                            R = 5L,
                            agg = c("median", "mean"),
                            lambda_grid = 2^seq(-1, 4, length.out = 24),
                            q = 0.975,
                            sigma_l = 1,
                            B_l = 1000L,
                            smooth_k = 3L,
                            rough_mode = c("dense", "observed"),
                            ndense = 512L,
                            omega0 = 0.10,
                            p = 3
                          ) {
  
  # ---- Checks ----
  agg <- match.arg(agg)
  rough_mode <- match.arg(rough_mode)
  
  times <- as.numeric(times) %% (2*pi)
  
  n <- length(times)
  if (any(!is.finite(times))) stop("Non-finite values in times.")
  
  # ---- 1) Time-density weights ----
  if (is.null(w)) {
    k_sel <- fmm_select_kappa_lcv(times, Kgrid = Kgrid)
    w_obj <- fmm_time_weights(times, kappa = k_sel$kappa)
    w_use <- normalize_weights_to_n(w_obj$w)
  } else {
    w_use <- normalize_weights_to_n(as.numeric(w))
  }
  if (length(w_use) != n) stop("weights must have length n.")
  if (any(!is.finite(w_use)) || any(w_use < 0)) stop("weights must be finite and nonnegative.")
  
  # ---- 2) Precompute grid ----
  grid <- fmm_grid_precompute(times, alpha_grid = alpha_grid, omega_grid = omega_grid)
  grid <- fmm_grid_add_dense_roughness(grid, ndense = ndense)
  
  # ---- 3) Cache for WLS (weights fixed) ----
  cache <- fmm_cache_prepare(w_use, grid)
  
  # ---- 4) Calibrate lambda* if needed ----
  cal <- NULL
  if (is.null(lambda_star)) {
    cal <- fmm_calibrate_lambda_repeat(
      R = as.integer(R),
      seed0 = 1L,
      agg = agg,
      t = times, weights = w_use,
      grid = grid, cache = cache,
      lambda_grid = lambda_grid,
      B = as.integer(B_l),
      q = q,
      sigma = sigma_l,
      omega0 = omega0,
      p = p,
      rough_mode = rough_mode,
      ndense = as.integer(ndense),
      smooth_k = as.integer(smooth_k)
    )
    lambda_star <- cal$lambda_star
  } else {
    lambda_star <- as.numeric(lambda_star)
    if (!is.finite(lambda_star) || lambda_star < 0) stop("lambda_star must be finite and >= 0.")
  }
  
  out <- list(
    weights = w_use,
    n = n,
    lambda_star = lambda_star,
    grid = grid,
    cache = cache,
    settings = list(
      times = times,
      Kgrid = Kgrid,
      alpha_grid = alpha_grid,
      omega_grid = omega_grid,
      R = R,
      agg = agg,
      lambda_grid = lambda_grid,
      q = q,
      sigma_l = sigma_l,
      B_l = B_l,
      smooth_k = smooth_k
    )
  )
  if (!is.null(cal)) out$calibration <- cal
  if (!is.null(w) && !is.null(cal)) out$note <- "weights were provided by user; lambda_star was calibrated using those weights."
  out
}

# usando el cpp
fmm_prepare_lrt <- function(times,
                            w = NULL,
                            lambda_star = NULL,
                            Kgrid = exp(seq(log(1), log(64), length.out = 10)),
                            alpha_grid = seq(0, 2*pi, length.out = 48),
                            omega_grid = exp(seq(log(1e-4), log(1), length.out = 30)),
                            R = 5L,
                            agg = c("median", "mean"),
                            lambda_grid = 2^seq(-1, 4, length.out = 24),
                            q = 0.975,
                            sigma_l = 1,
                            B_l = 1000L,
                            smooth_k = 3L,
                            rough_mode = c("dense", "observed"),
                            ndense = 512L,
                            omega0 = 0.10,
                            p = 3,
                            use_cpp_lambda = TRUE,
                            require_cpp_lambda = FALSE,
                            parallel_lambda = FALSE,
                            lambda_workers = NULL,
                            lambda_block_size = NULL,
                            cpp_file_lambda = NULL,
                            cluster_lambda = NULL,
                            backend = c("serial", "parallel"),
                            grain_size = 10L) {
  
  agg <- match.arg(agg)
  rough_mode <- match.arg(rough_mode)
  backend <- match.arg(backend)
  
  times <- as.numeric(times) %% (2*pi)
  n <- length(times)
  
  if (any(!is.finite(times))) stop("Non-finite values in times.")
  
  # ---- 1) Time-density weights ----
  if (is.null(w)) {
    k_sel <- fmm_select_kappa_lcv(times, Kgrid = Kgrid)
    w_obj <- fmm_time_weights(times, kappa = k_sel$kappa)
    w_use <- normalize_weights_to_n(w_obj$w)
  } else {
    w_use <- normalize_weights_to_n(as.numeric(w))
  }
  
  if (length(w_use) != n) stop("weights must have length n.")
  if (any(!is.finite(w_use)) || any(w_use < 0)) {
    stop("weights must be finite and nonnegative.")
  }
  
  # ---- 2) Precompute grid ----
  grid <- fmm_grid_precompute(times, alpha_grid = alpha_grid, omega_grid = omega_grid)
  grid <- fmm_grid_add_dense_roughness(grid, ndense = ndense)
  
  # ---- 3) Cache for WLS (weights fixed) ----
  cache <- fmm_cache_prepare(w_use, grid)
  
  # ---- 4) Calibrate lambda* if needed ----
  cal <- NULL
  
  if (is.null(lambda_star)) {
    
    cpp_lambda_available <- exists("fmm_fit_power_rough_cpp", mode = "function")
    
    if (isTRUE(use_cpp_lambda) && !cpp_lambda_available) {
      msg <- paste(
        "C++ lambda calibration requested, but",
        "'fmm_fit_power_rough_cpp' is not available.",
        "Compile/load the C++ backend first."
      )
      if (isTRUE(require_cpp_lambda)) {
        stop(msg)
      } else {
        warning(msg, " Falling back to the R version of calibration.")
      }
    }
    
    cal <- fmm_calibrate_lambda_repeat(
      R = as.integer(R),
      seed0 = 1L,
      agg = agg,
      t = times,
      weights = w_use,
      grid = grid,
      cache = cache,
      lambda_grid = lambda_grid,
      B = as.integer(B_l),
      q = q,
      sigma = sigma_l,
      omega0 = omega0,
      p = p,
      rough_mode = rough_mode,
      ndense = as.integer(ndense),
      smooth_k = as.integer(smooth_k),
      parallel_lambda = isTRUE(parallel_lambda),
      lambda_workers = lambda_workers,
      lambda_block_size = lambda_block_size,
      cpp_file_lambda = cpp_file_lambda,
      cluster = cluster_lambda,
      backend = backend,
      grain_size = grain_size
    )
    
    lambda_star <- cal$lambda_star
    
    if (!is.null(cal$reps) && length(cal$reps) > 0 && !is.null(cal$reps[[1]]$grid)) {
      grid <- cal$reps[[1]]$grid
      cache <- fmm_cache_prepare(w_use, grid)
    } else if (!is.null(cal$grid)) {
      grid <- cal$grid
      cache <- fmm_cache_prepare(w_use, grid)
    }
    
  } else {
    lambda_star <- as.numeric(lambda_star)
    if (!is.finite(lambda_star) || lambda_star < 0) {
      stop("lambda_star must be finite and >= 0.")
    }
  }
  
  out <- list(
    weights = w_use,
    n = n,
    lambda_star = lambda_star,
    grid = grid,
    cache = cache,
    settings = list(
      times = times,
      Kgrid = Kgrid,
      alpha_grid = alpha_grid,
      omega_grid = omega_grid,
      R = R,
      agg = agg,
      lambda_grid = lambda_grid,
      q = q,
      sigma_l = sigma_l,
      B_l = B_l,
      smooth_k = smooth_k,
      rough_mode = rough_mode,
      ndense = ndense,
      omega0 = omega0,
      p = p,
      use_cpp_lambda = use_cpp_lambda,
      parallel_lambda = parallel_lambda,
      lambda_workers = lambda_workers,
      lambda_block_size = lambda_block_size,
      cpp_file_lambda = cpp_file_lambda,
      backend = backend,
      grain_size = grain_size
    )
  )
  
  if (!is.null(cal)) out$calibration <- cal
  if (!is.null(w) && !is.null(cal)) {
    out$note <- "weights were provided by user; lambda_star was calibrated using those weights."
  }
  
  out
}

#' Cache dense roughness for FMM grid (speed-up for rough_mode = "dense")
#'
#' Precomputes and stores the dense curvature roughness term for each grid point
#' \eqn{(\alpha,\omega)} on a uniform dense time grid of length \code{ndense}.
#' This avoids recomputing the dense roughness inside \code{fmm_fit_power_rough()}
#' at every bootstrap replicate, which is typically the main bottleneck.
#'
#' The function adds/overwrites these fields in \code{grid}:
#' \itemize{
#'   \item \code{rough_dense}: numeric vector length G (G = nrow(grid$C))
#'   \item \code{t_dense}: numeric vector length ndense (uniform grid on [0,2*pi))
#'   \item \code{ndense}: integer, the dense grid size used
#' }
#'
#' @param grid List returned by \code{fmm_grid_precompute()}.
#'   Must contain \code{alpha_vec} and \code{omega_vec}.
#' @param ndense Integer. Number of equally spaced points in \eqn{[0,2\pi)}.
#' @param den_eps Small constant for numerical stability passed to \code{mobius_cos_sin()}.
#'
#' @return The input \code{grid} enriched with \code{rough_dense}, \code{t_dense} and \code{ndense}.
#'
fmm_grid_add_dense_roughness <- function(grid, ndense = 512L, den_eps = .Machine$double.eps) {
  
  ndense <- as.integer(ndense)
  if (ndense < 16L) stop("ndense must be >= 16.")
  if (is.null(grid$alpha_vec) || is.null(grid$omega_vec)) {
    stop("grid must contain alpha_vec and omega_vec.")
  }
  
  
  t_dense <- seq(0, 2*pi, length.out = ndense + 1L)
  t_dense <- t_dense[-length(t_dense)]
  nd <- length(t_dense)
  dt <- rep(2*pi/nd, nd)
  dt <- pmax(dt, .Machine$double.eps)
  
  G <- length(grid$alpha_vec)
  R_dense <- numeric(G)
  
  for (g in seq_len(G)) {
    cs <- mobius_cos_sin(t = t_dense, alpha = grid$alpha_vec[g], omega = grid$omega_vec[g], den_eps = den_eps)
    C <- cs$C; S <- cs$S
    
    Cn <- C[c(2:nd, 1)]; Sn <- S[c(2:nd, 1)]
    dC <- Cn - C; dS <- Sn - S
    
    slopeC <- dC / dt; slopeS <- dS / dt
    slopeC_prev <- slopeC[c(nd, 1:(nd-1))]
    slopeS_prev <- slopeS[c(nd, 1:(nd-1))]
    
    ddC <- slopeC - slopeC_prev
    ddS <- slopeS - slopeS_prev
    
    R_dense[g] <- sum(ddC*ddC) + sum(ddS*ddS)
  }
  
  grid$rough_dense <- R_dense
  grid$t_dense <- t_dense
  grid$ndense <- ndense
  grid
}



#' Sequential BC bootstrap p-value for the FMM LRT (single stage)
#'
#' Computes a bootstrap p-value for the likelihood-ratio-type statistic
#' \eqn{LRT = n \log((WRSS_0 + eps)/(WRSS_1 + eps))} where \eqn{WRSS_0} is the
#' weighted residual sum of squares under the null (intercept-only) model and
#' \eqn{WRSS_1} is the weighted RSS from the stabilized FMM1 fit
#' \code{fmm_fit_power_rough()} under the alternative.
#'
#' The bootstrap uses a wild bootstrap wilds residuals (Rademacher sign flip),
#' generated by \code{bootstrap_wild_residuals(expr, mu0, w)} where \code{mu0} is
#' the weighted mean under H0.
#'
#' The p-value is estimated sequentially using the Besag--Clifford stopping rule:
#' stop when either:
#' \itemize{
#'   \item \code{S == Smax} (enough exceedances) or
#'   \item \code{N == Nmax} (budget exhausted).
#' }
#' where \code{S} is the number of bootstrap LRTs \eqn{\ge LRT_obs} and \code{N} is
#' the number of generated replicates. The returned p-value is
#' \eqn{(S+1)/(N+1)} (finite-sample conservative correction).
#'
#' @param expr Numeric vector of expression values (length n).
#' @param w Numeric vector of nonnegative weights (length n), typically normalized to sum(w)=n.
#' @param grid Grid list from \code{fmm_grid_precompute()}, ideally enriched with dense roughness
#'   via \code{fmm_grid_add_dense_roughness()} when \code{rough_mode="dense"}.
#' @param cache Cache from \code{fmm_cache_prepare(w, grid)}.
#' @param lambda_star Nonnegative scalar lambda used in \code{fmm_fit_power_rough()}.
#' @param omega0,p Parameters of the penalty factor \eqn{g(\omega)=(\omega0/(\omega+\omega0))^p}.
#' @param rough_mode Character: \code{"dense"} or \code{"observed"} (passed to \code{fmm_fit_power_rough()}).
#' @param ndense Integer dense grid size (used only if \code{rough_mode="dense"}).
#' @param refine Logical. Whether to refine (alpha, omega) locally in \code{fmm_fit_power_rough()}.
#'   For speed, set FALSE in bootstrap-based testing.
#' @param refine_steps,alpha_step,omega_step_mult,ridge_refine Refinement controls (if refine=TRUE).
#' @param omega_min,omega_max Bounds for omega in refinement.
#' @param eps Small constant to stabilize log-ratio and avoid log(0).
#' @param Smax Integer. Stop after \code{Smax} exceedances.
#' @param Nmax Integer. Maximum number of bootstrap replicates.
#' @param store_LRTb Logical. If TRUE, stores all generated bootstrap LRTs (costs memory).
#'
#' @return A list with elements:
#' \itemize{
#'   \item \code{pvalue}: (S+1)/(N+1)
#'   \item \code{S}: number of exceedances
#'   \item \code{N}: number of replicates generated
#'   \item \code{LRT_b}: numeric vector of bootstrap LRTs (only if store_LRTb=TRUE)
#' }
#'
fmm_bc_bootstrap_lrt <- function(expr, w, grid, cache, lambda_star,
                                 omega0 = 0.10, p = 3,
                                 rough_mode = c("dense", "observed"),
                                 ndense = 512L,
                                 refine = FALSE,
                                 refine_steps = 3L,
                                 alpha_step = (2*pi)/48,
                                 omega_step_mult = 1.25,
                                 omega_min = 1e-4,
                                 omega_max = 1,
                                 ridge_refine = 1e-10,
                                 eps = 1e-12,
                                 Smax = 20L, Nmax = 10000L,
                                 store_LRTb = FALSE,
                                 penalized = FALSE) {
  
  rough_mode <- match.arg(rough_mode)
  expr <- as.numeric(expr)
  w <- as.numeric(w)
  n <- length(expr)
  
  if (length(w) != n) stop("w must have same length as expr.")
  if (any(!is.finite(expr)) || any(!is.finite(w))) stop("Non-finite values in expr/w.")
  if (any(w < 0)) stop("Weights must be nonnegative.")
  if (!(is.finite(lambda_star) && lambda_star >= 0)) stop("lambda_star must be finite and >= 0.")
  Smax <- as.integer(Smax); Nmax <- as.integer(Nmax)
  if (Smax < 1L || Nmax < 1L) stop("Smax and Nmax must be >= 1.")
  
  # Guard: ensure dense roughness is cached if requested
  if (rough_mode == "dense") {
    if (is.null(grid$rough_dense) || is.null(grid$t_dense) || is.null(grid$ndense)) {
      stop("rough_mode='dense' but grid$rough_dense is not cached. Run fmm_grid_add_dense_roughness() first.")
    }
    if (as.integer(grid$ndense) != as.integer(ndense)) {
      stop(sprintf("ndense mismatch: grid cached ndense=%d but ndense=%d was requested.",
                   as.integer(grid$ndense), as.integer(ndense)))
    }
  }
  
  # Observed LRT
  mu0 <- sum(w * expr) / sum(w)
  wrss0 <- sum(w * (expr - mu0)^2)
  
  fitH1 <- fmm_fit_power_rough(
    x = expr, grid = grid, cache = cache,
    center = TRUE, normalize = TRUE,
    lambda = lambda_star, omega0 = omega0, p = p,
    refine = refine,
    refine_steps = as.integer(refine_steps),
    alpha_step = alpha_step,
    omega_step_mult = omega_step_mult,
    omega_min = omega_min, omega_max = omega_max,
    ridge_refine = ridge_refine,
    keep_trace = FALSE,
    rough_mode = rough_mode,
    ndense = as.integer(ndense),
    return_grid = FALSE
  )
  wrss1 <- as.numeric(fitH1$rss)
  pen1 <- fitH1$penalty_sel
  if(!penalized) {
    #LRT_obs <- max(0, n * log((wrss0 + eps) / (wrss1 + eps)))
    LRT_obs <- max(0, n * log((wrss0) / (wrss1)))
  } else {  
    #LRT_obs <- n * log((wrss0 + eps) / (wrss1 + pen1 + eps))
    LRT_obs <- n * log((wrss0) / (wrss1 + pen1))
  }  
  
  # Sequential BC bootstrap
  S <- 0L; N <- 0L
  LRT_b <- if (store_LRTb) numeric(Nmax) else NULL
  mu0_vec <- rep(mu0, n)
  
  while (N < Nmax && S < Smax) {
    #x_b <- bootstrap_whitened_residuals(expr, mu0_vec, w, center_w = TRUE)
    x_b <- bootstrap_wild_residuals(expr, mu0_vec, w, center_w = TRUE)
    fitH1_b <- fmm_fit_power_rough(
      x = x_b, grid = grid, cache = cache,
      center = TRUE, normalize = TRUE,
      lambda = lambda_star, omega0 = omega0, p = p,
      refine = refine,
      refine_steps = as.integer(refine_steps),
      alpha_step = alpha_step,
      omega_step_mult = omega_step_mult,
      omega_min = omega_min, omega_max = omega_max,
      ridge_refine = ridge_refine,
      keep_trace = FALSE,
      rough_mode = rough_mode,
      ndense = as.integer(ndense),
      return_grid = FALSE
    )
    wrss1_b <- as.numeric(fitH1_b$rss)
    
    mu_b <- sum(w * x_b) / sum(w)
    wrss0_b <- sum(w * (x_b - mu_b)^2)
    pen1_b <- fitH1_b$penalty_sel
    
    if(!penalized) {
      #lrt_b <- max(0, n * log((wrss0_b + eps) / (wrss1_b + eps)))
      lrt_b <- max(0, n * log((wrss0_b) / (wrss1_b)))
    } else {  
      #lrt_b <- n * log((wrss0_b + eps) / (wrss1_b + pen1_b + eps))
      lrt_b <- n * log((wrss0_b) / (wrss1_b + pen1_b))
    }  
    
    N <- N + 1L
    if (store_LRTb) LRT_b[N] <- lrt_b
    if (lrt_b >= LRT_obs) S <- S + 1L
  }
  
  p_b <- (S + 1) / (N + 1)
  
  out <- list(pvalue = p_b, S = S, N = N, LRT_obs = LRT_obs)
  if (store_LRTb) out$LRT_b <- LRT_b[1:N]
  out
}


#' Two-stage sequential BC bootstrap p-value for the FMM LRT (handles "borderline" genes)
#'
#' Runs a fast preliminary BC stage and only "promotes" genes that look potentially
#' significant to a longer second stage. This is a practical strategy to reduce compute
#' time and to resolve borderline p-values by spending more replicates only
#' where needed.
#'
#' Stage 1 runs \code{fmm_bc_bootstrap_lrt()} with \code{(N1,S1)}. If it accumulates
#' \code{S1} exceedances quickly, the p-value is clearly not small and we stop.
#' Otherwise, if the number of exceedances after stage 1 is \code{<= promote_if_S_le},
#' we run stage 2 with larger \code{(N2,S2)} to obtain a more decisive (and possibly
#' smaller) p-value.
#'
#' This rule is simple, works well in large-scale testing, and concentrates effort on
#' genes near the decision boundary.
#'
#' @inheritParams fmm_bc_bootstrap_lrt
#' @param N1,S1 Integers. Stage-1 limits (replicates/exceedances).
#' @param N2,S2 Integers. Stage-2 limits (replicates/exceedances).
#' @param promote_if_S_le Integer. Promote to stage 2 only if stage-1 exceedances \code{S <= promote_if_S_le}.
#'
#' @return A list like \code{fmm_bc_bootstrap_lrt()}, with extra fields:
#' \itemize{
#'   \item \code{stage}: 1 or 2
#'   \item \code{stage1}: results from stage 1
#' }
#'
fmm_bc_two_stage_lrt <- function(expr, w, grid, cache, lambda_star,
                                 omega0 = 0.10, p = 3,
                                 rough_mode = c("dense", "observed"),
                                 ndense = 512L,
                                 refine = FALSE,
                                 refine_steps = 3L,
                                 alpha_step = (2*pi)/48,
                                 omega_step_mult = 1.25,
                                 omega_min = 1e-4,
                                 omega_max = 1,
                                 ridge_refine = 1e-10,
                                 eps = 1e-12,
                                 # stage 1
                                 N1 = 200L, S1 = 10L,
                                 # promotion rule
                                 promote_if_S_le = 2L,
                                 # stage 2
                                 N2 = 5000L, S2 = 20L,
                                 store_LRTb = FALSE) {
  
  rough_mode <- match.arg(rough_mode)
  
  stage1 <- fmm_bc_bootstrap_lrt(
    expr = expr, w = w, grid = grid, cache = cache, lambda_star = lambda_star,
    omega0 = omega0, p = p,
    rough_mode = rough_mode, ndense = ndense,
    refine = refine, refine_steps = refine_steps,
    alpha_step = alpha_step, omega_step_mult = omega_step_mult,
    omega_min = omega_min, omega_max = omega_max,
    ridge_refine = ridge_refine,
    eps = eps,
    Smax = as.integer(S1), Nmax = as.integer(N1),
    store_LRTb = store_LRTb
  )
  
  # If stage 1 hit S1, p is not small enough; stop.
  if (stage1$S >= as.integer(S1)) {
    stage1$stage <- 1L
    stage1$stage1 <- stage1
    return(stage1)
  }
  
  # Promote only if exceedances are very small (strong candidate).
  if (stage1$S > as.integer(promote_if_S_le)) {
    stage1$stage <- 1L
    stage1$stage1 <- stage1
    return(stage1)
  }
  
  stage2 <- fmm_bc_bootstrap_lrt(
    expr = expr, w = w, grid = grid, cache = cache, lambda_star = lambda_star,
    omega0 = omega0, p = p,
    rough_mode = rough_mode, ndense = ndense,
    refine = refine, refine_steps = refine_steps,
    alpha_step = alpha_step, omega_step_mult = omega_step_mult,
    omega_min = omega_min, omega_max = omega_max,
    ridge_refine = ridge_refine,
    eps = eps,
    Smax = as.integer(S2), Nmax = as.integer(N2),
    store_LRTb = store_LRTb
  )
  
  stage2$stage <- 2L
  stage2$stage1 <- stage1
  stage2
}

#' Two-stage BC bootstrap for multiple genes:
#' always return LRT_obs; store LRT_b only when pvalue < threshold
#'
#' This function runs the two-stage sequential BC bootstrap per gene.
#' It always returns the observed LRT statistic (LRT_obs) for every gene.
#' If the final bootstrap p-value is below \code{save_LRTb_if_p_lt}, it reruns
#' the same procedure for that gene with \code{store_LRTb=TRUE} to collect
#' bootstrap replicate statistics \code{LRT_b}.
#'
#' @param gene_names Character vector of gene names (must index rows of \code{data}).
#' @param data Matrix/data.frame genes x samples (rows=genes).
#' @param prepare_dataset Output of \code{fmm_prepare_lrt()}. If \code{rough_mode="dense"},
#'   \code{prepare_dataset$grid} must be enriched with \code{fmm_grid_add_dense_roughness()}.
#' @param rough_mode "dense" or "observed".
#' @param ndense Integer; must match cached \code{grid$ndense} if \code{rough_mode="dense"}.
#' @param refine Logical for the TEST (recommended FALSE for speed and consistency).
#'
#' @param N1,S1,promote_if_S_le Stage-1 controls.
#' @param N2,S2 Stage-2 controls.
#'
#' @param save_LRTb_if_p_lt Numeric threshold in (0,1). If final p-value < threshold,
#'   the function stores the bootstrap replicate LRTs \code{LRT_b} for that gene.
#'   Set NULL to never store \code{LRT_b}.
#'
#' @param parallel Logical; if TRUE uses future.apply (multisession).
#' @param workers Integer number of workers. If NULL and parallel=TRUE, uses
#'   \code{future::availableCores() - 1} (leaves 1 core free).
#'
#' @return A list with:
#' \itemize{
#'   \item \code{results}: data.frame with gene, pvalue, S, N, stage, LRT_obs
#'   \item \code{LRT_b}: named list with LRT_b vectors for genes with pvalue < threshold
#' }
fmm_test_genes_two_stage <- function(gene_names,
                                     data,
                                     prepare_dataset,
                                     rough_mode = c("dense","observed"),
                                     ndense = 512L,
                                     refine = FALSE,
                                     N1 = 200L, S1 = 10L, promote_if_S_le = 5L,
                                     N2 = 50000L, S2 = 20L,
                                     save_LRTb_if_p_lt = NULL,
                                     parallel = TRUE,
                                     workers = NULL,
                                     verbose = TRUE,
                                     progress_every = 100L,
                                     show_gene = TRUE) {
  
  rough_mode <- match.arg(rough_mode)
  ndense <- as.integer(ndense)
  progress_every <- as.integer(progress_every)
  if (progress_every < 1L) progress_every <- 1L
  
  gene_names <- as.character(gene_names)
  n_genes <- length(gene_names)
  
  grid <- prepare_dataset$grid
  if (rough_mode == "dense") {
    if (is.null(grid$rough_dense) || is.null(grid$ndense)) {
      stop("Grid has no cached dense roughness. Run fmm_grid_add_dense_roughness() on prepare_dataset$grid.")
    }
    if (as.integer(grid$ndense) != ndense) {
      stop(sprintf("ndense mismatch: cached grid$ndense=%d but ndense=%d was requested.",
                   as.integer(grid$ndense), ndense))
    }
  }
  
  if (!is.null(save_LRTb_if_p_lt)) {
    save_LRTb_if_p_lt <- as.numeric(save_LRTb_if_p_lt)
    if (!is.finite(save_LRTb_if_p_lt) || save_LRTb_if_p_lt <= 0 || save_LRTb_if_p_lt >= 1) {
      stop("save_LRTb_if_p_lt must be in (0,1), or NULL to disable.")
    }
  }
  
  run_one <- function(gn) {
    expr <- as.numeric(data[gn, ])
    
    out <- fmm_bc_two_stage_lrt(
      expr = expr,
      w = prepare_dataset$weights,
      grid = prepare_dataset$grid,
      cache = prepare_dataset$cache,
      lambda_star = prepare_dataset$lambda_star,
      rough_mode = rough_mode,
      ndense = ndense,
      refine = refine,
      N1 = as.integer(N1), S1 = as.integer(S1),
      promote_if_S_le = as.integer(promote_if_S_le),
      N2 = as.integer(N2), S2 = as.integer(S2),
      store_LRTb = FALSE
    )
    
    row <- data.frame(
      gene = gn,
      pvalue = out$pvalue,
      S = out$S,
      N = out$N,
      stage = out$stage,
      LRT_obs = out$LRT_obs,
      stringsAsFactors = FALSE
    )
    
    LRT_b <- NULL
    if (!is.null(save_LRTb_if_p_lt) && (out$pvalue < save_LRTb_if_p_lt)) {
      out2 <- fmm_bc_two_stage_lrt(
        expr = expr,
        w = prepare_dataset$weights,
        grid = prepare_dataset$grid,
        cache = prepare_dataset$cache,
        lambda_star = prepare_dataset$lambda_star,
        rough_mode = rough_mode,
        ndense = ndense,
        refine = refine,
        N1 = as.integer(N1), S1 = as.integer(S1),
        promote_if_S_le = as.integer(promote_if_S_le),
        N2 = as.integer(N2), S2 = as.integer(S2),
        store_LRTb = TRUE
      )
      LRT_b <- out2$LRT_b
    }
    
    list(row = row, LRT_b = LRT_b)
  }
  
  if (verbose) {
    cat(sprintf("[fmm_test_genes_two_stage] genes=%d | parallel=%s | rough_mode=%s | ndense=%d | refine=%s\n",
                n_genes, parallel, rough_mode, ndense, refine))
    cat(sprintf("[fmm_test_genes_two_stage] stage1: N1=%d S1=%d promote_if_S_le=%d | stage2: N2=%d S2=%d\n",
                as.integer(N1), as.integer(S1), as.integer(promote_if_S_le),
                as.integer(N2), as.integer(S2)))
    if (!is.null(save_LRTb_if_p_lt)) {
      cat(sprintf("[fmm_test_genes_two_stage] will store LRT_b when p < %.4g\n", save_LRTb_if_p_lt))
    }
  }
  
  if (!parallel) {
    # Serial: puedes ver gene a gene sin problema
    res_list <- vector("list", n_genes)
    for (i in seq_along(gene_names)) {
      if (verbose && (i %% progress_every == 0L || i == 1L || i == n_genes)) {
        if (show_gene) {
          cat(sprintf("  progress: %d/%d (%.1f%%) | gene=%s\n",
                      i, n_genes, 100*i/n_genes, gene_names[i]))
        } else {
          cat(sprintf("  progress: %d/%d (%.1f%%)\n",
                      i, n_genes, 100*i/n_genes))
        }
      }
      res_list[[i]] <- run_one(gene_names[i])
    }
    
  } else {
    # Parallel: barra de progreso limpia con progressr
    if (!requireNamespace("future.apply", quietly = TRUE) ||
        !requireNamespace("future", quietly = TRUE)) {
      stop("Install future + future.apply to use parallel=TRUE.")
    }
    if (!requireNamespace("progressr", quietly = TRUE)) {
      stop("Install progressr to show progress in parallel: install.packages('progressr')")
    }
    if (is.null(workers)) {
      workers <- max(1L, future::availableCores() - 1L)
    }
    if (verbose) cat(sprintf("[fmm_test_genes_two_stage] future::multisession workers=%d\n", workers))
    
    future::plan(future::multisession, workers = workers)
    
    # Handler: elige uno. "txtprogressbar" funciona en consola.
    progressr::handlers(global = TRUE)
    progressr::handlers("txtprogressbar")
    
    p <- progressr::progressor(steps = n_genes)
    
    res_list <- progressr::with_progress({
      future.apply::future_lapply(gene_names, function(gn) {
        out <- run_one(gn)
        p()  # tick
        out
      })
    })
  }
  
  results <- do.call(rbind, lapply(res_list, `[[`, "row"))
  
  LRTb_list <- setNames(lapply(res_list, `[[`, "LRT_b"), gene_names)
  LRTb_list <- LRTb_list[!vapply(LRTb_list, is.null, logical(1))]
  
  if (verbose) {
    cat(sprintf("[fmm_test_genes_two_stage] done. results=%d rows | stored LRT_b for %d genes\n",
                nrow(results), length(LRTb_list)))
  }
  
  list(results = results, LRT_b = LRTb_list)
}



#' Wild bootstrap with Rademacher sign-flip (design-weight recentering)
#'
#' @param expr numeric vector (observations), length n
#' @param mu_hat numeric vector (fitted mean under H0), length n
#' @param w numeric vector of weights, length n (positive; typically sum(w)=n)
#' @param center_w logical; if TRUE, enforce weighted-mean-zero bootstrap residuals
#' @param seed optional integer seed
#' @return numeric vector x_star, length n
bootstrap_wild_residuals <- function(expr, mu_hat, w,
                                     center_w = TRUE,
                                     seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  expr   <- as.numeric(expr)
  mu_hat <- as.numeric(mu_hat)
  w      <- as.numeric(w)
  
  n <- length(expr)
  if (length(mu_hat) != n) stop("mu_hat must have same length as expr.")
  if (length(w) != n) stop("w must have same length as expr.")
  if (any(!is.finite(expr)) || any(!is.finite(mu_hat)) || any(!is.finite(w))) stop("Non-finite values.")
  if (any(w <= 0)) stop("Weights must be positive.")
  sw <- sum(w)
  if (!is.finite(sw) || sw <= 0) stop("Sum of weights must be > 0.")
  
  # residuals under H0 (or given mu_hat)
  e <- expr - mu_hat
  
  # Rademacher multipliers
  s <- sample.int(2L, n, replace = TRUE)
  s <- ifelse(s == 1L, -1, +1)
  
  # wild bootstrap residuals
  e_star <- s * e
  
  # enforce weighted-mean-zero residuals (optional)
  if (center_w) {
    e_star <- e_star - (sum(w * e_star) / sw)
  }
  
  mu_hat + e_star
}

#' Wild bootstrap with whitened residuals (Rademacher sign-flip)
#'
#' @param expr numeric vector (observations), length n
#' @param mu_hat numeric vector (fitted mean under the null or alternative), length n
#' @param w numeric vector of weights, length n (nonnegative). Typically normalized to sum(w)=n.
#' @param center_w logical; if TRUE, enforce weighted-mean-zero residuals in the bootstrap sample
#' @param seed optional integer seed
#' @return numeric vector x_star, length n
bootstrap_whitened_residuals <- function(expr, mu_hat, w,
                                         center_w = TRUE,
                                         seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  expr  <- as.numeric(expr)
  mu_hat <- as.numeric(mu_hat)
  w     <- as.numeric(w)
  
  n <- length(expr)
  if (length(mu_hat) != n) stop("mu_hat must have same length as expr.")
  if (length(w) != n) stop("w must have same length as expr.")
  if (any(!is.finite(expr)) || any(!is.finite(mu_hat)) || any(!is.finite(w))) stop("Non-finite values.")
  if (any(w < 0)) stop("Weights must be nonnegative.")
  if (sum(w) <= 0) stop("Sum of weights must be > 0.")
  
  # residuals
  e <- expr - mu_hat
  
  # handle zero weights safely
  sw <- sqrt(w)
  sw_safe <- ifelse(sw > 0, sw, 1)   # avoid division by 0
  # whiten
  r <- sw * e
  
  # Rademacher multipliers
  s <- sample.int(2L, n, replace = TRUE)
  s <- ifelse(s == 1L, -1, +1)
  
  # bootstrap whitened residuals + unwhiten
  r_star <- s * r
  e_star <- r_star / sw_safe
  e_star[sw == 0] <- 0  # if w_i=0, that point carries no weight; keep unchanged
  
  # optional: enforce weighted-mean-zero residuals 
  if (center_w) {
    e_star <- e_star - (sum(w * e_star) / sum(w))
  }
  
  mu_hat + e_star
}


#' Compute observed weighted LRT for one gene (no bootstrap)
#'
#' @param expr numeric vector length n (gene expression)
#' @param prepare_dataset list from fmm_prepare_lrt(); must contain weights, grid, cache, lambda_star
#' @param rough_mode "dense" or "observed"
#' @param ndense integer (only used if rough_mode="dense")
#' @param omega0,p penalty parameters
#' @param refine logical (recommend FALSE for speed/consistency in testing)
#' @param refine_steps,alpha_step,omega_step_mult refinement controls (ignored if refine=FALSE)
#' @param ridge_refine numeric ridge used in local WLS during refinement
#' @param eps small stabilizer for log ratio
#'
#' @return numeric scalar LRT_obs
fmm_LRT_obs_only <- function(expr,
                             prepare_dataset,
                             rough_mode = c("dense","observed"),
                             ndense = 512L,
                             omega0 = 0.10,
                             p = 3,
                             refine = FALSE,
                             refine_steps = 3L,
                             alpha_step = (2*pi)/48,
                             omega_step_mult = 1.25,
                             ridge_refine = 1e-10,
                             eps = 1e-12,
                             penalized = FALSE) {
  rough_mode <- match.arg(rough_mode)
  
  times <- as.numeric(prepare_dataset$settings$times) %% (2*pi)
  w_use <- as.numeric(prepare_dataset$weights)
  w_use <- normalize_weights_to_n(w_use)
  
  expr <- as.numeric(expr)
  n <- length(times)
  if (length(expr) != n) stop("expr length mismatch with times.")
  if (any(!is.finite(expr))) stop("Non-finite expr.")
  
  grid <- prepare_dataset$grid
  cache <- prepare_dataset$cache
  lambda_star <- as.numeric(prepare_dataset$lambda_star)
  
  fitH1 <- fmm_fit_power_rough(
    x = expr, grid = grid, cache = cache,
    center = TRUE, normalize = TRUE,
    lambda = lambda_star, omega0 = omega0, p = p,
    refine = refine,
    refine_steps = as.integer(refine_steps),
    alpha_step = alpha_step,
    omega_step_mult = omega_step_mult,
    omega_min = min(prepare_dataset$settings$omega_grid, na.rm = TRUE),
    omega_max = max(prepare_dataset$settings$omega_grid, na.rm = TRUE),
    ridge_refine = ridge_refine,
    keep_trace = FALSE,
    rough_mode = rough_mode,
    ndense = as.integer(ndense),
    return_grid = FALSE
  )
  
  wrss1 <- as.numeric(fitH1$rss)
  pen1 <- as.numeric(fitH1$penalty_sel)
  mu0   <- sum(w_use * expr) / sum(w_use)
  wrss0 <- sum(w_use * (expr - mu0)^2)
  
  if(!penalized){
    #max(0, n * log((wrss0 + eps) / (wrss1 + eps)))
    max(0, n * log((wrss0) / (wrss1)))
  } else {  
    #max(0, n * log((wrss0 + eps) / (wrss1 + pen1 + eps)))
    max(0, n * log((wrss0) / (wrss1 + pen1)))
  }  
}

#' Build a global null empirical distribution for the FMM-LRT (fixed dataset)
#'
#' Simulates LRT under H0 using parametric Gaussian noise (scaled out by the WRSS ratio),
#' keeping design (times, weights, grid, lambda, rough_mode) fixed.
#'
#' @param prepare_dataset list from fmm_prepare_lrt()
#' @param B integer number of null simulations (e.g., 50000 or 200000)
#' @param rough_mode "dense" or "observed"
#' @param ndense integer (only used if rough_mode="dense")
#' @param omega0,p penalty parameters
#' @param refine logical; recommend FALSE
#' @param sigma numeric noise sd (scale cancels approximately; keep 1)
#' @param center_null logical; enforce weighted mean 0 for each null draw
#' @param seed integer seed for reproducibility
#' @param eps stabilizer for LRT computation
#'
#' @return list with sorted LRT_null, B, and a pval() function
fmm_build_null_cdf <- function(prepare_dataset,
                               B = 1000000L,
                               rough_mode = c("dense","observed"),
                               ndense = 512L,
                               omega0 = 0.10,
                               p = 3,
                               refine = FALSE,
                               sigma = 1,
                               normalice_null = FALSE,
                               seed = 1L,
                               eps = 1e-12) {
  rough_mode <- match.arg(rough_mode)
  B <- as.integer(B)
  if (B < 1000) warning("B is small; tail resolution will be limited.")
  
  times <- as.numeric(prepare_dataset$settings$times) %% (2*pi)
  n <- length(times)
  
  w <- normalize_weights_to_n(as.numeric(prepare_dataset$weights))
  sw <- sum(w)
  if (sw <= 0) stop("Sum of weights must be > 0.")
  
  if (!is.null(seed)) set.seed(seed)
  
  # Draw all Z at once (fast) and optionally weighted-center each row
  Z <- matrix(rnorm(B * n), nrow = B, ncol = n)
  X <- sigma * Z
  
  if (normalice_null) {
    X <- t(apply(X,1,normalice))
  }
  
  LRT_null <- numeric(B)
  for (b in seq_len(B)) {
    LRT_null[b] <- fmm_LRT_obs_only(
      expr = X[b, ],
      prepare_dataset = prepare_dataset,
      rough_mode = rough_mode,
      ndense = ndense,
      omega0 = omega0, p = p,
      refine = refine,
      eps = eps
    )
  }
  
  LRT_null <- sort(LRT_null)
  
  # ---- ECDF / tail / p-value (right tail) ----
  # CDF: P(LRT <= x)
  F_fun <- function(x) {
    x <- as.numeric(x)
    le <- findInterval(x, LRT_null, left.open = FALSE)  # # <= x
    le / B
  }
  
  # Right tail: P(LRT >= x) (empirical, no +1 correction)
  tail_fun <- function(x) {
    x <- as.numeric(x)
    lt <- findInterval(x, LRT_null, left.open = TRUE)   # # < x
    (B - lt) / B
  }
  
  # Monte Carlo p-value with +1 correction: (1 + #null >= x)/(B+1)
  pval_fun <- function(LRT_obs) {
    x <- as.numeric(LRT_obs)
    lt <- findInterval(x, LRT_null, left.open = TRUE)   # # < x
    ge <- B - lt                                        # # >= x
    (ge + 1) / (B + 1)
  }
  
  list(
    LRT_null = LRT_null,
    B = B,
    settings = list(
      rough_mode = rough_mode,
      ndense = as.integer(ndense),
      omega0 = omega0, p = p,
      refine = refine,
      sigma = sigma,
      normalice_null = normalice_null,
      seed = seed,
      eps = eps
    ),
    F = F_fun,
    tail = tail_fun,
    pval = pval_fun
  )
}

#' Fast p-values for many genes using a precomputed global null CDF
#'
#' @param gene_names character vector
#' @param data matrix genes x samples (rows=genes)
#' @param prepare_dataset list from fmm_prepare_lrt()
#' @param null_calib output from fmm_build_null_cdf()
#'
#' @return data.frame with gene, LRT_obs, pvalue
fmm_pvals_from_null_cdf <- function(gene_names, data, prepare_dataset, null_calib) {
  gene_names <- as.character(gene_names)
  
  LRT_obs <- numeric(length(gene_names))
  for (k in seq_along(gene_names)) {
    g <- gene_names[k]
    LRT_obs[k] <- fmm_LRT_obs_only(
      expr = as.numeric(data[g, ]),
      prepare_dataset = prepare_dataset,
      rough_mode = null_calib$settings$rough_mode,
      ndense = null_calib$settings$ndense,
      omega0 = null_calib$settings$omega0,
      p = null_calib$settings$p,
      refine = null_calib$settings$refine,
      eps = null_calib$settings$eps
    )
  }
  
  pval <- null_calib$pval(LRT_obs)
  
  data.frame(
    gene = gene_names,
    LRT_obs = LRT_obs,
    pvalue = pval,
    stringsAsFactors = FALSE
  )
}

