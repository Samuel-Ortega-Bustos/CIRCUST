# ============================================================
# Weighted mean (weights normalized or not; works either way)
wmean <- function(x, w) sum(w * x) / sum(w)

# ============================================================
#' Normalize weights to sum(w)=n
#'
#' @param w numeric vector (nonnegative)
#' @return numeric vector with sum(w)=length(w)
normalize_weights_to_n <- function(w) {
  w <- as.numeric(w)
  n <- length(w)
  if (any(!is.finite(w))) stop("Non-finite weights detected.")
  if (any(w < 0)) stop("Weights must be nonnegative.")
  sw <- sum(w)
  if (sw <= 0) stop("Sum of weights must be > 0.")
  n * w / sw
}

# ============================================================
#' Circular angular difference in (-pi, pi]
#'
#' @param theta numeric vector
#' @return numeric vector in (-pi, pi]
angdiff <- function(theta) atan2(sin(theta), cos(theta))

# ============================================================
#' Normalize a numeric vector to [-1,1]
#'
#' Affine scaling: x -> 2*(x-min)/(max-min) - 1.
#' If constant, returns zeros.
#'
#' @param x numeric vector
#' @param eps small positive tolerance for constant detection
#' @return numeric vector in [-1,1]
normalice <- function(x, eps = 1e-12) {
  x <- as.numeric(x)
  xmin <- min(x, na.rm = TRUE)
  xmax <- max(x, na.rm = TRUE)
  r <- xmax - xmin
  if (!is.finite(r) || r < eps) return(rep(0, length(x)))
  2 * (x - xmin) / r - 1
}


# ============================================================
#' Simulate a flat (non-oscillatory) gene under H0
#'
#' Generates x_i = mu + eps_i with eps_i ~ N(0,sigma^2).
#' Optionally weighted-centers the output (useful for projection-based steps).
#'
#' @param n integer number of time points
#' @param w numeric weights length n (used only for centering)
#' @param sigma positive scalar standard deviation
#' @param mu scalar mean level
#' @param center_w logical; if TRUE, subtract weighted mean
#' @return numeric vector length n
simulate_H0_whitened <- function(n, w, sigma = 1, mu = 0, center_w = TRUE) {
  n <- as.integer(n)
  if (n < 1) stop("n must be >= 1.")
  if (!(is.finite(sigma) && sigma > 0)) stop("sigma must be > 0.")
  w <- normalize_weights_to_n(as.numeric(w))
  if (length(w) != n) stop("w must have length n.")
  
  x <- mu + rnorm(n, mean = 0, sd = sigma)
  
  if (center_w) {
    muw <- sum(w * x) / sum(w)
    x <- x - muw
  }
  x
}

# ============================================================
#' Simulate a matrix of flat genes and normalize each to [-1,1]
#'
#' @param G integer number of genes
#' @param times numeric vector of times (used only for column count/labels)
#' @param w numeric weights length ncol (used for optional weighted-centering)
#' @param sigma positive scalar
#' @param mu_sd baseline SD for random mu per gene (kept small)
#' @param center_w logical; if TRUE, weighted-center each gene
#' @param normalize_rows logical; if TRUE, scale each gene to [-1,1]
#' @param seed optional integer seed
#' @return numeric matrix G x length(times)
simulate_flat_genes <- function(G, times, w, sigma = 1,
                                mu_sd = 0.2,
                                center_w = TRUE,
                                normalize_rows = TRUE,
                                seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  G <- as.integer(G)
  if (G < 1) stop("G must be >= 1.")
  times <- as.numeric(times)
  n <- length(times)
  
  w <- normalize_weights_to_n(as.numeric(w))
  if (length(w) != n) stop("w must have same length as times.")
  if (!(is.finite(sigma) && sigma > 0)) stop("sigma must be > 0.")
  
  X <- matrix(NA_real_, nrow = G, ncol = n)
  for (g in seq_len(G)) {
    mu <- rnorm(1, mean = 0, sd = mu_sd)
    x <- mu + rnorm(n, mean = 0, sd = sigma)
    
    if (center_w) {
      muw <- sum(w * x) / sum(w)
      x <- x - muw
    }
    if (normalize_rows) x <- normalize_to_unit_range(x)
    X[g, ] <- x
  }
  
  rownames(X) <- paste0("flat_", seq_len(G))
  colnames(X) <- paste0("t", seq_len(n))
  X
}



# ============================================================
#' Predict an FMM1 fit on a dense time grid.
#'
#' This uses the identity cos(beta + phi) = cos(beta)cos(phi) - sin(beta)sin(phi)
#' and a numerically stable routine that returns cos(phi) and sin(phi) directly.
#'
#' @param t_dense Numeric vector of times where predictions are required.
#'               Recommended scale: radians. Values can be outside [0, 2*pi).
#' @param fit     List returned by fitFMM_power_rough() (must contain fit$parameters).
#' @param mobius_fun Function with signature mobius_fun(t, alpha, omega, ...) returning list(C=..., S=...).
#' @param wrap_times Logical; if TRUE, wraps times to [0, 2*pi).
#' @return Numeric vector of fitted mean values mu(t_dense).
predict_fmm1_dense <- function(t_dense, fit, mobius_fun = mobius_cos_sin, wrap_times = TRUE) {
  if (is.null(fit$parameters)) stop("fit must contain a $parameters element.")
  
  par <- fit$parameters
  need <- c("M","A","beta","alpha","omega")
  miss <- setdiff(need, names(par))
  if (length(miss) > 0) stop("Missing parameters in fit$parameters: ", paste(miss, collapse = ", "))
  
  M <- par$M; A <- par$A; beta <- par$beta; alpha <- par$alpha; omega <- par$omega
  
  t_use <- as.numeric(t_dense)
  if (wrap_times) t_use <- t_use %% (2*pi)
  
  # Stable computation of cos(phi) and sin(phi)
  cs <- mobius_fun(t = t_use, alpha = alpha, omega = omega)
  
  # cos(beta + phi) from cos/sin components
  osc <- cos(beta) * cs$C - sin(beta) * cs$S
  
  M + A * osc
}


#' Plot one gene: observed points with transfer-level aesthetics + dense FMM1 fit
#'
#' Transfer level is typically the *raw* circular density (e.g. w_raw from weights_vonmises),
#' while WLS uses the *normalized* weights w (sum(w)=n).
#'
#' @param t               Numeric times (radians).
#' @param x               Numeric observations.
#' @param w_fit           WLS weights (used for fitting diagnostics only), can be NULL.
#' @param transfer        Numeric transfer levels for plotting (e.g. w_raw). Same length as t.
#' @param fit             Output of fitFMM_power_rough().
#' @param n_dense         Number of dense points for curve.
#' @param wrap            Wrap t to [0,2*pi) for plotting.
#' @param cex_min,cex_max Point size range.
#' @param alpha_min,alpha_max Transparency range (0..1). Used only if use_alpha=TRUE.
#' @param use_alpha        If TRUE, also map transfer->alpha (helps show low-transfer points).
#' @param main            Title.
#' @param show_alpha_pi    If TRUE, vertical line at alpha+pi.
#' @return Invisibly dense grid + predictions.
plot_fmm1_gene <- function(t, x, fit,
                           transfer = NULL,          # optional: e.g. w.raw
                           n_dense = 400,
                           point_cex = 1.0,          # fixed point size
                           use_alpha = TRUE,         # map transfer -> alpha
                           alpha_range = c(0.2, 1.0),
                           use_greyscale = FALSE,    # if TRUE: map transfer -> grey
                           main = "FMM1 fit",
                           show_alpha_pi = TRUE,
                           wrap = TRUE) {
  
  stopifnot(length(t) == length(x))
  tt <- as.numeric(t)
  if (wrap) tt <- tt %% (2*pi)
  ord <- order(tt)
  
  tt_o <- tt[ord]
  x_o  <- x[ord]
  
  # Dense prediction
  t_dense <- seq(0, 2*pi, length.out = n_dense)
  y_dense <- predict_fmm1_dense(t_dense, fit, wrap_times = FALSE)
  
  # Default point style
  col_pts <- "black"
  
  # Optional transfer mapping (NOT size)
  if (!is.null(transfer)) {
    tr <- as.numeric(transfer)[ord]
    tr[!is.finite(tr)] <- NA_real_
    
    # Robust rescale to [0,1]
    lo <- suppressWarnings(quantile(tr, 0.02, na.rm = TRUE, names = FALSE))
    hi <- suppressWarnings(quantile(tr, 0.98, na.rm = TRUE, names = FALSE))
    
    if (!is.finite(lo) || !is.finite(hi) || hi <= lo + 1e-15) {
      tr01 <- rep(0.5, length(tr))
    } else {
      tr_clipped <- pmin(pmax(tr, lo), hi)
      tr01 <- (tr_clipped - lo) / (hi - lo)
    }
    
    if (use_greyscale) {
      # 0 -> light grey, 1 -> black
      grey_val <- 0.85 - 0.85 * tr01
      col_pts <- gray(grey_val)
    } else if (use_alpha) {
      a0 <- alpha_range[1]; a1 <- alpha_range[2]
      a  <- a0 + tr01 * (a1 - a0)
      col_pts <- rgb(0, 0, 0, alpha = a)
    }
  }
  
  plot(tt_o, x_o, pch = 16, cex = point_cex, col = col_pts,
       xlab = "t (radians)", ylab = "x", main = main)
  lines(t_dense, y_dense, lwd = 2)
  
  if (show_alpha_pi && !is.null(fit$parameters$alpha)) {
    a_pi <- (fit$parameters$alpha + pi) %% (2*pi)
    abline(v = a_pi, lty = 2)
  }
  
  invisible(NULL)
}

# ============================================================
# Running median smoother
runmed_safe <- function(y, k = 3L) {
  k <- as.integer(k)
  if (k <= 1) return(y)
  if (k %% 2 == 0) k <- k + 1L
  stats::runmed(y, k = k, endrule = "median")
}
