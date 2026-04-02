# =============================================================================
# rscripts/validate_export.R
# =============================================================================
# Runs CIRCUST Stages 1-2 on matrixIn, exports intermediate results to CSV
# for comparison against the Python implementation.
#
# Run from the repo ROOT:
#   Rscript rscripts/validate_export.R
#
# Outputs written to:  validation/reference/
# =============================================================================

REPO_ROOT  <- normalizePath(".")
RSCRIPTS   <- file.path(REPO_ROOT, "rscripts")
RDATA_PATH <- file.path(REPO_ROOT, "data", "matrixIn.RData")
OUT_DIR    <- file.path(REPO_ROOT, "validation", "reference")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Load source files, skipping Windows .dll / circular-package dependencies
# ---------------------------------------------------------------------------
old_wd <- getwd()
setwd(RSCRIPTS)

cat("Loading R source files ...\n")
source("FMM.R")

lines <- readLines("functionGTEX_cores.R")
skip_patterns <- c(
  "install.packages",
  'source("upDownUp_NP_Code_NoParalelizado.R")',
  'source("upDownUp_NP_Code_Paralelizado.R")',
  'source("upDownUp_NP_Code_modif.R")',
  'source("NucleoComun.R")',
  'require("circular")',
  "require('circular')"
)

# Stubs for circular-package functions used only in plotting
as.circular   <- function(x, ...) x
plot.circular <- function(x, ...) invisible(NULL)
axis.circular <- function(...) invisible(NULL)

keep <- rep(TRUE, length(lines))
for (pat in skip_patterns) {
  keep <- keep & !grepl(pat, lines, fixed = TRUE)
}
eval(parse(text = paste(lines[keep], collapse = "\n")), envir = .GlobalEnv)
setwd(old_wd)

cat("  Done.\n")

# ---------------------------------------------------------------------------
# Larriba et al. (2023) core clock genes — matches Python constants.py
coreG <- c("PER1", "PER2", "PER3", "CRY1", "CRY2", "ARNTL", "CLOCK",
           "NR1D1", "RORA", "DBP", "TEF", "STAT3")
timing <- list()

# ===========================================================================
# STAGE 1: PREPROCESSING
# ===========================================================================
cat("\nLoading matrixIn.RData ...\n")
t0 <- proc.time()
load(RDATA_PATH)
timing$load <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Loaded: %d genes x %d samples\n", nrow(matrixIn), ncol(matrixIn)))

# Step 1 — Drop unnamed genes
cat("Step 1: dropping unnamed genes ...\n")
t0 <- proc.time()
mFull0 <- matrixIn
indNoNames <- which(is.na(rownames(mFull0)), arr.ind = TRUE)
if (length(indNoNames) > 0) mFull0 <- mFull0[-indNoNames, ]
timing$drop_unnamed <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Removed %d unnamed genes\n", length(indNoNames)))

# Step 2 — Drop sparse genes >30% zeros or NaN
cat("Step 2: dropping sparse genes ...\n")
t0 <- proc.time()
drop <- c()
for (i in 1:nrow(mFull0)) {
  if (length(which(mFull0[i,] == 0, arr.ind = TRUE)) > 0.30 * ncol(mFull0) |
      sum(is.na(mFull0[i,])) > 0.30 * ncol(mFull0)) {
    drop <- c(drop, i)
  }
}
if (length(drop) > 0) {
  dropped_sparse_names <- rownames(mFull0)[drop]
  mFull1 <- mFull0[-drop, ]
  rownames(mFull1) <- rownames(mFull0)[-drop]
} else {
  dropped_sparse_names <- character(0)
  mFull1 <- mFull0
  rownames(mFull1) <- rownames(mFull0)
}
timing$drop_sparse <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Removed %d sparse genes\n", length(drop)))
write.csv(data.frame(gene = dropped_sparse_names),
          file.path(OUT_DIR, "r_dropped_sparse.csv"), row.names = FALSE)

# Step 3 — Resolve duplicates, keep highest MAD
cat("Step 3: resolving duplicates ...\n")
t0 <- proc.time()
namesRep <- names(which(sort(table(rownames(mFull1))) > 1))
dropped_dupes_names <- character(0)
if (length(namesRep) > 0) {
  filasOut <- c()
  for (i in 1:length(namesRep)) {
    filas  <- which(namesRep[i] == rownames(mFull1), arr.ind = TRUE)
    mads   <- apply(mFull1[filas, ], 1, mad, constant = 1)
    losers <- filas[-which.max(mads)]
    dropped_dupes_names <- c(dropped_dupes_names, rownames(mFull1)[losers])
    filasOut <- c(filasOut, losers)
  }
  mFull1 <- mFull1[-filasOut, ]
}
timing$resolve_dupes <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Resolved %d duplicate gene names\n", length(namesRep)))
write.csv(data.frame(gene = dropped_dupes_names),
          file.path(OUT_DIR, "r_dropped_dupes.csv"), row.names = FALSE)

# Step 4 — Normalise each gene to [-1, 1]
cat("Step 4: normalising ...\n")
t0 <- proc.time()
mFullNorm <- t(apply(mFull1, 1, normalice))
rownames(mFullNorm) <- rownames(mFull1)
timing$normalise <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Normalised: %d genes x %d samples\n", nrow(mFullNorm), ncol(mFullNorm)))

# Export preprocessing outputs
write.csv(mFull1,    file.path(OUT_DIR, "r_expr_raw.csv"))
write.csv(mFullNorm, file.path(OUT_DIR, "r_expr_norm.csv"))

# Core gene extraction
mCoreNorm <- mFullNorm[match(coreG, rownames(mFullNorm)), ]
write.csv(mCoreNorm, file.path(OUT_DIR, "r_core_norm.csv"))

# ===========================================================================
# CPCA — obtainCPCA13
# ===========================================================================
cat("CPCA: running circular PCA ...\n")
t0 <- proc.time()
cpca <- obtainCPCA13(mCoreNorm, "Example", coreG, 8, FALSE)
timing$cpca <- (proc.time() - t0)["elapsed"]

# cpca returns: list(orderCPCA8, escalaPhi8, mOutliers, outs,
#                    varPer8, eigen18, eigen28, eigen38)
write.csv(data.frame(sample_order = cpca[[1]]),
          file.path(OUT_DIR, "r_sample_order.csv"), row.names = FALSE)
write.csv(data.frame(circular_scale = cpca[[2]]),
          file.path(OUT_DIR, "r_circular_scale.csv"), row.names = FALSE)
write.csv(data.frame(pc1 = cpca[[6]], pc2 = cpca[[7]], pc3 = cpca[[8]]),
          file.path(OUT_DIR, "r_pc_loadings.csv"), row.names = FALSE)
write.csv(data.frame(variance_explained = cpca[[5]]),
          file.path(OUT_DIR, "r_variance.csv"), row.names = FALSE)
write.csv(data.frame(candidate_idx = cpca[[3]][1, 1:8]),
          file.path(OUT_DIR, "r_outlier_candidates.csv"), row.names = FALSE)

# ===========================================================================
# FMM + COSINOR fits on core genes (initial CPCA ordering)
# Replicates giveMatIniNP_v3_cores lines 3964-4012
# ===========================================================================
cat("FMM/Cosinor: fitting core genes on CPCA order ...\n")
t0 <- proc.time()

orderCPCA <- cpca[[1]]
escala    <- cpca[[2]]
nG        <- length(coreG)

# Matrices for core genes + 3 eigengenes
datCore         <- matrix(0, nG + 3, ncol(mCoreNorm))
FMMParCoreG     <- matrix(0, nG + 3, 5)
CosParCoreG     <- matrix(0, nG + 3, 3)
fitParCore      <- matrix(0, nG + 3, ncol(mCoreNorm))
fitCosCore      <- matrix(0, nG + 3, ncol(mCoreNorm))
resParStTissue  <- matrix(0, nG + 3, ncol(mCoreNorm))
resStTissue     <- matrix(0, nG + 3, ncol(mCoreNorm))
phisCoreG       <- c()
peaksCoreG      <- c()

for (i in 1:(nG + 3)) {
  # Data: core genes or eigengenes
  if (i <= nG) {
    datCore[i, ] <- mCoreNorm[i, orderCPCA]
  } else if (i == nG + 1) {
    datCore[i, ] <- cpca[[6]][orderCPCA]
  } else if (i == nG + 2) {
    datCore[i, ] <- cpca[[7]][orderCPCA]
  } else {
    datCore[i, ] <- cpca[[8]][orderCPCA]
  }

  # Fits
  funCos <- funcionCosinor(datCore[i, ], escala, length(escala))
  funPar <- fitFMM_Par(datCore[i, ], escala)

  fitCosCore[i, ] <- funCos[[1]]
  fitParCore[i, ] <- funPar[[1]]

  # Residuals
  resCos <- datCore[i, ] - funCos[[1]]
  resPar <- datCore[i, ] - funPar[[1]]
  resStTissue[i, ]    <- (resCos - mean(resCos)) / sd(resCos)
  resParStTissue[i, ] <- (resPar - mean(resPar)) / sd(resPar)

  # Params
  phisCoreG[i]    <- (-funCos[[5]]) %% (2 * pi)
  FMMParCoreG[i, ] <- c(funPar[[2]], funPar[[3]], funPar[[4]], funPar[[5]], funPar[[6]])
  CosParCoreG[i, ] <- c(funCos[[2]], funCos[[3]], funCos[[5]])
  peaksCoreG[i]   <- compUU(FMMParCoreG[i, 1], FMMParCoreG[i, 2], FMMParCoreG[i, 3])
}

timing$fmm_cosinor <- (proc.time() - t0)["elapsed"]

# Row names
gene_labels <- c(coreG, "Eigen1", "Eigen2", "Eigen3")
rownames(FMMParCoreG) <- gene_labels
colnames(FMMParCoreG) <- c("M", "A", "alpha", "beta", "omega")
rownames(CosParCoreG) <- gene_labels
colnames(CosParCoreG) <- c("M", "A", "phi")

write.csv(FMMParCoreG, file.path(OUT_DIR, "r_fmm_params_initial.csv"))
write.csv(CosParCoreG, file.path(OUT_DIR, "r_cos_params_initial.csv"))
write.csv(data.frame(gene = gene_labels, fmm_peak = peaksCoreG, cos_peak = phisCoreG),
          file.path(OUT_DIR, "r_peaks_initial.csv"), row.names = FALSE)

# Export standardised FMM residuals for core genes only (12 rows x n_samples)
resFmm_df <- as.data.frame(resParStTissue[1:nG, ])
rownames(resFmm_df) <- coreG
write.csv(resFmm_df, file.path(OUT_DIR, "r_std_residuals_fmm.csv"))

# ===========================================================================
# OUTLIER DETECTION (replicates lines 4049-4083)
# ===========================================================================
cat("Outlier detection ...\n")
t0 <- proc.time()

outsMult  <- c()
outsMultG <- c()
outsUni   <- c()
outsUniG  <- c()
nOutsTot  <- 0

for (i in 1:nG) {
  # Multivariate: |std_res| > 3 AND in CPCA candidate set
  if (sum(abs(resParStTissue[i, ]) > 3) > 0 &
      nOutsTot <= ceiling(0.05 * ncol(resParStTissue))) {
    cuantos <- which(abs(resParStTissue[i, ]) > 3, arr.ind = TRUE)
    for (j in 1:length(cuantos)) {
      if (!is.na(match(orderCPCA[cuantos[j]], cpca[[3]])) &
          match(orderCPCA[cuantos[j]], cpca[[3]]) <= cpca[[3]][1, ncol(cpca[[3]])]) {
        outsMult  <- c(outsMult, orderCPCA[cuantos[j]])
        outsMultG <- c(outsMultG, coreG[i])
        nOutsTot  <- nOutsTot + 1
      }
    }
  }
  # Univariate: |std_res| > 4
  if (sum(abs(resParStTissue[i, ]) > 4) > 0 &
      nOutsTot <= ceiling(0.05 * ncol(resParStTissue))) {
    outsUni  <- c(outsUni, orderCPCA[which(abs(resParStTissue[i, ]) > 4, arr.ind = TRUE)])
    outsUniG <- c(outsUniG,
                  rep(coreG[i], length(orderCPCA[which(abs(resParStTissue[i, ]) > 4, arr.ind = TRUE)])))
    nOutsTot <- nOutsTot + 1
  }
}

dropTissueOut <- union(outsUni, outsMult)
timing$outliers <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Univariate outliers: %d, Multivariate outliers: %d\n",
            length(outsUni), length(outsMult)))
cat(sprintf("  Total samples to drop: %d\n", length(dropTissueOut)))

write.csv(data.frame(sample_idx = dropTissueOut),
          file.path(OUT_DIR, "r_outlier_samples.csv"), row.names = FALSE)
write.csv(data.frame(sample_idx = outsUni, gene = outsUniG),
          file.path(OUT_DIR, "r_outlier_uni.csv"), row.names = FALSE)
write.csv(data.frame(sample_idx = outsMult, gene = outsMultG),
          file.path(OUT_DIR, "r_outlier_multi.csv"), row.names = FALSE)

# ===========================================================================
# RE-CPCA AFTER OUTLIER REMOVAL (lines 4084-4101)
# ===========================================================================
cat("Re-running CPCA after outlier removal ...\n")
t0 <- proc.time()

mCoreRaw <- mFull1[match(coreG, rownames(mFull1)), ]

if (length(dropTissueOut) > 0) {
  mCoreRaw2      <- mCoreRaw[, -dropTissueOut]
  mCoreNorm2     <- t(apply(mCoreRaw2, 1, normalice))
  rownames(mCoreNorm2) <- coreG
  cpca2          <- obtainCPCA13(mCoreNorm2, "Example", coreG, 8, FALSE)
} else {
  mCoreNorm2 <- mCoreNorm
  cpca2      <- cpca
}

initialOrd  <- cpca2[[1]]
initialEsc  <- cpca2[[2]]

# Build the final full ordered matrix (lines 4094-4101)
if (length(dropTissueOut) > 0) {
  mFullTissue     <- (mFull1[, -dropTissueOut])[, initialOrd]
} else {
  mFullTissue     <- mFull1[, initialOrd]
}
mFullTissueNorm <- t(apply(mFullTissue, 1, normalice))
rownames(mFullTissueNorm) <- rownames(mFullTissue)

timing$cpca2 <- (proc.time() - t0)["elapsed"]

write.csv(data.frame(sample_order = cpca2[[1]]),
          file.path(OUT_DIR, "r_sample_order_2.csv"), row.names = FALSE)
write.csv(data.frame(circular_scale = cpca2[[2]]),
          file.path(OUT_DIR, "r_circular_scale_2.csv"), row.names = FALSE)
write.csv(data.frame(variance_explained = cpca2[[5]]),
          file.path(OUT_DIR, "r_variance_2.csv"), row.names = FALSE)

# ===========================================================================
# FMM FITS AFTER OUTLIER REMOVAL (lines 4112-4135)
# ===========================================================================
cat("FMM/Cosinor: re-fitting core genes after outlier removal ...\n")
t0 <- proc.time()

allParAfter  <- list()
r2ParAfter   <- c()
phisFMMAfter <- c()
phisCosAfter <- c()
r2CosAfter   <- c()

for (i in 1:nG) {
  datAfter <- mFullTissueNorm[match(coreG[i], rownames(mFullTissueNorm)), ]
  # FMM
  allParAfter[[i]] <- fitFMM_Par(datAfter, initialEsc)
  phisFMMAfter     <- c(phisFMMAfter,
                        compUU(allParAfter[[i]][[2]], allParAfter[[i]][[3]],
                               allParAfter[[i]][[4]]))
  r2ParAfter[i]    <- 1 - mean((datAfter - allParAfter[[i]][[1]])^2) /
                            mean((datAfter - mean(datAfter))^2)
  # Cosinor
  cosFit           <- funcionCosinor(datAfter, initialEsc, length(initialEsc))
  phisCosAfter     <- c(phisCosAfter, (-cosFit[[5]]) %% (2 * pi))
  r2CosAfter[i]    <- 1 - mean((datAfter - cosFit[[1]])^2) /
                            mean((datAfter - mean(datAfter))^2)
}

timing$fmm_after <- (proc.time() - t0)["elapsed"]

# Build R² table (line 4138)
m3r2 <- cbind(r2ParAfter, r2CosAfter)
rownames(m3r2) <- coreG
colnames(m3r2) <- c("r2_fmm", "r2_cos")

# FMM params after
FMMAfterParams <- matrix(0, nG, 5)
for (i in 1:nG) {
  FMMAfterParams[i, ] <- c(allParAfter[[i]][[2]], allParAfter[[i]][[3]],
                            allParAfter[[i]][[4]], allParAfter[[i]][[5]],
                            allParAfter[[i]][[6]])
}
rownames(FMMAfterParams) <- coreG
colnames(FMMAfterParams) <- c("M", "A", "alpha", "beta", "omega")

write.csv(FMMAfterParams, file.path(OUT_DIR, "r_fmm_params_after.csv"))
write.csv(m3r2,           file.path(OUT_DIR, "r_r2_after.csv"))
write.csv(data.frame(gene = coreG, fmm_peak = phisFMMAfter, cos_peak = phisCosAfter),
          file.path(OUT_DIR, "r_peaks_after.csv"), row.names = FALSE)

# ===========================================================================
# STAGE 2.1: basicPreOder_cores (lines 4192-4263)
# ===========================================================================
cat("Stage 2.1: preliminary pre-order ...\n")
t0 <- proc.time()

# This is what basicPreOder_cores receives as listaStep1:
# [[2]] = initialEsc (from cpca2), [[3]] = mFullTissueNorm,
# [[19]] = allParAfter, [[22]] = m3r2, [[25]] = phisFMMAfter

peakRefARNTL <- phisFMMAfter[match("ARNTL", coreG)]
peakDBP      <- phisFMMAfter[match("DBP", coreG)]
escT         <- order((initialEsc - peakRefARNTL + pi) %% (2 * pi))
esc2         <- ((initialEsc - peakRefARNTL + pi) %% (2 * pi))[escT]

# Compute FMM peaks with shifted alpha
peakEpidermis2 <- c()
parCore        <- matrix(0, nG, 5)

for (i in 1:nG) {
  newAlpha <- (allParAfter[[i]][[2]] - peakRefARNTL + pi) %% (2 * pi)
  if ((peakDBP - peakRefARNTL + pi) %% (2 * pi) < pi) {
    peakEpidermis2[i] <- compUU(newAlpha, allParAfter[[i]][[3]],
                                allParAfter[[i]][[4]])
    parCore[i, ] <- c(allParAfter[[i]][[5]], allParAfter[[i]][[6]],
                       newAlpha, allParAfter[[i]][[3]], allParAfter[[i]][[4]])
  } else {
    peakEpidermis2[i] <- compUU(2 * pi - newAlpha,
                                2 * pi - allParAfter[[i]][[3]],
                                allParAfter[[i]][[4]])
    parCore[i, ] <- c(allParAfter[[i]][[5]], allParAfter[[i]][[6]],
                       2 * pi - newAlpha, 2 * pi - allParAfter[[i]][[3]],
                       allParAfter[[i]][[4]])
  }
}

# Reorder matrix
if ((peakDBP - peakRefARNTL + pi) %% (2 * pi) < pi) {
  oNew      <- escT
  escNew    <- esc2
  matNew    <- mFullTissueNorm[, escT]
  preReversed <- FALSE
} else {
  oNew      <- rev(escT)
  escNew    <- 2 * pi - rev(esc2)
  matNew    <- matrix(0, nrow(mFullTissueNorm), ncol(mFullTissueNorm))
  for (i in 1:nrow(mFullTissueNorm)) {
    matNew[i, ] <- rev(mFullTissueNorm[i, escT])
  }
  preReversed <- TRUE
}
rownames(matNew) <- rownames(mFullTissueNorm)

# Day/night classification
coreDay <- c(); namesDay <- c(); r2Day <- c()
coreNight <- c(); namesNight <- c(); r2Night <- c()
for (i in 1:length(peakEpidermis2)) {
  if (coreG[i] != "ARNTL" & coreG[i] != "DBP") {
    if (0 <= peakEpidermis2[i] & peakEpidermis2[i] < pi) {
      coreDay   <- c(coreDay, peakEpidermis2[i])
      r2Day     <- c(r2Day, r2ParAfter[i])
      namesDay  <- c(namesDay, coreG[i])
    } else {
      coreNight   <- c(coreNight, peakEpidermis2[i])
      r2Night     <- c(r2Night, r2ParAfter[i])
      namesNight  <- c(namesNight, coreG[i])
    }
  }
}

timing$pre_order <- (proc.time() - t0)["elapsed"]

write.csv(data.frame(gene = coreG, peak_time = peakEpidermis2),
          file.path(OUT_DIR, "r_pre_peaks.csv"), row.names = FALSE)
write.csv(data.frame(sample_order = oNew),
          file.path(OUT_DIR, "r_pre_sample_order.csv"), row.names = FALSE)
write.csv(data.frame(circular_scale = escNew),
          file.path(OUT_DIR, "r_pre_circular_scale.csv"), row.names = FALSE)
write.csv(data.frame(pre_reversed = preReversed),
          file.path(OUT_DIR, "r_pre_direction.csv"), row.names = FALSE)

fmmPreParams <- parCore
rownames(fmmPreParams) <- coreG
colnames(fmmPreParams) <- c("M", "A", "alpha", "beta", "omega")
write.csv(fmmPreParams, file.path(OUT_DIR, "r_fmm_params_pre.csv"))

# ===========================================================================
# STAGE 2.2-2.3: basicOder_cores (lines 4527-4593)
# ===========================================================================
cat("Stage 2.2-2.3: basic order ...\n")
t0 <- proc.time()

peaksPre <- peakEpidermis2
aviso    <- FALSE
peakD    <- 0
p6am     <- 1 - cos(peaksPre[match("DBP", coreG)])
p6pm     <- 1 - cos(peaksPre[match("DBP", coreG)] - pi)
for (i in 1:nG) {
  if (i != match("ARNTL", coreG) & i != match("DBP", coreG) &
      peaksPre[i] > 0 & peaksPre[i] < pi) {
    peakD <- peakD + 1
  }
}
mitad <- floor(nG - 2) / 2
if (p6am <= 0.1 | p6pm <= 0.1 | peakD < mitad) aviso <- TRUE
cambiaOri <- FALSE

if (aviso & !(peaksPre[match("DBP", coreG)] < peaksPre[match("CRY1", coreG)] &
              peaksPre[match("CRY1", coreG)] < peaksPre[match("ARNTL", coreG)])) {
  peaksPreNew <- (2 * pi - peaksPre) %% (2 * pi)
  oNewNew     <- rev(oNew)
  escNewNew   <- 2 * pi - rev(escNew)
  matNewNew   <- matrix(0, nrow(matNew), ncol(matNew))
  pars        <- matrix(0, nG, 5)
  for (i in 1:nG) {
    pars[i, ] <- c(parCore[i, 1], parCore[i, 2],
                   2 * pi - parCore[i, 3], 2 * pi - parCore[i, 4],
                   parCore[i, 5])
  }
  for (i in 1:nrow(matNew)) {
    matNewNew[i, ] <- rev(matNew[i, ])
  }
  rownames(matNewNew) <- rownames(matNew)
  indNewNew   <- rev(1:length(oNewNew))
  cambiaOri   <- TRUE
} else {
  peaksPreNew <- peaksPre
  oNewNew     <- oNew
  escNewNew   <- escNew
  matNewNew   <- matNew
  rownames(matNewNew) <- rownames(matNew)
  indNewNew   <- 1:length(oNewNew)
  pars        <- matrix(0, nG, 5)
  for (i in 1:nG) {
    pars[i, ] <- parCore[i, ]
  }
}

timing$basic_order <- (proc.time() - t0)["elapsed"]

write.csv(data.frame(gene = coreG, peak_time = peaksPreNew),
          file.path(OUT_DIR, "r_final_peaks.csv"), row.names = FALSE)
write.csv(data.frame(sample_order = oNewNew),
          file.path(OUT_DIR, "r_final_sample_order.csv"), row.names = FALSE)
write.csv(data.frame(circular_scale = escNewNew),
          file.path(OUT_DIR, "r_final_circular_scale.csv"), row.names = FALSE)
write.csv(data.frame(direction_flipped = cambiaOri),
          file.path(OUT_DIR, "r_final_direction.csv"), row.names = FALSE)

finalFmmParams <- pars
rownames(finalFmmParams) <- coreG
colnames(finalFmmParams) <- c("M", "A", "alpha", "beta", "omega")
write.csv(finalFmmParams, file.path(OUT_DIR, "r_fmm_params_final.csv"))

# ===========================================================================
# Timing summary
# ===========================================================================
timing_df <- data.frame(step = names(timing), seconds = unlist(timing))
write.csv(timing_df, file.path(OUT_DIR, "r_timing.csv"), row.names = FALSE)

cat("\n=== R Timing Summary ===\n")
for (i in 1:nrow(timing_df)) {
  cat(sprintf("  %-20s : %.3f s\n", timing_df$step[i], timing_df$seconds[i]))
}
cat(sprintf("  %-20s : %.3f s\n", "TOTAL (excl. load)",
            sum(timing_df$seconds[timing_df$step != "load"])))
cat(sprintf("\nReference files written to: %s\n", OUT_DIR))
