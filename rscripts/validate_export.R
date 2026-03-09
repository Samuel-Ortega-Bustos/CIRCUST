# =============================================================================
# rscripts/r_export.R
# =============================================================================
# Runs CIRCUST preprocessing + CPCA on matrixIn, saves reference outputs
# to CSV, and times each step.
#
# Run from the repo ROOT (circust/):
#   Rscript rscripts/r_export.R
#
# WHY WE DON'T source("functionGTEX_cores.R") DIRECTLY:
#   The top of that file calls install.packages("Iso"), require("circular"),
#   and sources upDownUp_NP_Code_NoParalelizado.R which calls
#   dyn.load("pava.dll") — a Windows-only compiled binary.
#   For preprocessing + CPCA we only need functions defined in
#   functionGTEX_cores.R itself and FMM.R, so we load just those
#   by reading the file and skipping the problematic lines.
#
# Outputs written to:  validation/reference/
# =============================================================================

REPO_ROOT  <- normalizePath(".")
RSCRIPTS   <- file.path(REPO_ROOT, "rscripts")
RDATA_PATH <- file.path(REPO_ROOT, "data", "matrixIn.RData")
OUT_DIR    <- file.path(REPO_ROOT, "validation", "reference")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Load only what preprocessing + CPCA need, skipping Windows .dll calls
# ---------------------------------------------------------------------------
old_wd <- getwd()
setwd(RSCRIPTS)

cat("Loading R source files (skipping Windows .dll dependencies)...\n")

source("FMM.R")

lines <- readLines("functionGTEX_cores.R")

skip_patterns <- c(
  "install.packages",
  'source("upDownUp_NP_Code_NoParalelizado.R")',
  'source("upDownUp_NP_Code_Paralelizado.R")',
  'source("upDownUp_NP_Code_modif.R")',
  'source("NucleoComun.R")'
)

keep <- rep(TRUE, length(lines))
for (pat in skip_patterns) {
  keep <- keep & !grepl(pat, lines, fixed = TRUE)
}

eval(parse(text = paste(lines[keep], collapse = "\n")), envir = .GlobalEnv)
setwd(old_wd)

cat("  Done. normalice, centrado, obtainCPCA13, fitFMM_Par ready.\n")

# ---------------------------------------------------------------------------
coreG <- c("ARNTL", "DBP", "NR1D1", "NR1D2", "PER1", "PER2", "PER3","USP2", "TSC22D3", "TSPAN4")
timing <- list()

# ---------------------------------------------------------------------------
cat("\nLoading matrixIn.RData ...\n")
t0 <- proc.time()
#matrixIn <- read.csv("~/Documents/Universidad/TFG-Statistics/CIRCUST/data/BA46_glut_sample_no_minmax.csv", row.names=1)
load(RDATA_PATH)
timing$load <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Loaded: %d genes x %d samples\n", nrow(matrixIn), ncol(matrixIn)))

# ---------------------------------------------------------------------------
# Step 1 — Drop unnamed genes (lines 3914-3915)
# ---------------------------------------------------------------------------
cat("Step 1: dropping unnamed genes ...\n")
t0 <- proc.time()
mFull0 <- matrixIn
indNoNames <- which(is.na(rownames(mFull0)), arr.ind = TRUE)
if (length(indNoNames) > 0) mFull0 <- mFull0[-indNoNames, ]
timing$drop_unnamed <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Removed %d unnamed genes\n", length(indNoNames)))

# ---------------------------------------------------------------------------
# Step 2 — Drop sparse genes >30% zeros or NaN (lines 3918-3933)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Step 3 — Resolve duplicates, keep highest MAD (lines 3936-3948)
# ---------------------------------------------------------------------------
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
write.csv(mFull1, file.path(OUT_DIR, "r_expr_raw.csv"))

# ---------------------------------------------------------------------------
# Step 4 — Normalise each gene to [-1, 1] (lines 3949-3950)
# ---------------------------------------------------------------------------
cat("Step 4: normalising ...\n")
t0 <- proc.time()
mFullNorm <- t(apply(mFull1, 1, normalice))
rownames(mFullNorm) <- rownames(mFull1)
timing$normalise <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Normalised: %d genes x %d samples\n", nrow(mFullNorm), ncol(mFullNorm)))
write.csv(mFullNorm, file.path(OUT_DIR, "r_expr_norm.csv"))

# ---------------------------------------------------------------------------
# Core gene extraction (lines 3954-3955)
# ---------------------------------------------------------------------------
mCoreNorm <- mFullNorm[match(coreG, rownames(mFullNorm)), ]
write.csv(mCoreNorm, file.path(OUT_DIR, "r_core_norm.csv"))

# ---------------------------------------------------------------------------
# CPCA — obtainCPCA13 (lines 3804-3893)
# ---------------------------------------------------------------------------
cat("CPCA: running circular PCA ...\n")
t0 <- proc.time()
cpca <- obtainCPCA13(mCoreNorm, "Example", coreG, 8, FALSE)
timing$cpca <- (proc.time() - t0)["elapsed"]

write.csv(data.frame(sample_order      = cpca[[1]]),
          file.path(OUT_DIR, "r_sample_order.csv"),    row.names = FALSE)
write.csv(data.frame(circular_scale    = cpca[[2]]),
          file.path(OUT_DIR, "r_circular_scale.csv"),  row.names = FALSE)
write.csv(data.frame(pc1 = cpca[[6]], pc2 = cpca[[7]], pc3 = cpca[[8]]),
          file.path(OUT_DIR, "r_pc_loadings.csv"),     row.names = FALSE)
write.csv(data.frame(variance_explained = cpca[[5]]),
          file.path(OUT_DIR, "r_variance.csv"),        row.names = FALSE)
write.csv(data.frame(candidate_idx = cpca[[3]][1, 1:8]),
          file.path(OUT_DIR, "r_outlier_candidates.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# Timing summary
# ---------------------------------------------------------------------------
timing_df <- data.frame(step = names(timing), seconds = unlist(timing))
write.csv(timing_df, file.path(OUT_DIR, "r_timing.csv"), row.names = FALSE)

cat("\n=== R Timing Summary ===\n")
for (i in 1:nrow(timing_df)) {
  cat(sprintf("  %-20s : %.3f s\n", timing_df$step[i], timing_df$seconds[i]))
}
cat(sprintf("  %-20s : %.3f s\n", "TOTAL (excl. load)",
            sum(timing_df$seconds[timing_df$step != "load"])))
cat(sprintf("\nReference files written to: %s\n", OUT_DIR))