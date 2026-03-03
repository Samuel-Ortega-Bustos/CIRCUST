# preprocessing.py  ── loader section
import pathlib
from typing import Optional
from dataclasses import dataclass, field
from constants import ZERO_COUNT_THRESHOLD,NORM_MIN,NORM_MAX

import pandas as pd
import numpy as np


# ── supported formats ──────────────────────────────────────────────────────
_LOADERS = {
    ".csv":  "csv",
    ".tsv":  "tsv",
    ".txt":  "tsv",   # assume tab-separated when .txt
    ".xlsx": "excel",
    ".xls":  "excel",
    ".parquet": "parquet",
}


def load_expression_matrix(
    path: str,
    gene_column: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a gene expression matrix from disk into a pandas DataFrame.

    Expected shape after loading
    ----------------------------
    - Rows    = genes  (row index = gene symbols, e.g. "ARNTL", "PER1")
    - Columns = samples (each column = one individual/sample)
    - Values  = raw expression counts (float); zeros and NaN are allowed

    Supported formats
    -----------------
    .csv            Comma-separated values
    .tsv / .txt     Tab-separated values
    .xlsx / .xls    Excel workbook (first sheet is read)
    .parquet        Apache Parquet — recommended for large datasets

    Parameters
    ----------
    path : str
        Path to the file. The format is detected from the file extension.

    gene_column : str, optional
        Name of the column that contains gene symbols.
        Pass a name explicitly if your file has a non-standard layout,
        e.g. gene_column="gene_id".

    chunk_size : int, optional
        Only used for .parquet files. Number of row-groups to read at a
        time. Useful when the file is larger than available RAM.
        If None (default), the entire file is loaded at once.
        Ignored for CSV/TSV/Excel — those formats do not support efficient
        chunked reading in a way that is meaningful for this use case.

    Returns
    -------
    pd.DataFrame
        Matrix with gene symbols as the row index (index.name = "gene_symbol")
        and sample IDs as column names. All values are float64. Cells that
        could not be parsed as numbers are set to NaN.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``path``.
    ValueError
        If the file extension is not in the supported list.
    ImportError
        If reading a .parquet file and pyarrow is not installed.

    Examples
    --------
    Load a standard CSV:

    >>> df = load_expression_matrix("data/raw/gtex_brain.csv")
    >>> df.shape
    (56200, 479)
    >>> df.index[:3].tolist()
    ['ARNTL', 'PER1', 'CRY1']

    Load a large parquet file with chunked reading:

    >>> df = load_expression_matrix(
    ...     "data/raw/gtex_brain.parquet",
    ...     chunk_size=10_000,
    ... )
    """
    file_path = pathlib.Path(path)

    # ── existence check ────────────────────────────────────────────────────
    if not file_path.exists():
        raise FileNotFoundError(
            f"Expression matrix file not found: '{path}'\n"
            f"Check the path and make sure the file is in the data/raw/ folder."
        )

    # ── format detection ───────────────────────────────────────────────────
    extension = file_path.suffix.lower()
    fmt = _LOADERS.get(extension)

    if fmt is None:
        supported = ", ".join(_LOADERS.keys())
        raise ValueError(
            f"Unsupported file format: '{extension}'\n"
            f"Supported formats: {supported}"
        )

    # ── which column becomes the row index? ───────────────────────────────
    # index_col=0 means "use the first column as the row index"
    # This is the standard layout: first column = gene names,
    # remaining columns = samples.
    # If the caller knows their gene column by name, use that instead.
    index_col = gene_column if gene_column is not None else None

    # ── loading ────────────────────────────────────────────────────────────
    if fmt == "csv":
        matrix = pd.read_csv(file_path, index_col=index_col)

    elif fmt == "tsv":
        matrix = pd.read_csv(file_path, sep="\t", index_col=index_col)

    elif fmt == "excel":
        matrix = pd.read_excel(file_path, index_col=index_col)

    elif fmt == "parquet":
        matrix = _load_parquet(file_path, index_col=index_col,
                               chunk_size=chunk_size)

    # ── enforce numeric values ─────────────────────────────────────────────
    # Some Excel files or poorly formatted CSVs contain stray strings.
    # errors="coerce" turns those into NaN instead of crashing.
    # Parquet files are already typed so this is a no-op for them.
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    # ── name the index ─────────────────────────────────────────────────────
    matrix.index.name = "gene_symbol"

    return matrix


def _load_parquet(
    file_path: pathlib.Path,
    index_col,
    chunk_size: Optional[int],
) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "Reading parquet files requires pyarrow.\n"
            "Install it with:  pip install pyarrow"
        )

    parquet_file = pq.ParquetFile(file_path)

    if chunk_size is None:
        # Use pandas directly — it correctly restores the saved index
        matrix = pd.read_parquet(file_path)
    else:
        chunks = []
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            chunks.append(batch.to_pandas())
        matrix = pd.concat(chunks, ignore_index=False)

    # Only set index manually if it wasn't restored automatically
    # (i.e. the file was written WITHOUT a named index)
    if matrix.index.name != "gene_symbol":
        if isinstance(index_col, str) and index_col in matrix.columns:
            matrix = matrix.set_index(index_col)
        elif isinstance(index_col, int) and index_col < len(matrix.columns):
            matrix = matrix.set_index(matrix.columns[index_col])

    return matrix


@dataclass
class PreprocessingResult:
    """
    All outputs produced by :class: Preprocessor.

    Having named fields instead of a positional list (like R's [[1]], [[3]])
    makes every downstream module self-documenting and protects against
    accidentally using the wrong value.

    Attributes
    ----------
    expr_norm : pd.DataFrame
        Normalised expression matrix, genes x samples, values in [-1, 1].
        This is the matrix every downstream step (CPCA, FMM) will consume.

    expr_raw : pd.DataFrame
        Filtered raw matrix — after removing bad genes but before
        normalisation. Useful for debugging and validation.

    dropped_sparse : List[str]
        Gene symbols removed because > constants.ZERO_COUNT_THRESHOLD of samples were zero or NaN.

    dropped_duplicates : List[str]
        Gene symbols removed during duplicate resolution (lower-MAD rows).

    n_genes_in : int
        Number of genes that entered the preprocessor.

    n_genes_out : int
        Number of genes that survived all filtering steps.

    n_samples : int
        Number of samples. Never changes during preprocessing.
    """

    expr_norm:           pd.DataFrame
    expr_raw:            pd.DataFrame
    dropped_sparse:      list[str] = field(default_factory=list)
    dropped_duplicates:  list[str] = field(default_factory=list)
    n_genes_in:          int = 0
    n_genes_out:         int = 0
    n_samples:           int = 0

    def summary(self) -> str:
        """
        Return a human-readable summary of what happened during preprocessing.
        Mirrors the print statements in the original R function.
        """
        lines = [
            "=== Preprocessing Summary ===",
            f"  Genes input          : {self.n_genes_in}",
            f"  Genes output         : {self.n_genes_out}",
            f"  Samples              : {self.n_samples}",
            f"  Dropped (sparse)     : {len(self.dropped_sparse)}",
            f"  Dropped (duplicates) : {len(self.dropped_duplicates)}",
        ]
        return "\n".join(lines)
    
class Preprocessor:
    """
    Clean and normalise a raw gene-expression matrix.

    Parameters
    ----------
    zero_threshold : float
        Fraction of samples allowed to be zero before a gene is dropped.
        The comparison is strict (>), so a gene at exactly 30% is kept.

    nan_threshold : float
        Same as zero_threshold but for NaN values.

    verbose : bool
        If True, print a progress message after each step.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create a small toy matrix (genes × samples)
    >>> rng = np.random.default_rng(0)
    >>> mat = pd.DataFrame(
    ...     rng.poisson(10, size=(200, 60)).astype(float),
    ...     index=[f"GENE_{i}" for i in range(200)],
    ...     columns=[f"sample_{j}" for j in range(60)],
    ... )
    >>>
    >>> result = Preprocessor().run(mat)
    >>> result.expr_norm.values.min() >= -1.0
    True
    >>> result.expr_norm.values.max() <= 1.0
    True
    """
    def __init__(
            self,
            zero_threshold: float = ZERO_COUNT_THRESHOLD,
            nan_threshold: float = ZERO_COUNT_THRESHOLD,
            verbose: bool = True
    )-> None:
        
        if not 0.0 < zero_threshold < 1.0:
            raise ValueError(
                f"zero_threshold must be between 0 and 1, got {zero_threshold}"
            )
        if not 0.0 < nan_threshold < 1.0:
            raise ValueError(
                f"nan_threshold must be between 0 and 1, got {nan_threshold}"
            )
        
        self.zero_threshold = zero_threshold
        self.nan_threshold = nan_threshold
        self.verbose = verbose

    def run(self,matrix:pd.DataFrame) -> PreprocessingResult:
        """
        Execute all four preprocessing steps in order.

        Parameters
        ----------
        matrix : pd.DataFrame
            Raw expression matrix loaded by ``load_expression_matrix()``.
            Rows = genes, columns = samples. May contain zeros and NaN.

        Returns
        -------
        PreprocessingResult
        """

        if matrix.empty:
            raise ValueError("Input matrix is empty.")
        
        n_genes_in, n_samples = matrix.shape
        self._log(f"Input matrix: {n_genes_in} genes × {n_samples} samples")

        # step 1 — remove genes with no name
        mat = self._drop_unnamed(matrix)

        # step 2 — remove sparse genes
        mat, dropped_sparse = self._drop_sparse(mat)

        # step 3 — resolve duplicate gene names
        mat, dropped_duplicates = self._resolve_duplicates(mat)

        # step 4 — normalise each gene to [NORM_MIN, NORM_MAX]
        expr_norm = self._normalise(mat)

        result = PreprocessingResult(
            expr_norm          = expr_norm,
            expr_raw           = mat,
            dropped_sparse     = dropped_sparse,
            dropped_duplicates = dropped_duplicates,
            n_genes_in         = n_genes_in,
            n_genes_out        = len(expr_norm),
            n_samples          = n_samples,
        )

        self._log(result.summary())
        return result
    
    def _drop_unnamed(self,mat:pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows whose gene symbol is NaN or an empty string.

        R equivalent (lines 3914-3915):
            indNoNames <- which(is.na(rownames(mFull0Tissue)))
            if(length(indNoNames) > 0)
                mFull0Tissue <- mFull0Tissue[-indNoNames, ]
        """
