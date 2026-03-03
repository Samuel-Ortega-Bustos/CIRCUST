# preprocessing.py  ── loader section
import pathlib
from typing import Optional

import pandas as pd


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


