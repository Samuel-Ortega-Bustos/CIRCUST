# preprocessing.py  ── seccion de carga y preprocesamiento
import pathlib
from typing import Optional
from dataclasses import dataclass, field
from scipy.stats import median_abs_deviation

import pandas as pd
import numpy as np


# ── formatos soportados ───────────────────────────────────────────────────
_LOADERS = {
    ".csv":  "csv",
    ".tsv":  "tsv",
    ".txt":  "tsv",       # se asume separado por tabuladores cuando .txt
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
    Carga una matriz de expresion genica desde disco en un DataFrame de pandas.

    Forma esperada tras la carga
    ----------------------------
    - Filas    = genes  (indice de fila = simbolos genicos, ej. "ARNTL", "PER1")
    - Columnas = muestras (cada columna = un individuo/muestra)
    - Valores  = conteos de expresion brutos (float); ceros y NaN permitidos

    Formatos soportados
    -------------------
    .csv            Valores separados por comas
    .tsv / .txt     Valores separados por tabuladores
    .xlsx / .xls    Libro Excel (se lee la primera hoja)
    .parquet        Apache Parquet — recomendado para conjuntos grandes

    Parameters
    ----------
    path : str
        Ruta al archivo. El formato se detecta por la extension.

    gene_column : str, opcional
        Nombre de la columna que contiene los simbolos genicos.
        Especificar explicitamente si el archivo tiene un layout no estandar,
        ej. gene_column="gene_id".

    chunk_size : int, opcional
        Solo para archivos .parquet. Numero de row-groups a leer por vez.
        Util cuando el archivo es mas grande que la RAM disponible.
        Si None (defecto), se carga el archivo completo de una vez.
        Ignorado para CSV/TSV/Excel.

    Returns
    -------
    pd.DataFrame
        Matriz con simbolos genicos como indice de fila (index.name = "gene_symbol")
        e IDs de muestra como nombres de columna. Todos los valores son float64.
        Celdas que no se pudieron parsear como numeros se establecen a NaN.

    Raises
    ------
    FileNotFoundError
        Si no existe archivo en ``path``.
    ValueError
        Si la extension del archivo no esta en la lista soportada.
    ImportError
        Si se lee un .parquet y pyarrow no esta instalado.

    Example
    -------
    Cargar un CSV estandar:

    >>> df = load_expression_matrix("data/raw/gtex_brain.csv")
    >>> df.shape
    (56200, 479)
    >>> df.index[:3].tolist()
    ['ARNTL', 'PER1', 'CRY1']

    Cargar un parquet grande con lectura por bloques:

    >>> df = load_expression_matrix(
    ...     "data/raw/gtex_brain.parquet",
    ...     chunk_size=10_000,
    ... )
    """
    file_path = pathlib.Path(path)

    # ── verificacion de existencia ────────────────────────────────────────
    if not file_path.exists():
        raise FileNotFoundError(
            f"Archivo de matriz de expresion no encontrado: '{path}'\n"
            f"Verifica la ruta y asegurate de que el archivo esta en la carpeta data/."
        )

    # ── deteccion de formato ──────────────────────────────────────────────
    extension = file_path.suffix.lower()
    fmt = _LOADERS.get(extension)

    if fmt is None:
        supported = ", ".join(_LOADERS.keys())
        raise ValueError(
            f"Formato de archivo no soportado: '{extension}'\n"
            f"Formatos soportados: {supported}"
        )

    # ── que columna se convierte en el indice de fila? ────────────────────
    # index_col=0 significa "usar la primera columna como indice de fila"
    # Layout estandar: primera columna = nombres de genes,
    # columnas restantes = muestras.
    # Si el usuario conoce su columna de genes por nombre, usar esa.
    if fmt in ("csv", "tsv", "excel"):
        index_col = gene_column if gene_column is not None else 0
    elif fmt == "parquet":
        if gene_column is None:
            raise ValueError(
                "Para archivos Parquet debes especificar el nombre de la columna "
                "que contiene los identificadores de genes mediante el parametro 'gene_column'."
            )
        index_col = gene_column
    else:
        # no deberia ocurrir porque ya validamos fmt
        index_col = None

    # ── carga ─────────────────────────────────────────────────────────────
    if fmt == "csv":
        matrix = pd.read_csv(file_path, index_col=index_col)

    elif fmt == "tsv":
        matrix = pd.read_csv(file_path, sep="\t", index_col=index_col)

    elif fmt == "excel":
        matrix = pd.read_excel(file_path, index_col=index_col)

    elif fmt == "parquet":
        matrix = _load_parquet(file_path, index_col=index_col,
                               chunk_size=chunk_size)

    # ── forzar valores numericos ──────────────────────────────────────────
    # Algunos Excel o CSV mal formateados contienen cadenas sueltas.
    # errors="coerce" las convierte en NaN en lugar de fallar.
    # Los archivos Parquet ya estan tipados asi que esto es un no-op para ellos.
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    # ── nombrar el indice ─────────────────────────────────────────────────
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
            "Leer archivos parquet requiere pyarrow.\n"
            "Instalalo con:  pip install pyarrow"
        )

    parquet_file = pq.ParquetFile(file_path)

    if chunk_size is None:
        # Usar pandas directamente — restaura correctamente el indice guardado
        matrix = pd.read_parquet(file_path)
    else:
        chunks = []
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            chunks.append(batch.to_pandas())
        matrix = pd.concat(chunks, ignore_index=False)

    # Solo establecer indice manualmente si no se restauro automaticamente
    # (es decir, el archivo se escribio SIN un indice con nombre)
    if matrix.index.name != "gene_symbol":
        if isinstance(index_col, str) and index_col in matrix.columns:
            matrix = matrix.set_index(index_col)
        elif isinstance(index_col, int) and index_col < len(matrix.columns):
            matrix = matrix.set_index(matrix.columns[index_col])

    return matrix


@dataclass
class PreprocessingResult:
    """
    Todos los resultados producidos por :class:`Preprocessor`.

    Attributes
    ----------
    expr_norm : pd.DataFrame
        Matriz de expresion normalizada, genes x muestras, valores en [-1, 1].
        Es la matriz que consume cada paso (CPCA, FMM).

    expr_raw : pd.DataFrame
        Matriz bruta filtrada despues de eliminar genes malos pero antes
        de normalizar. Util para depuracion y validacion.

    dropped_sparse : List[str]
        Simbolos genicos eliminados porque > zero_threshold de muestras eran cero o NaN.

    dropped_duplicates : List[str]
        Simbolos genicos eliminados durante la resolucion de duplicados (filas con menor MAD).

    n_genes_in : int
        Numero de genes que entraron al preprocesador.

    n_genes_out : int
        Numero de genes que sobrevivieron a todos los pasos de filtrado.

    n_samples : int
        Numero de muestras. No cambia nunca durante el preprocesamiento.
    """

    expr_norm:           pd.DataFrame
    expr_raw:            pd.DataFrame
    dropped_sparse:      list[str] = field(default_factory=list)
    dropped_duplicates:  list[str] = field(default_factory=list)
    n_genes_in:          int = 0
    n_genes_out:         int = 0
    n_samples:           int = 0

    def summary(self) -> str:
        """Devuelve un resumen legible de lo ocurrido durante el preprocesamiento."""
        lines = [
            "=== Resumen del Preprocesamiento ===",
            f"  Genes entrada        : {self.n_genes_in}",
            f"  Genes salida         : {self.n_genes_out}",
            f"  Muestras             : {self.n_samples}",
            f"  Eliminados (sparse)  : {len(self.dropped_sparse)}",
            f"  Eliminados (duplicados): {len(self.dropped_duplicates)}",
        ]
        if self.dropped_sparse:
            shown = self.dropped_sparse[:5]
            tail  = f" ... (+{len(self.dropped_sparse)-5} mas)" if len(self.dropped_sparse) > 5 else ""
            lines.append(f"    genes sparse       : {shown}{tail}")
        if self.dropped_duplicates:
            unique_dupes = sorted(set(self.dropped_duplicates))[:5]
            tail = f" ... (+{len(set(self.dropped_duplicates))-5} mas)" if len(set(self.dropped_duplicates)) > 5 else ""
            lines.append(f"    genes duplicados   : {unique_dupes}{tail}")
        return "\n".join(lines)


def normalise_matrix(
    mat: pd.DataFrame,
    norm_min: float = -1.0,
    norm_max: float =  1.0,
) -> pd.DataFrame:
    """
    Normaliza min-max cada fila de ``mat`` al intervalo [norm_min, norm_max].

    Filas constantes (min == max) se mapean a todos ceros.

    Parameters
    ----------
    mat : pd.DataFrame
        Matriz genes × muestras. Valores pueden ser float o int.
    norm_min, norm_max : float
        Extremos del rango de salida. Por defecto [-1, 1].

    Returns
    -------
    pd.DataFrame con la misma forma e indices que ``mat``.
    """
    values  = mat.values.astype(float)
    row_min = values.min(axis=1, keepdims=True)
    row_max = values.max(axis=1, keepdims=True)
    span    = row_max - row_min
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.where(
            span == 0,
            0.0,
            (norm_max - norm_min) * ((values - row_min) / span) + norm_min,
        )
    return pd.DataFrame(normed, index=mat.index, columns=mat.columns)


class Preprocessor:
    """
    Limpia y normaliza una matriz de expresion genica bruta.

    1. Eliminar genes sin nombre (NA rownames en R)
    2. Eliminar genes sparse (> ``sparse_threshold`` de muestras son cero O NaN)
    3. Resolver nombres de genes duplicados — conservar la fila con mayor MAD
    4. Normalizar cada gen a [-1, 1]

    Parameters
    ----------
    sparse_threshold : float
        Fraccion de muestras que pueden ser cero o NaN antes de que un gen
        se elimine. La comparacion es estricta (>), asi que un gen
        exactamente en el umbral se conserva.

    zero_threshold : float o None
        Override para el umbral solo de ceros. Si None (defecto), usa
        ``sparse_threshold`` para ceros.

    nan_threshold : float o None
        Override para el umbral solo de NaN. Si None (defecto), usa
        ``sparse_threshold`` para NaN.

    verbose : bool
        Si True, imprime mensajes de progreso tras cada paso.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
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
            sparse_threshold: float         = 0.3,
            zero_threshold:   Optional[float] = None,
            nan_threshold:    Optional[float] = None,
            norm_min:         float         = -1.0,
            norm_max:         float         =  1.0,
            verbose:          bool          = True,
    ) -> None:

        if not 0.0 < sparse_threshold < 1.0:
            raise ValueError(
                f"sparse_threshold debe estar entre 0 y 1, se recibio {sparse_threshold}"
            )

        self.sparse_threshold = sparse_threshold
        self.zero_threshold   = zero_threshold if zero_threshold is not None else sparse_threshold
        self.nan_threshold    = nan_threshold  if nan_threshold  is not None else sparse_threshold
        self.norm_min         = norm_min
        self.norm_max         = norm_max
        self.verbose          = verbose

    def run(self, matrix: pd.DataFrame) -> PreprocessingResult:
        """
        Ejecuta los cuatro pasos de preprocesamiento en orden.

        Parameters
        ----------
        matrix : pd.DataFrame
            Matriz de expresion bruta cargada por ``load_expression_matrix()``.
            Filas = genes, columnas = muestras. Puede contener ceros y NaN.

        Returns
        -------
        PreprocessingResult
        """

        if matrix.empty:
            raise ValueError("La matriz de entrada esta vacia.")

        n_genes_in, n_samples = matrix.shape
        self._log(f"Matriz de entrada: {n_genes_in} genes x {n_samples} muestras")

        # paso 1 — eliminar genes sin nombre
        mat = self._drop_unnamed(matrix)

        # paso 2 — eliminar genes sparse
        mat, dropped_sparse = self._drop_sparse(mat)

        # paso 3 — resolver nombres de genes duplicados
        mat, dropped_duplicates = self._resolve_duplicates(mat)

        # paso 4 — normalizar cada gen a [-1, 1]
        expr_norm = normalise_matrix(mat, self.norm_min, self.norm_max)

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

    def _drop_unnamed(self, mat: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina filas cuyo simbolo genico falta o esta efectivamente vacio.
        - NaN real en el indice (R: NA rowname)
        - Cadena vacia ``""``
        - Cadena de solo espacios ``"   "``
        - La cadena literal ``"nan"`` (ocurre cuando NaN se convierte a str)
        """
        index_str = mat.index.astype(str)
        is_invalid = (
            mat.index.isna()                          # NaN real
            | (index_str.str.strip() == "")           # vacio / solo espacios
            | (index_str.str.lower() == "nan")        # NaN convertido a cadena
        )

        n_bad = int(is_invalid.sum())
        if n_bad > 0:
            self._log(f"  Paso 1 — eliminados {n_bad} gen(es) sin nombre")
        else:
            self._log("  Paso 1 — no se encontraron genes sin nombre")

        return mat.loc[~is_invalid]

    def _drop_sparse(self, mat: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Elimina genes donde mas de zero_threshold de muestras son cero O NaN.

        Nota: la condicion es estrictamente > asi que un gen exactamente en el
        30% se conserva. Preservado aqui: (frac > threshold), no (frac >= threshold).
        """
        # fraccion de ceros y NaN por gen (por fila)
        zero_frac = (mat == 0).mean(axis=1)
        nan_frac  = mat.isna().mean(axis=1)

        is_sparse = (zero_frac > self.zero_threshold) | (nan_frac > self.nan_threshold)

        dropped = mat.index[is_sparse].tolist()

        # replica el print exacto de R (linea 3925)
        self._log(
            f"  Paso 2 — eliminados {len(dropped)} gen(es) con mas del "
            f"{int(self.zero_threshold * 100)}% de ceros o NaN"
        )

        return mat.loc[~is_sparse].copy(), dropped

    def _resolve_duplicates(self, mat: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Para cada nombre de gen que aparece mas de una vez, conserva la fila
        con mayor MAD y descarta el resto.
        """
        # encontrar nombres de genes que aparecen mas de una vez
        duplicated_mask  = mat.index.duplicated(keep=False)
        duplicated_names = mat.index[duplicated_mask].unique().tolist()

        if not duplicated_names:
            self._log("  Paso 3 — no se encontraron nombres de genes duplicados")
            return mat, []

        self._log(
            f"  Paso 3 — resolviendo {len(duplicated_names)} "
            f"nombre(s) de gen duplicado(s)"
        )

        # recopilar posiciones enteras de filas a eliminar
        rows_to_drop:  list[int] = []
        dropped_names: list[str] = []

        for gene in duplicated_names:
            # np.where da posiciones enteras — necesario porque dos filas
            # comparten la misma etiqueta, asi que .loc devolveria ambas
            positions = np.where(mat.index == gene)[0]

            # calcular MAD para cada fila duplicada usando scale=1 para replicar R, con default seria "1.482"
            mads = np.array([
                float(median_abs_deviation(mat.iloc[pos].values, scale=1))
                for pos in positions
            ])

            # conservar la fila con mayor MAD, eliminar el resto
            best_position   = positions[int(np.argmax(mads))]
            loser_positions = [p for p in positions if p != best_position]

            rows_to_drop.extend(loser_positions)
            dropped_names.extend([gene] * len(loser_positions))

        # construir mascara booleana de conservacion y aplicar
        keep_mask = np.ones(len(mat), dtype=bool)
        keep_mask[rows_to_drop] = False

        return mat.iloc[keep_mask].copy(), dropped_names

    # -----------------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Imprime solo cuando verbose=True."""
        if self.verbose:
            print(message)
