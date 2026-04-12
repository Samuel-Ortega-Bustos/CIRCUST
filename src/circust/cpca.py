"""
circust/cpca.py
===============
Analisis de Componentes Principales Circular (CPCA).

Que es CPCA?
------------
El PCA estandar sobre una matriz de expresion genica da componentes principales
que explican varianza lineal. CPCA va un paso mas alla: toma los loadings de
PC1 y PC2 — un valor por muestra — y los trata como coordenadas (x, y) en
un plano 2D. Cada muestra se proyecta sobre el circulo unitario mediante
un angulo:

    phi = atan2(PC2_loading, PC1_loading)   en [0, 2*pi)

Ordenar las muestras por phi da un *ordenamiento circular* que refleja el
ritmo circadiano subyacente. Las muestras cercanas al origen (pequena
distancia desde (0,0) en el espacio PC1-PC2) son outliers potenciales
porque no participan fuertemente en la oscilacion dominante.

Posicion en el pipeline
-----------------------
    preprocessing.py  ->  cpca.py  ->  outlier.py  ->  ...

Entrada: PreprocessingResult.expr_norm  (matriz normalizada completa)
Salida:  CPCAResult                     (orden de muestras + flags de outliers)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from circust.core_genes import SEED_GENES_DEFAULT

# ===========================================================================
# Dataclass de resultados
# ===========================================================================

@dataclass
class CPCAResult:
    """
    Todos los resultados producidos por :class:`CPCA`.

    Atributos
    ---------
    sample_order : np.ndarray de int, forma (n_muestras,)
        Indices enteros que ordenan las muestras por su fase circular phi.
        Aplicar a cualquier eje de columnas de matriz para obtener el orden CPCA.
        Equivalente en R: ``orderCPCA8`` (obtainCPCA13 linea 3816).

    circular_scale : np.ndarray de float, forma (n_muestras,)
        Valores de phi ordenados en [0, 2*pi). Es el eje temporal circular
        usado por todos los pasos de ajuste posteriores.
        Equivalente en R: ``escalaPhi8`` (obtainCPCA13 linea 3817).

    pc1 : np.ndarray de float, forma (n_muestras,)
        Loadings de PC1 — un valor por muestra.
        Equivalente en R: ``eigen18 = cp8$rotation[,1]`` (linea 3810).

    pc2 : np.ndarray de float, forma (n_muestras,)
        Loadings de PC2 — un valor por muestra.
        Equivalente en R: ``eigen28 = cp8$rotation[,2]`` (linea 3811).

    pc3 : np.ndarray de float, forma (n_muestras,)
        Loadings de PC3 — un valor por muestra.
        Equivalente en R: ``eigen38 = cp8$rotation[,3]`` (linea 3812).

    variance_explained : np.ndarray de float, forma (3,)
        Fraccion de varianza total explicada por PC1, PC2, PC3.
        Equivalente en R: ``varPer8`` (lineas 3807-3809).

    outlier_candidate_idx : np.ndarray de int
        Indices de columna (en la matriz ORIGINAL) de las N_OUTLIER_CANDIDATES
        muestras con menor distancia PC1-PC2 al origen.
        Equivalente en R: ``obs8 = order(d8)[1:nOuts]`` (linea 3824).

    outlier_idx : np.ndarray de int
        Subconjunto de outlier_candidate_idx que realmente caen dentro del
        radio tight (<=0.10) o loose (<=0.15).
        Equivalente en R: los indices recopilados en el bucle s8 (lineas 3831-3846).

    outlier_positions_in_order : np.ndarray de int
        Posicion de cada outlier dentro de ``sample_order`` (no en la
        matriz original). Usado por graficos de residuos posteriores.
        Equivalente en R: ``outs = match(obs8[1:ss8], orderCPCA8)`` (linea 3872).

    n_outliers : int
        Numero de outliers confirmados (= len(outlier_idx)).
        Equivalente en R: ``ss8`` (linea 3847).

    used_loose_radius : bool
        True si se necesito el radio de respaldo 0.15 porque ninguna muestra
        clasifico a 0.10.
        Equivalente en R: ``rojo8`` (linea 3818 / 3843).

    core_genes_found : List[str]
        Los genes centrales que estaban realmente presentes en la matriz.
        (Generalizacion sobre R: R asume que los 12 siempre estan presentes.)

    core_matrix : pd.DataFrame, forma (n_genes_centrales, n_muestras)
        La submatriz normalizada de genes centrales usada para PCA, con
        simbolos genicos como indice. Almacenada para que plot_gene_panels()
        pueda dibujar trazas de expresion sin re-ejecutar la extraccion.
        Es None si CPCA se ejecuto con store_core_matrix=False.
    """

    sample_order:              np.ndarray
    circular_scale:            np.ndarray
    pc1:                       np.ndarray
    pc2:                       np.ndarray
    pc3:                       np.ndarray
    variance_explained:        np.ndarray
    outlier_candidate_idx:     np.ndarray
    outlier_idx:               np.ndarray
    outlier_positions_in_order: np.ndarray
    n_outliers:                int
    used_loose_radius:         bool
    core_genes_found:          list[str] = field(default_factory=list)
    core_matrix:               pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lines = [
            "=== Resumen CPCA ===",
            f"  Genes centrales usados : {len(self.core_genes_found)}  {self.core_genes_found}",
            f"  Varianza PC1           : {self.variance_explained[0]:.1%}",
            f"  Varianza PC2           : {self.variance_explained[1]:.1%}",
            f"  Varianza PC3           : {self.variance_explained[2]:.1%}",
            f"  Candidatos a outlier   : {len(self.outlier_candidate_idx)}",
            f"  Outliers confirmados   : {self.n_outliers}"
            + (" (se uso radio relajado)" if self.used_loose_radius else ""),
        ]
        return "\n".join(lines)

# ===========================================================================
# Clase CPCA
# ===========================================================================

class CPCA:
    """
    Extrae genes centrales y computa un ordenamiento circular de muestras via PCA.

    Esta clase implementa dos responsabilidades que estan acopladas en
    el codigo R pero se mantienen separadas aqui por claridad:

    1. **Extraccion de genes centrales** — extraer las 12 filas de genes
       centrales del reloj de la matriz normalizada completa (R lineas 3954-3955).

    2. **CPCA** — ejecutar PCA sobre la matriz de genes centrales centrada por
       filas, proyectar cada muestra sobre el circulo unitario usando los
       loadings de PC1 y PC2, ordenar muestras por su angulo phi, y marcar
       outliers cercanos al origen (funcion R ``obtainCPCA13``, lineas 3804-3893).

    Parametros
    ----------
    core_genes : lista de str, opcional
        Simbolos genicos a usar como conjunto ancla circadiano.
        Por defecto los 12 genes del articulo CIRCUST.
        Se puede pasar una lista personalizada si se trabaja con un organismo
        no humano o una anotacion genica diferente.

    n_outlier_candidates : int
        Cuantas muestras examinar como outliers potenciales (las de menor
        norma PC1-PC2). Valor R por defecto: 8.

    tight_radius : float
        Umbral de distancia primario. Muestras con norma <= tight_radius se
        confirman como outliers. R: 0.10.

    loose_radius : float
        Umbral de respaldo usado cuando ninguna muestra califica con tight_radius.
        R: 0.15.

    verbose : bool
        Imprime mensajes de progreso si True.

    Ejemplo
    -------
    >>> from circust.preprocessing import load_expression_matrix, Preprocessor
    >>> from circust.cpca import CPCA
    >>>
    >>> matrix  = load_expression_matrix("data/raw/expression.csv")
    >>> prep    = Preprocessor().run(matrix)
    >>> result  = CPCA().run(prep.expr_norm)
    >>> print(result.summary())
    """

    def __init__(
        self,
        core_genes:           list[str] = None,
        n_outlier_candidates: int       = 8,
        tight_radius:         float     = 0.10,
        loose_radius:         float     = 0.15,
        verbose:              bool      = True,
    ) -> None:

        self.core_genes           = core_genes if core_genes is not None else SEED_GENES_DEFAULT
        self.n_outlier_candidates = n_outlier_candidates
        self.tight_radius         = tight_radius
        self.loose_radius         = loose_radius
        self.verbose              = verbose

    # -----------------------------------------------------------------------
    # API publica
    # -----------------------------------------------------------------------

    def run(self, expr_norm: pd.DataFrame) -> CPCAResult:
        """
        Extrae genes centrales y computa el ordenamiento circular de muestras.

        Parametros
        ----------
        expr_norm : pd.DataFrame
            Matriz de expresion normalizada completa (genes x muestras,
            valores en [-1, 1]). Es ``PreprocessingResult.expr_norm``.

        Devuelve
        --------
        CPCAResult
        """
        # paso 1 — extraer filas de genes centrales de la matriz completa
        core_matrix, genes_found = self._extract_core_genes(expr_norm)

        # paso 2 — ejecutar CPCA sobre la matriz de genes centrales
        result = self._run_cpca(core_matrix, genes_found)

        self._log(result.summary())
        return result

    # -----------------------------------------------------------------------
    # Pasos privados
    # -----------------------------------------------------------------------

    def _extract_core_genes(self, expr_norm: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Selecciona las filas de genes centrales de la matriz normalizada completa."""
        genes_found  = []
        missing      = []

        for gene in self.core_genes:
            if gene in expr_norm.index:
                genes_found.append(gene)
            else:
                missing.append(gene)

        if missing:
            self._log(
                f"  AVISO — {len(missing)} gen(es) central(es) no encontrado(s) en la matriz "
                f"y se omitiran: {missing}"
            )

        if len(genes_found) < 2:
            raise ValueError(
                f"CPCA requiere al menos 2 genes centrales. "
                f"Solo se encontraron: {genes_found}. "
                f"Verifica que las filas de la matriz usen simbolos genicos estandar."
            )

        self._log(
            f"  Genes centrales extraidos: {len(genes_found)} / "
            f"{len(self.core_genes)}"
        )

        # preservar el orden de coreG, igual que match() de R
        core_matrix = expr_norm.loc[genes_found]
        return core_matrix, genes_found

    def _run_cpca(
        self,
        core_matrix: pd.DataFrame,
        genes_found: list[str],
    ) -> CPCAResult:
        """
        Ejecuta el procedimiento CPCA completo sobre la matriz normalizada de genes.
        1. Centrar por filas la matriz
        2. Escalar por columnas y ejecutar SVD (PCA)
        3. Extraer loadings de PC1, PC2, PC3 y calcular varianza explicada
        4. Calcular norma PC1-PC2 por muestra y proyectar sobre el circulo unitario
        5. Ordenar muestras por angulo phi
        6. Marcar muestras outlier (cercanas al origen)
        7. Mapear posiciones de outliers al orden clasificado
        """
        values   = core_matrix.values.astype(float)   # forma: (n_genes, n_muestras)
        n_genes, n_samples = values.shape

        # ── Paso 1: centrar por filas ───────────────────────────────────────
        # En numpy: axis=1 significa "a lo largo de columnas" -> se resta la media de cada fila.
        row_means = values.mean(axis=1, keepdims=True)
        centred   = values - row_means    # forma: (n_genes, n_muestras)

        # ── Paso 2: escalar por columnas y SVD (PCA) ────────────────────────
        # R: prcomp(centrado(mNorm8), scale.=TRUE, center=FALSE)
        #
        # En R, prcomp trata las COLUMNAS como variables y las FILAS como observaciones.
        # Aqui la matriz es (n_genes x n_muestras), por lo que:
        #   - columnas = muestras  -> variables en la vista de R
        #   - filas    = genes     -> observaciones en la vista de R
        #
        # scale.=TRUE divide cada COLUMNA (= cada muestra) por su desv. estandar.
        # Esto iguala la contribucion de cada muestra al PCA.
        #
        # sklearn PCA tambien trata las filas como observaciones. Asi que pasamos la
        # matriz tal cual (genes x muestras) y sklearn ve genes como observaciones
        # y muestras como features — replicando exactamente R.
        #
        # Tras el escalado de columnas:
        col_rms = np.sqrt(np.sum(centred**2, axis=0) / (n_genes - 1))
        col_rms[col_rms == 0] = 1.0          # evitar division por cero
        scaled  = centred / col_rms          # forma: (n_genes, n_muestras)

        # POR QUE NO sklearn PCA?
        # sklearn PCA siempre resta las medias de columnas internamente antes de SVD
        # y no tiene opcion center=False. Pasar nuestra matriz a PCA anadira un
        # segundo centrado no deseado sobre centrado().
        #
        # POR QUE NO TruncatedSVD.explained_variance_ratio_?
        # TruncatedSVD calcula su ratio como var(proyecciones) / var(X_entrada),
        # que usa un denominador diferente a R. R usa
        # sigma_k^2 / sum(TODOS sigma^2) — un ratio puramente de valores singulares.
        # Los numeros difieren en ~0.1%, lo cual importa para un port fiel.
        #
        # SOLUCION: SVD completo de numpy para TODOS los valores singulares para
        # obtener el denominador correcto.
        n_components = min(3, n_genes, n_samples)
        _, sigma_all, Vt = np.linalg.svd(scaled, full_matrices=False)

        # ── Paso 3: extraer loadings y calcular varianza explicada ──────────
        # Vt tiene forma (n_componentes, n_muestras); cada fila es un PC.
        # Equivalente a cp8$rotation de R (columnas PC1, PC2, PC3).
        pc1 = Vt[0]
        pc2 = Vt[1]
        pc3 = Vt[2] if n_components >= 3 else np.zeros(n_samples)

        var_exp = np.zeros(3)
        var_exp[:n_components] = (sigma_all[:n_components]**2) / np.sum(sigma_all**2)

        # ── Paso 4: norma PC1-PC2 y proyeccion sobre el circulo unitario ────
        # Equivalente a la normalizacion z_p = a_p/r, z_q = a_q/r del paper
        # (Scholz 2007, eq. 5). La norma mide cuanto participa cada muestra
        # en la oscilacion dominante; muestras cercanas al origen son outliers.
        norm12    = np.sqrt(pc1**2 + pc2**2)                        # R: d8
        safe_norm = np.where(norm12 == 0, 1.0, norm12)

        xi  = pc1 / safe_norm
        yi  = pc2 / safe_norm
        phi = np.arctan2(yi, xi) % (2 * np.pi)                     # R: phi8

        # ── Paso 5: ordenar muestras por angulo ────────────────────────────
        sample_order   = np.argsort(phi)                            # R: orderCPCA8
        circular_scale = phi[sample_order]                          # R: escalaPhi8

        # ── Paso 6: marcar muestras outlier ────────────────────────────────
        # Tomar las n_outlier_candidates muestras con menor norma (mas cercanas
        # al origen) y confirmarlas si caen dentro del radio tight o loose.
        n_cands       = min(self.n_outlier_candidates, n_samples)
        candidate_idx = np.argsort(norm12)[:n_cands]                # R: obs8

        tight_mask = norm12[candidate_idx] <= self.tight_radius
        used_loose = False

        if tight_mask.any():
            outlier_idx = candidate_idx[tight_mask]
        else:
            loose_mask  = norm12[candidate_idx] <= self.loose_radius
            outlier_idx = candidate_idx[loose_mask]
            used_loose  = loose_mask.any()

        n_outliers = len(outlier_idx)

        # ── Paso 7: mapear posiciones de outliers al orden clasificado ─────
        # Para cada indice original de outlier, encontrar su posicion en sample_order.
        if n_outliers > 0:
            outlier_positions = np.array([
                int(np.where(sample_order == idx)[0][0])
                for idx in outlier_idx
            ])
        else:
            outlier_positions = np.array([], dtype=int)

        return CPCAResult(
            sample_order               = sample_order,
            circular_scale             = circular_scale,
            pc1                        = pc1,
            pc2                        = pc2,
            pc3                        = pc3,
            variance_explained         = var_exp,
            outlier_candidate_idx      = candidate_idx,
            outlier_idx                = outlier_idx,
            outlier_positions_in_order = outlier_positions,
            n_outliers                 = n_outliers,
            used_loose_radius          = used_loose,
            core_genes_found           = genes_found,
            core_matrix                = core_matrix,
        )

    # -----------------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
