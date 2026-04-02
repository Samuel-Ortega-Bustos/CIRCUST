"""
Constantes del proyecto CIRCUST.

Genes de referencia (core clock genes):
    Zhang et al (2014):   Arntl, Dbp, Nr1d1, Nr1d2, Per1, Per2, Per3, Usp2, Tsc22d3, Tspan4
    Larriba et al (2023): PER1, PER2, PER3, CRY1, CRY2, ARNTL, CLOCK, NR1D1, RORA, DBP, TEF, STAT3
    Ruben et al. (2018):  Ver Excel, 54 genes (solo se necesitan los nombres).
"""
from math import pi


# =============================================================================
# GENES SEMILLA (CORE CLOCK GENES)
# =============================================================================

# Lista de genes semilla del articulo original CIRCUST (Larriba et al. 2023)
SEED_GENES_LARRIBA = [
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "ARNTL", "CLOCK",
    "NR1D1", "RORA",
    "DBP", "TEF", "STAT3"
]

# Lista de genes semilla propuestos por Zhang et al. (2014)
# Originalmente en raton, aqui como referencia (conversion a humano si procede)
SEED_GENES_ZHANG = [
    "ARNTL", "DBP", "NR1D1", "NR1D2", "PER1", "PER2", "PER3",
    "USP2", "TSC22D3", "TSPAN4"
]

# Genes semilla de Ruben et al. (2018) — 54 genes.
# Completar a partir del Excel de la publicacion.
SEED_GENES_RUBEN = []

# Por defecto CIRCUST usa la lista de Larriba et al. 2023.
SEED_GENES_DEFAULT = SEED_GENES_LARRIBA


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

# Proporcion maxima de ceros permitida por gen.
# Genes con mas de este umbral de muestras con valor 0 se eliminan.
ZERO_COUNT_THRESHOLD = 0.3

# Rango de normalizacion min-max. Normalmente [-1, 1].
NORM_MIN = -1.0
NORM_MAX = 1.0

# =============================================================================
# CPCA (ANALISIS DE COMPONENTES PRINCIPALES CIRCULAR)
# =============================================================================

# Numero de candidatos a outlier examinados (los mas cercanos al origen)
N_OUTLIER_CANDIDATES = 8

# Varianza minima acumulada explicada por los dos primeros eigengenes
# para considerar que la estructura circular es fiable.
MIN_TOTAL_VAR = 0.4

# Varianza minima explicada por el segundo eigengene (debe ser >= 0.1).
MIN_SECOND_VAR = 0.1

# Umbral de distancia radial para deteccion de outliers mediante CPCA.
# Muestras con distancia al origen menor que este valor se consideran outliers.
OUTLIER_RADIAL_THRESHOLD = 0.1
OUTLIER_RADIAL_THRESHOLD_LOOSE = 0.15

# Umbral de residuos estandarizados para deteccion de outliers mediante FMM.
# Muestras con |residuo| > este umbral se consideran outliers.
OUTLIER_RESIDUAL_THRESHOLD = 3.0

# =============================================================================
# ORDEN TEMPORAL PRELIMINAR
# =============================================================================

# Angulo de referencia para ARNTL (en radianes). Se fija en pi (amanecer subjetivo).
ARNTL_REF_ANGLE = pi

# =============================================================================
# MODELO FMM (FREQUENCY MODULATED MOBIUS)
# =============================================================================

# Valor minimo del parametro omega (asimetria) para considerar un gen valido.
# Genes con omega < FMM_OMEGA_MIN se descartan de la lista TOP.
FMM_OMEGA_MIN = 0.1

# Umbral de R-cuadrado para considerar un gen ritmico segun el modelo FMM.
FMM_R2_THRESHOLD = 0.5

# Numero maximo de iteraciones en la optimizacion de FMM.
FMM_MAX_ITER = 1000

# Tolerancia para la convergencia en la optimizacion.
FMM_TOL = 1e-6

# Valores iniciales por defecto para los parametros FMM (M, A, alpha, beta, omega)
FMM_INIT_M = 0.0          # Se actualizara con la media de los datos
FMM_INIT_A = 0.5          # Se actualizara con (max-min)/2
FMM_INIT_ALPHA = 0.0
FMM_INIT_BETA = 0.0
FMM_INIT_OMEGA = 0.5

# =============================================================================
# SELECCION DE GENES TOP
# =============================================================================

# Umbral de R-cuadrado FMM para que un gen sea considerado TOP.
TOP_R2_THRESHOLD = 0.5

# Umbral de omega para evitar genes demasiado puntiagudos (spiky).
TOP_OMEGA_THRESHOLD = 0.1

# Numero de cuartos del circulo en que deben distribuirse las fases pico
# de los genes TOP (para asegurar heterogeneidad). 4 = un gen por cuadrante.
TOP_N_QUARTERS = 4

# Numero de remuestreos aleatorios (K) para obtener ordenes multiples.
K_MULTIPLE_ORDERS = 5

# Fraccion de genes TOP a tomar en cada remuestreo (2/3 por defecto).
TOP_SAMPLE_FRACTION = 2/3

# Condiciones para aceptar un orden generado por remuestreo:
# - Los angulos deben cubrir mas de la mitad del circulo.
MIN_CIRCLE_COVERAGE = 0.5  # fraccion de 2*pi

# =============================================================================
# OTROS
# =============================================================================

# Semilla aleatoria para reproducibilidad.
RANDOM_SEED = 42
