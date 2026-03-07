"""Constants for the CIRCUST project

Top Genes:
Zhang et al (2014):  Arntl, Dbp, Nr1d1, Nr1d2, Per1, Per2, and Per3, Usp2, Tsc22d3, and Tspan4
Larriba et al (2023): PER1, PER2, PER3, CRY1, CRY2, ARNTL, CLOCK, NR1D1, RORA, DBP, TEF and STAT3
Ruben et al. (2018): Ver Excel, son 54, solo necesitas los nombres.
"""
from math import pi


# Genes semilla utilizados en el artículo original de CIRCUST (Larriba et al. 2023)
SEED_GENES_LARRIBA = [
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "ARNTL", "CLOCK",
    "NR1D1", "RORA",
    "DBP", "TEF", "STAT3"
]

# Genes semilla propuestos por Zhang et al. (2014) - originalmente en ratón,
# se muestran aquí como referencia (pueden necesitar conversión a humano).
SEED_GENES_ZHANG = [
    "ARNTL", "DBP", "NR1D1", "NR1D2", "PER1", "PER2", "PER3",
    "USP2", "TSC22D3", "TSPAN4"
]

# Genes semilla de Ruben et al. (2018) - 54 genes. Se recomienda completar
# esta lista a partir del archivo Excel proporcionado en la publicación.
# Ejemplo (primeros 5):
# SEED_GENES_RUBEN = ["GEN1", "GEN2", ...]
SEED_GENES_RUBEN = []  # Completa según tus necesidades

# Por defecto, CIRCUST usa la lista de Larriba et al. 2023.
# Puedes cambiar esta variable para usar otra lista.
SEED_GENES_DEFAULT = SEED_GENES_ZHANG


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

# Proporción máxima de ceros permitida por gen.
# Genes con más de ZERO_COUNT_THRESHOLD (ej. 30%) de muestras con valor 0 se eliminan.
ZERO_COUNT_THRESHOLD = 0.3

# Rango de normalización min-max. Normalmente [-1, 1].
NORM_MIN = -1.0
NORM_MAX = 1.0

# =============================================================================
# CPCA (CIRCULAR PRINCIPAL COMPONENT ANALYSIS)
# =============================================================================

#Candidatos outliers
N_OUTLIER_CANDIDATES = 8

# Varianza mínima acumulada explicada por los dos primeros eigengenes
# para considerar que la estructura circular es fiable.
MIN_TOTAL_VAR = 0.4

# Varianza mínima explicada por el segundo eigengene (debe ser al menos 0.1).
MIN_SECOND_VAR = 0.1

# Umbral de distancia radial para detección de outliers mediante CPCA.
# Muestras con distancia al origen menor que este valor se consideran outliers.
OUTLIER_RADIAL_THRESHOLD = 0.1
OUTLIER_RADIAL_THRESHOLD_LOOSE = 0.15

# Umbral de residuos estandarizados para detección de outliers mediante FMM.
# Muestras con |residuo| > OUTLIER_RESIDUAL_THRESHOLD se consideran outliers.
OUTLIER_RESIDUAL_THRESHOLD = 3.0

# =============================================================================
# ORDEN TEMPORAL PRELIMINAR
# =============================================================================

# Ángulo de referencia para ARNTL (en radianes). Se fija en π (mitad del día).
ARNTL_REF_ANGLE = pi  # π

# Supuestos para elegir la dirección (sentido horario/antihorario):
# - DBP debe aparecer después de los genes de la familia ROR (ARNTL, CLOCK, NPAS...)
# - La mayoría de los genes del reloj principal tienen su pico en el periodo activo [0, π).
# Estas reglas se implementan directamente en order_estimation.py, no como constantes.

# =============================================================================
# MODELO FMM (FREQUENCY MODULATED MÖBIUS)
# =============================================================================

# Valor mínimo del parámetro ω (sharpness) para considerar un gen no "spiky".
# Genes con ω < FMM_OMEGA_MIN se descartan de la lista TOP.
FMM_OMEGA_MIN = 0.1

# Umbral de R² para considerar un gen rítmico según el modelo FMM.
FMM_R2_THRESHOLD = 0.5

# Número máximo de iteraciones en la optimización de FMM.
FMM_MAX_ITER = 1000

# Tolerancia para la convergencia en la optimización.
FMM_TOL = 1e-6

# Valores iniciales por defecto para los parámetros FMM (M, A, alpha, beta, omega)
FMM_INIT_M = 0.0          # Se actualizará con la media de los datos
FMM_INIT_A = 0.5          # Se actualizará con (max-min)/2
FMM_INIT_ALPHA = 0.0
FMM_INIT_BETA = 0.0
FMM_INIT_OMEGA = 0.5

# =============================================================================
# SELECCIÓN DE GENES TOP
# =============================================================================

# Umbral de R² FMM para que un gen sea considerado TOP.
TOP_R2_THRESHOLD = 0.5

# Umbral de ω para evitar genes demasiado puntiagudos (spiky).
TOP_OMEGA_THRESHOLD = 0.1

# Número de cuartos del círculo en que deben distribuirse las fases pico
# de los genes TOP (para asegurar heterogeneidad). El valor 4 significa
# que debe haber al menos un gen con pico en cada cuadrante [0, π/2), etc.
TOP_N_QUARTERS = 4

# Número de remuestreos aleatorios (K) para obtener órdenes múltiples.
K_MULTIPLE_ORDERS = 5

# Fracción de genes TOP a tomar en cada remuestreo (2/3 por defecto).
TOP_SAMPLE_FRACTION = 2/3

# Condiciones para aceptar un orden generado por remuestreo:
# - Los ángulos deben cubrir más de la mitad del círculo.
MIN_CIRCLE_COVERAGE = 0.5  # fracción de 2π

# - La distancia máxima entre ángulos consecutivos no debe exceder
#   la máxima distancia observada en el orden preliminar.
#   Este cálculo se realiza dinámicamente, no es una constante.

# =============================================================================
# OTROS
# =============================================================================

# Semilla aleatoria para reproducibilidad (opcional, puede fijarse en el script).
RANDOM_SEED = 42
