Preprocessing — Carga y limpieza de datos
==========================================

Que hace este modulo
--------------------

``preprocessing.py`` es la **Etapa 0** del pipeline: transforma un fichero
de expresion genica bruta (CSV, TSV, Excel o Parquet) en una matriz limpia
y normalizada lista para CPCA o bien el filtrado de genes semilla.

El flujo completo es:

.. code-block:: text

   Fichero en disco
        |
        v
   load_expression_matrix()     <- carga y detecta formato
        |
        v
   Preprocessor.run()           <- 4 pasos de limpieza
   |  Paso 1: eliminar genes sin nombre
   |  Paso 2: eliminar genes sparse (>30% ceros o NaN)
   |  Paso 3: resolver duplicados (conservar mayor MAD)
   |  Paso 4: normalizar cada gen a [-1, 1]
        |
        v
   PreprocessingResult          <- resultado empaquetado


Ejemplo de uso
--------------

.. code-block:: python

   from circust.preprocessing import load_expression_matrix, Preprocessor

   # 1. Cargar la matriz desde un CSV
   raw = load_expression_matrix("data/raw/BA11_astrocytes.csv")
   print(raw.shape)   # (56200, 479)

   # 2. Preprocesar
   prep = Preprocessor(sparse_threshold=0.3, verbose=True)
   result = prep.run(raw)

   # 3. Usar el resultado
   print(result.expr_norm.shape)       # genes filtrados x muestras
   print(result.summary())             # resumen legible
   print(result.dropped_sparse[:5])    # primeros genes eliminados


Los 4 pasos en detalle
-----------------------

Paso 1: Genes sin nombre
^^^^^^^^^^^^^^^^^^^^^^^^^

Elimina filas cuyo simbolo genico falta, esta vacio, lleno de espacios o es ``"nan"``.
Ocurre cuando un CSV tiene filas sin rowname o con celdas vacías.

Paso 2: Genes sparse
^^^^^^^^^^^^^^^^^^^^^

Un gen se elimina si mas del 30% de sus muestras son cero **o** NaN.
El umbral se controla con ``sparse_threshold`` o bien con ``nan_threshold`` y ``zero_threshold``.

.. note::

   La comparacion es estricta (``>``): un gen con exactamente 30% de ceros
   se **conserva**.

Paso 3: Duplicados
^^^^^^^^^^^^^^^^^^^

Si un gen aparece mas de una vez (frecuente en microarrays), se conserva
la fila con mayor **MAD** (Median Absolute Deviation) y se descarta el resto.


Paso 4: Normalizacion
^^^^^^^^^^^^^^^^^^^^^^^

Cada gen se escala independientemente al rango [-1, 1] con min-max:

.. math::

   x_{norm} = 2 \cdot \frac{x - x_{min}}{x_{max} - x_{min}} - 1

Genes constantes (min == max) se mapean a 0.


Referencia de la API
--------------------

Ver :doc:`/api/preprocessing`.
