Etapa 1 — CPCA (Circular PCA)
===============================

Que problema resuelve
---------------------

Dada una matriz de expresion normalizada (genes x muestras), CPCA obtiene
un **orden circular preliminar** de las muestras proyectandolas sobre el
circulo unitario mediante PCA.

Como funciona
-------------

1. Centrar cada gen (restar la media por filas).
2. Escalar cada muestra (dividir por RMS de columnas).
3. SVD truncada: extraer PC1 y PC2.
4. Proyectar cada muestra al circulo: ``phi = arctan2(PC2, PC1)``.
5. Detectar outliers (muestras con norma ||PC1, PC2|| pequena).
6. Repetir sin outliers hasta convergencia.

Ejemplo de uso
--------------

.. code-block:: python

   from circust.cpca import CPCA

   cpca = CPCA(core_genes=["ARNTL", "PER1", "PER2", "CRY1"])
   result = cpca.run(expr_norm)

   result.sample_order      # indices que ordenan las muestras
   result.circular_scale    # angulos phi en [0, 2*pi)
   result.samples_dropped   # indices de outliers eliminados


Referencia API
--------------

Ver :doc:`/api/cpca`.
