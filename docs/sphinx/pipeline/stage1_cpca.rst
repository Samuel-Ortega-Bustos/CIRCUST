Etapa 1 — CPCA (Circular PCA)
===============================

Que hace este modulo
--------------------

Dada una matriz de expresion normalizada (genes x muestras), CPCA obtiene
un **orden circular preliminar** de las muestras proyectandolas sobre el
circulo unitario mediante PCA, ademas de un filtrado.

Como funciona
-------------

1. Centrar cada gen (restar la media por filas).
2. Escalar cada muestra (dividir por RMS de columnas).
3. SVD truncada: extraer PC1 y PC2.
4. Proyectar cada muestra al circulo: ``phi = arctan2(PC2, PC1)``.
5. Detectar outliers (muestras con norma ||PC1, PC2|| pequeña).
6. Repetir sin outliers.

Ejemplo de uso
--------------

.. code-block:: python

   from circust.cpca import CPCA
   from circust.core_genes import CoreGeneSelector

   sel    = CoreGeneSelector(preset="circust")
   result = sel.select(prep.expr_norm)
   cpca = CPCA(core_genes=result.genes)
   result = cpca.run(expr_norm)

   result.sample_order      # indices que ordenan las muestras
   result.circular_scale    # angulos phi en [0, 2*pi)
   result.samples_dropped   # indices de outliers eliminados


Referencia API
--------------

Ver :doc:`/api/cpca`.
