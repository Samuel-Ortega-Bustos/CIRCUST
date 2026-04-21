Etapa 3 — Seleccion de genes TOP
==================================

Que problema resuelve
---------------------

No todos los genes del genoma son ritmicos. Esta etapa selecciona el
subconjunto de genes con **patron circadiano significativo** (los genes
TOP) que se usaran para refinar el orden circular.

Como funciona
-------------

1. Ajustar FMM y Cosinor a cada gen sobre la escala circular.
2. Filtrar por R² minimo (calidad del ajuste).
3. Clasificar cada gen superviviente en uno de 8 sectores del circulo
   segun la fase de su pico.
4. Asegurar cobertura: si faltan genes core, anadirlos forzosamente.

Ejemplo de uso
--------------

.. code-block:: python

   from circust.top_genes import TopGeneSelector

   selector = TopGeneSelector(r2_threshold=0.5)
   top_result = selector.run(sync_result, expr_norm, core_genes)

   top_result.gene_names       # nombres de los genes TOP
   top_result.sector_labels    # sector (1-8) de cada gen
   top_result.candidate_matrix # submatriz filtrada


Referencia API
--------------

Ver :doc:`/api/top_genes`.
