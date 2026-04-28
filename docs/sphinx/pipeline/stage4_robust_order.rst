Etapa 4 — Orden robusto
========================

Que problema resuelve
---------------------

El orden circular depende de que genes se usen. Si cambias ligeramente
el conjunto, el orden puede cambiar. Esta etapa obtiene un orden
**robusto** repitiendo el proceso K veces con subconjuntos aleatorios
y combinando los resultados.

Como funciona
-------------

1. **Seleccion aleatoria controlada**: generar K subconjuntos de 2/3 de
   los genes TOP, verificando cobertura de cuadrantes y calidad.

2. **CPCA + sincronizacion por repeticion**: para cada subconjunto,
   ejecutar CPCA, sincronizar, y re-ajustar FMM/Cosinor/NP.

3. **Calcular mediana R²** por repeticion sobre genes de evaluacion.

4. **Seleccionar el orden final** mediante uno de dos metodos:

   - ``best_k``: elegir la repeticion con mayor mediana R².
   - ``aggregate``: agregar los K ordenes en un consenso circular.

Metodo aggregate (Barragan et al. 2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combina dos aproximaciones:

- **TSP (alpha3)**: construir una matriz de distancias asimetricas
  (CORAM) y resolver un problema del viajante con multiples heuristicas.
- **Hodge**: calcular la precedencia neta entre pares usando la teoria
  de Hodge sobre grafos.

El mejor candidato se refina con **CLM** (Circular Local Minimization).

Ejemplo de uso
--------------

.. code-block:: python

   from circust.robust_order import RobustOrderEstimator

   robust = RobustOrderEstimator(
       n_reps=5,
       method="best_k",       # o "aggregate"
       anchor_gene="ARNTL",
       seed=42,
   )
   result = robust.run(
       top_result=top_result,
       expr_full_norm=expr_norm,
       core_genes=core_genes,
   )

   result.sample_order       # orden final
   result.circular_scale     # escala circular final
   result.median_r2_per_rep  # R² de cada repeticion


Referencia API
--------------

Ver :doc:`/api/robust_order`.
