Etapa 0.1 — Seleccion de genes semilla
========================================

Que problema resuelve este modulo
---------------------------------

Antes de ejecutar CPCA, necesitamos definir un conjunto de **genes semilla**
(core clock genes) que sabemos que son ritmicos. Estos genes sirven como
referencia para la proyeccion circular inicial. Para ello usamos el fichero
``core_genes.py`` que se encarga de la seleccion, validacion de la presencia de estos 
genes en nuestra matriz de expresion.

CIRCUST incluye un catalogo predefinido de 12 genes reloj centrales:
ARNTL, CLOCK, CRY1, CRY2, CSNK1D, CSNK1E, DBP, NPAS2, NR1D1, NR1D2,
PER1, PER2, PER3, TEF.

.. note::
    Existen otros conjuntos disponibles como el de Zhang et al. 2014 que recoge los siguientes:
    ARNTL, DBP, NR1D1, NR1D2, PER1, PER2, PER3, USP2, TSC22D3, TSPAN4
    Ademas de en un futuro tener un modelo de seleccion automatica


Ejemplo de uso
--------------
.. code-block:: python
    
    from circust.core_genes import CoreGeneSelector
    from circust.preprocessing import Preprocessor, load_expression_matrix 

    matrix = load_expression_matrix("data/raw/gtex_brain.csv")
    prep   = Preprocessor().run(matrix) 

    # Preset por nombre (circust,zhang...)
    sel    = CoreGeneSelector(preset="circust")
    result = sel.select(prep.expr_norm)

    result.genes          # lista validada
    result.missing        # genes no encontrados en la matriz   

    # Lista personalizada
    sel    = CoreGeneSelector(custom_genes=["ARNTL", "DBP", "PER1", "PER2"])
    result = sel.select(prep.expr_norm)


Referencia de la API
--------------------

Ver :doc:`/api/core_genes`.
