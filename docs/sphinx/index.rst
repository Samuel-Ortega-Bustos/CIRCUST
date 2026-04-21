CIRCUST - Documentacion
=======================

**CIRCUST** (CIRCular-robUST) es un pipeline robusto para la reconstrucción del
orden temporal de ritmos moleculares a partir de datos ruidosos y donde se 
desconocen los momentos exactos del muestreo.

.. note::

   Esta documentacion esta en desarrollo activo, es posible que se lleven a cabo cambios con el paso del tiempo y pueden tardar en verse
   reflejado en esta documentacion.


.. toctree::
   :maxdepth: 2
   :caption: Guia del Pipeline

   pipeline/overview
   pipeline/stage0_preprocessing
   pipeline/stage0.1_core_genes
   pipeline/stage1_cpca
   pipeline/stage2_synchronization
   pipeline/stage3_topgenes
   pipeline/stage4_robust_order
   pipeline/stage5_fitting


.. toctree::
   :maxdepth: 2
   :caption: Referencia API

   api/preprocessing
   api/core_genes
   api/cpca
   api/synchronizer
   api/top_genes
   api/robust_order
   api/fitting
   api/visualization
