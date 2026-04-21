Vision general del pipeline
============================

CIRCUST estima el **orden circular** (hora interna del reloj biologico)
de muestras de expresion genica. El pipeline se compone de 6(7) etapas
secuenciales:

.. code-block:: text

   Etapa 0 ── Preprocessing
   │  Carga, filtrado y normalizacion de la matriz de expresion.
   │
   v
   Etapa 0.1 ── Selección de genes semilla
   │  En este momento filtramos por una serie de genes semilla que se consideran gold-standard
   │
   v
   Etapa 1 ── CPCA (Circular PCA)
   │  Orden circular preliminar mediante PCA + proyeccion al circulo.
   │  Deteccion y eliminacion de outliers mediante ajustes FMM.
   │
   v
   Etapa 2 ── Sincronizacion biologica
   │  Rotacion (anclar ARNTL al pico) y orientacion (dia/noche).
   │
   v
   Etapa 3 ── Seleccion de genes TOP
   │  Seleccionar los genes mas ritmicos por ajuste FMM/Cosinor.
   │  Clasificacion en 8 sectores del circulo.
   │
   v
   Etapa 4 ── Orden robusto
   │  K repeticiones con subconjuntos aleatorios.
   │  Seleccion (best_k) o agregacion circular (TSP + Hodge + CLM).
   │
   v
   Etapa 5 ── Ajuste final
      Modelos FMM, Cosinor y no parametrico sobre el orden definitivo.
      Tabla de 25 estadisticos por gen.


Cada etapa consume la salida de la anterior y produce un objeto ``Result``
con campos con nombre.

Modulos del paquete
-------------------

========================================  =========================================
Modulo                                    Etapa
========================================  =========================================
``circust.preprocessing``                 Etapa 0 — Carga y limpieza
``circust.core_genes``                    Etapa 0.1 — Extraccion de genes core
``circust.cpca``                          Etapa 1 — CPCA
``circust.synchronizer``                  Etapa 2 — Sincronizacion
``circust.core_genes``                    Listas de genes reloj centrales
``circust.top_genes``                     Etapa 3 — Genes TOP
``circust.robust_order``                  Etapa 4 — Orden robusto
``circust.fitting.fmm``                   Etapa 5 — Modelo FMM
``circust.fitting.cosinor``               Etapa 5 — Modelo Cosinor
``circust.fitting.ori``                   Etapa 5 — Regresion isotonica circular
``circust.visualization.*``               Graficos de cada etapa
========================================  =========================================
