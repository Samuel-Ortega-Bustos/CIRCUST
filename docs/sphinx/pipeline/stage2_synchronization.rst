Etapa 2 — Sincronizacion biologica
====================================

Que problema resuelve
---------------------

El orden que da CPCA es geometricamente correcto pero **biologicamente
arbitrario**: no sabemos donde empieza el "dia" ni en que direccion avanza
el reloj. La sincronizacion ancla el orden a la biologia real.

Como funciona
-------------

Dos ajustes:

1. **Rotacion**: ajustar FMM a los genes core y rotar la escala circular
   para que el pico de ARNTL caiga en una posicion fija (convencion del
   campo).

2. **Orientacion**: determinar si la escala avanza en la direccion
   biologicamente correcta (dia → noche). Si no, invertir. Se usa
   DBP como gen de referencia direccional.

Ejemplo de uso
--------------

.. code-block:: python

   from circust.synchronizer import CircularSynchronizer

   sync = CircularSynchronizer(
       anchor_gene="ARNTL",
       direction_gene="DBP",
   )
   sync_result = sync.run(cpca_result, core_genes)


Referencia API
--------------

Ver :doc:`/api/synchronizer`.
