Etapa 5 — Ajuste de modelos
============================

Que problema resuelve
---------------------

Una vez que tenemos el orden circular definitivo, necesitamos **cuantificar
la ritmicidad** de cada gen: amplitud, fase (hora del pico), bondad de
ajuste, etc. Para ello ajustamos tres modelos complementarios.

Los tres modelos
----------------

FMM (Frequency Modulated Model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modelo parametrico de 5 parametros (M, A, alpha, beta, omega) que captura
formas de onda asimetricas. Es el modelo principal del pipeline.

Cosinor
^^^^^^^

Modelo parametrico de 3 parametros (M, A, phi): una sinusoide pura.
Mas restrictivo que FMM pero util como referencia y para comparar.

ORI (Isotonic Regression Circular)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ajuste no parametrico que no asume forma funcional: solo exige que la
curva sea unimodal (un pico y un valle). Util cuando la forma de onda
real no encaja en FMM ni Cosinor.

Tabla de 25 estadisticos
-------------------------

Para cada gen, el pipeline produce una fila con 25 columnas:

.. code-block:: text

   FMM:    M, A, alpha, beta, omega, pkU, pkL, %pkU, %pkL, s², MSE, R²
   Cosinor: M, A, phi, pk1, pk2, %pk1, %pk2, s², MSE, R²
   NP:     s², MSE, R²

Referencia API
--------------

Ver :doc:`/api/fitting`.
