# CIRCUST - Python Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1371%2Fjournal.pcbi.1011510-blue)](https://doi.org/10.1371/journal.pcbi.1011510)

**Autores de la metodología original (R):** Yolanda Larriba, Ivy C. Mason, Richa Saxena, Frank A.J.L. Scheer, Cristina Rueda  
**Implementación en Python:** [Tu Nombre] (TFG supervisado por Yolanda Larriba)

Este repositorio contiene una implementación en Python de la metodología **CIRCUST** (CIRCular-robUST), originalmente desarrollada en R por Yolanda Larriba y colaboradores (Larriba et al., 2023) para la reconstrucción del orden temporal de ritmos moleculares y la construcción de atlas de expresión génica circadiana en humanos.

## 🌐 Contexto

Este proyecto es un *fork* del [repositorio original en R](https://github.com/yolandalago/CIRCUST/) y forma parte de un Trabajo de Fin de Grado (TFG) supervisado por la autora principal. El objetivo es traducir y adaptar el flujo de trabajo a Python, aprovechando las ventajas de este ecosistema para el procesamiento de datos genómicos a gran escala, manteniendo la fidelidad al método original.

## 📄 Descripción

CIRCUST aborda dos problemas clave en cronobiología molecular a partir de datos de expresión génica con tiempos de muestreo desconocidos o imprecisos (por ejemplo, muestras post-mortem):

1. **Reconstrucción del orden temporal** de las muestras utilizando Análisis de Componentes Principales Circular (CPCA).
2. **Estimación robusta de ritmicidad** mediante el modelo paramétrico FMM (Frequency Modulated Möbius), que permite capturar patrones asimétricos no sinusoidales.

La versión original en R fue validada en múltiples conjuntos de datos (epidermis humana, biopsias de músculo, cerebro post-mortem, tejidos de babuino) y aplicada a 34 tejidos del proyecto GTEx, generando el atlas de ritmos diarios más completo hasta la fecha.

Esta implementación en Python sigue los pasos descritos en el artículo original:

- Preprocesamiento (filtrado de genes con baja expresión, normalización min-max).
- Detección de outliers mediante CPCA y residuos FMM.
- Ordenación preliminar usando genes semilla (12 genes centrales del reloj circadiano).
- Selección de genes TOP (altamente rítmicos y con fases distribuidas) y remuestreo para obtener órdenes robustos.
- Ajuste del modelo FMM y cálculo de parámetros (amplitud, fase pico, nitidez).
- Generación de atlas por tejido.

## 🔁 Relación con el repositorio original

- **Repositorio original (R)**: [yolandalago/CIRCUST](https://github.com/yolandalago/CIRCUST/)
- **Artículo de referencia**: [PLOS Comput Biol 19(9): e1011510](https://doi.org/10.1371/journal.pcbi.1011510)
- **Este repositorio**: Traducción a Python, manteniendo la lógica y los pasos del método original, pero con adaptaciones propias del entorno Python (uso de `pandas`, `numpy`, `scipy`, `matplotlib`, y `scikit-learn` para PCA).

Se ha partido de los archivos de datos y resultados generados por la autora (en formato RData) para verificar la corrección de la implementación.

## 🚀 Características

- Implementación modular (cada paso del método en un módulo separado).
- Soporte para leer datos desde archivos Parquet (recomendado) o directamente desde RData vía `pyreadr`.
- Visualización de resultados (fases pico, patrones de expresión, comparación con tiempos reales).
- Generación de tablas resumen similares a las del artículo (genes TOP por tejido con sus parámetros FMM).
- Validación reproducible mediante notebooks que replican las figuras del paper.

## 📦 Estructura del repositorio
```bash
circust/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt # Dependencias Python
├── setup.py # Instalación como paquete
├── src/ # Paquete principal
│ ├── __init__.py
│ ├── preprocessing.py
│ ├── cpca.py
│ ├── outlier_detection.py
│ ├── order_estimation.py
│ ├── fmm.py
│ ├── top_genes.py
│ ├── utils.py
│ └── constants.py
├── tests/ # Pruebas unitarias
│ └── test_*.py
├── notebooks/ # Jupyter notebooks con ejemplos y validación
│ ├── 
├── scripts/ # Scripts para ejecutar análisis completos
│ └── run_circust.py
└── data/ # Carpeta para datos (ignorada por git)
```