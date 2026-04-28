import os
import sys

# -- Ruta al codigo fuente --------------------------------------------------
# Sphinx necesita poder importar tus modulos para leer los docstrings.
# Le decimos donde esta la carpeta 'src/' de CIRCUST.
sys.path.insert(0, os.path.abspath("../../src"))

# -- Informacion del proyecto ------------------------------------------------
project = "CIRCUST"
author = "Samuel Ortega Bustos"
release = "0.1.0"

# -- Extensiones -------------------------------------------------------------
# Cada extension anade funcionalidad a Sphinx:
extensions = [
    "sphinx.ext.autodoc",          # Lee docstrings del codigo y genera docs
    "sphinx.ext.napoleon",         # Soporta docstrings estilo NumPy/Google
    "sphinx.ext.viewcode",         # Anade enlace "[source]" al codigo fuente
    "sphinx_autodoc_typehints",    # Muestra los type hints bonitos
]

# -- Opciones de autodoc ----------------------------------------------------
autodoc_member_order = "bysource"  # Ordenar miembros como aparecen en el .py
autodoc_typehints = "description"  # Type hints en la descripcion, no en la firma

# -- Opciones de Napoleon ----------------------------------------------------
napoleon_use_ivar = True           # Usar :ivar: para atributos (evita duplicados
                                   # con campos de dataclass documentados por autodoc)

# -- Tema visual -------------------------------------------------------------
html_theme = "furo"

# -- Opciones generales ------------------------------------------------------
exclude_patterns = ["_build"]
