# Makefile para Sphinx
# Uso:
#   make html     <- genera el sitio web en _build/html/
#   make clean    <- borra todo lo generado

SPHINXBUILD = sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)

.PHONY: html clean
