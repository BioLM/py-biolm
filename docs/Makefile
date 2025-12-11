# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = biolmai
SOURCEDIR     = .
BUILDDIR      = _build

# Pass through IFRAME_MODE environment variable if set
ifdef IFRAME_MODE
export IFRAME_MODE
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

rebuild: ## rebuild HTML docs (routine updates)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean-html: ## clean HTML build directory
	rm -rf $(BUILDDIR)/html

rebuild-clean: clean-html rebuild ## clean and rebuild HTML docs

.PHONY: help Makefile rebuild clean-html rebuild-clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
