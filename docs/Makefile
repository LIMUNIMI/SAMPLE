# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PROJECTDIR    = ../sample
IGNOREPATHS   = ../sample/widgets ../sample/gui.py ../sample/vid.py
INDICES       = index.rst changelog.rst

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean rst Makefile

%: Makefile rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean: clean_rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

rst:
	sphinx-apidoc -e -T -M -o "$(SOURCEDIR)" "$(PROJECTDIR)" $(IGNOREPATHS) -d 1

clean_rst:
	mkdir -p .tmp
	for i in $(INDICES) ; do \
		mv $(SOURCEDIR)/$$i .tmp; \
	done
	rm -f $(SOURCEDIR)/*.rst
	mv .tmp/* $(SOURCEDIR);
	rm -rf .tmp

build:
	mkdir $@