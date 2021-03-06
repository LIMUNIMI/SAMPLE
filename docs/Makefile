# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PROJECTDIR    = ../sample
INDICES       = index.rst changelog.rst
PYLINTRC      = https://google.github.io/styleguide/pylintrc

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean pylint rst Makefile

%: Makefile rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean: clean_rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -f pylintrc

rst:
	sphinx-apidoc -e -T -M -o "$(SOURCEDIR)" "$(PROJECTDIR)"

clean_rst:
	mkdir -p .tmp
	for i in $(INDICES) ; do \
		mv $(SOURCEDIR)/$$i .tmp; \
	done
	rm -f $(SOURCEDIR)/*.rst
	mv .tmp/* $(SOURCEDIR);
	rm -rf .tmp

pylint: pylintrc
	pylint $(PROJECTDIR) -r y --exit-zero --jobs 1 --persistent=n

build:
	mkdir $@

pylintrc:
ifeq (,$(wildcard $(PYLINTRC)))
	wget $(PYLINTRC) -O $@
endif
