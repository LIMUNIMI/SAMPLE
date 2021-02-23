# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PROJECTDIR    = ..
INDICES       = index.rst pylint.rst doctest.rst
LINTTARGET    = "$(PROJECTDIR)/sample"
PYLINTRC      = https://google.github.io/styleguide/pylintrc
PYLINTREPORT  = "$(SOURCEDIR)/pylint_report.rst"
DOCTESTREPORT = "$(SOURCEDIR)/doctest_report.rst"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean_all pylint all_rst

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile all_rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doctest: Makefile rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean: clean_rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

all_rst: rst $(PYLINTREPORT) $(DOCTESTREPORT)

rst:
	sphinx-apidoc -T -M -o "$(SOURCEDIR)" "$(PROJECTDIR)"

$(DOCTESTREPORT): doctest
	mv build/doctest/output.txt $@

clean_rst:
	mkdir -p .tmp
	for i in $(INDICES) ; do \
		mv $(SOURCEDIR)/$$i .tmp; \
	done
	rm -f $(SOURCEDIR)/*.rst
	mv .tmp/* $(SOURCEDIR);
	rm -rf .tmp

pylint: $(PYLINTREPORT)

$(PYLINTREPORT): pylintrc build
	pylint $(LINTTARGET) -r y --exit-zero --jobs 1 --persistent=n > $@

build:
	mkdir $@

pylintrc:
	wget $(PYLINTRC) -O $@
