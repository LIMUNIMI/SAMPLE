"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html"""
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "Spectral Analysis for Modal Parameter Linear Estimate"
copyright = "2021-2022, Marco Tiraboschi"  # pylint: disable=W0622
author = "Marco Tiraboschi"

SAMPLE_SPHINX_VERSION_IS_SHA = os.environ.get("SAMPLE_SPHINX_VERSION_IS_SHA",
                                              "")
if not SAMPLE_SPHINX_VERSION_IS_SHA:
  # Get version from package
  version = importlib.import_module("sample").__version__
elif len(SAMPLE_SPHINX_VERSION_IS_SHA) > 1:
  # Get version from provided git sha
  version = SAMPLE_SPHINX_VERSION_IS_SHA
else:
  # Get version from git sha
  version = subprocess.check_output("git rev-parse HEAD",
                                    shell=True).decode("utf-8")
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "m2r2",
]

source_suffix = [".rst", ".md"]

napoleon_google_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_show_sourcelink = False
