# Spectral Analysis for Modal Parameter Linear Estimate
[![DOI](https://zenodo.org/badge/342648141.svg)](https://zenodo.org/badge/latestdoi/342648141)
[![PyPI version](https://badge.fury.io/py/lim-sample.svg)](https://badge.fury.io/py/lim-sample)
[![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/limunimi/sample/main.yml?branch=main)](https://github.com/limunimi/sample/actions?query=workflow%3Amain)
[![Coverage](https://gist.githubusercontent.com/chromaticisobar/fb3ce2e55493c80839ca8985d0c38146/raw/lim-sample-coverage-badge.svg)](https://github.com/limunimi/sample/actions?query=workflow%3Amain)
[![Pylint](https://gist.githubusercontent.com/chromaticisobar/fb3ce2e55493c80839ca8985d0c38146/raw/lim-sample-pylint-badge.svg)](https://github.com/limunimi/sample/actions?query=workflow%3Amain)

Python package with tools for spectral analysis and modal parameters estimate

## Table of Contents
- [Spectral Analysis for Modal Parameter Linear Estimate](#spectral-analysis-for-modal-parameter-linear-estimate)
  - [Table of Contents](#table-of-contents)
  - [Install](#install)
  - [GUI](#gui)
    - [Windows](#windows)
    - [Python](#python)
  - [Documentation](#documentation)
  - [Source Code](#source-code)
    - [Notebooks](#notebooks)
    - [Scripts](#scripts)
  - [References](#references)

## Install
We recommend installing in a virtual environment. For how to create virtual environments, please, refer to the official documentation for [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io).

You can install the `sample` package from [PyPI](https://pypi.org/project/lim-sample) via pip.
```
pip install lim-sample
```

Available extras are
 - `plots`: for plotting utilities
 - `notebooks`: for running notebooks
 - `gui`: for running the GUI

## GUI
If you don't want write code to use SAMPLE,
you can use the graphical user interface

### Windows
For Windows, a stand-alone executable is available. You can download the
latest version from GitHub:

 - Go to https://github.com/limunimi/sample/releases
 - Download the zip file from the latest release (`SAMPLE_win_<version>.zip`)
 - Unzip the `SAMPLE.exe` file
 - That's it, you can run it!

### Python
You can install the GUI from the command line with Python via pip.  
We recommend to [install in a virtual environment](#install)

```
pip install lim-sample[gui]
```

To run the GUI from the command line, run

```
python -m sample.gui
```

## Documentation
API documentation can be found online here:

https://limunimi.github.io/SAMPLE

## Source Code
Source code is available on GitHub

https://github.com/limunimi/sample

### Notebooks
For learning to use the package, you can refer to the interactive
notebooks in the [notebooks](https://github.com/limunimi/sample/blob/main/notebooks) folder

### Scripts
In the [scripts](https://github.com/limunimi/sample/blob/main/scripts) folder, there are Python scripts for the reproducibility of experiments

## References
References are available both as a [BibTeX](https://github.com/limunimi/sample/blob/main/SAMPLE.bib) and a [CITATION.cff](https://github.com/limunimi/sample/blob/main/CITATION.cff) file.

If you use this software in your research, please, consider citing the following items
 - The SMC 2020 paper ["Spectral Analysis for Modal Parameters Linear Estimate"](https://doi.org/10.5281/zenodo.3898795)
 - The [SAMPLE](https://doi.org/10.5281/zenodo.6536419) package for Python
