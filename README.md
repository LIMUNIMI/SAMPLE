# Spectral Analysis for Modal Parameter Linear Estimate
[![DOI](https://zenodo.org/badge/342648141.svg)](https://zenodo.org/badge/latestdoi/342648141)
[![PyPI version](https://badge.fury.io/py/lim-sample.svg)](https://badge.fury.io/py/lim-sample)
[![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/limunimi/sample/main/main?event=push)](https://github.com/limunimi/sample/actions?query=workflow%3Amain)
[![Coverage](https://gist.githubusercontent.com/chromaticisobar/fb3ce2e55493c80839ca8985d0c38146/raw/lim-sample-coverage-badge.svg)](https://github.com/limunimi/sample/actions?query=workflow%3Amain)
[![Pylint](https://gist.githubusercontent.com/chromaticisobar/fb3ce2e55493c80839ca8985d0c38146/raw/lim-sample-pylint-badge.svg)](https://github.com/limunimi/sample/actions?query=workflow%3Amain)

Python package with tools for spectral analysis and modal parameters estimate

## Install
You can install the `sample` package from [PyPI](https://pypi.org/project/lim-sample) via pip

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
We recommend to install in a [virtual environment](https://docs.python.org/3/library/venv.html), in order to to keep your system clean

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

## Source code
Source code is available on GitHub

https://github.com/limunimi/sample

### Notebooks
For learning to use the package, you can refer to the interactive
notebooks in the [notebooks](https://github.com/limunimi/sample/tree/master/notebooks) folder

## References
If you use this software in your research, please, consider referencing the following items
 - The SMC 2020 paper [_"Spectral Analysis for Modal Parameters Linear Estimate"_](https://doi.org/10.5281/zenodo.3898795)
```
@inproceedings{tiraboschi2020spectral,
  author       = {Tiraboschi, Marco and Avanzini, Federico and Ntalampiras, Stavros},
  doi          = {10.5281/zenodo.3898795},
  month        = {6},
  pages        = {276--283},
  title        = {{Spectral Analysis for Modal Parameters Linear Estimate}},
  address      = {Torino, Italy},
  editor       = {Simone Spagnol and Andrea Valle},
  organization = {Sound and Music Computing Network},
  publisher    = {Axea sas/SMC Network},
  booktitle    = {Proceedings of the 17th Sound and Music Computing Conference},
  year         = {2020}
}
```
 - The [SAMPLE](https://doi.org/10.5281/zenodo.6536419) package for Python
```
@software{tiraboschi2021sample,
  author       = {Tiraboschi, Marco},
  title        = {SAMPLE -- Python package},
  year         = {2021},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6536419},
  url          = {https://doi.org/10.5281/zenodo.6536485},
  organization = {LIM, University of Milan}
}
```
