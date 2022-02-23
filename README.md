# Spectral Analysis for Modal Parameter Linear Estimate
[![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/limunimi/sample/main/main?event=push)](https://github.com/limunimi/sample/actions?query=workflow%3Amain)
[![Coverage](https://gist.githubusercontent.com/chromaticisobar/fb3ce2e55493c80839ca8985d0c38146/raw/lim-sample-coverage-badge.svg)](https://github.com/limunimi/sample/actions?query=workflow%3Amain)
[![Pylint](https://gist.githubusercontent.com/chromaticisobar/fb3ce2e55493c80839ca8985d0c38146/raw/lim-sample-pylint-badge.svg)](https://github.com/limunimi/sample/actions?query=workflow%3Amain)
[![PyPI version](https://badge.fury.io/py/lim-sample.svg)](https://badge.fury.io/py/lim-sample)

## Install
You can install the `sample` package from [PyPI](https://pypi.org/project/lim-sample) via pip

```pip install lim-sample```

Available extras are
 - `plots`: for plotting utilities
 - `notebooks`: for running notebooks
 - `gui`: for running the GUI

## GUI
If you don't want write code to use SAMPLE,
you can use the graphical user interface

> Please, note that the GUI is still in its beta release phase

### Windows
For Windows, a stand-alone executable is available. You can download the
latest version from GitHub:

 - Go to https://github.com/limunimi/sample/releases
 - Download the zip file from the latest release (`SAMPLE_win_<version>.zip`)
 - Unzip the `SAMPLE.exe` file
 - That's it, you can run it!

### Python
You can install the GUI from the command line with Python via pip.  
It is recommended run these commands in a virtual environment in  
order to to keep your system clean

```pip install lim-sample[gui]==1.5.0b0```

To run the GUI from the command line, run

```python -m sample.gui```

## Documentation
API documentation can be found online here:

https://limunimi.github.io/SAMPLE/

## Source code
Source code is available on GitHub

https://github.com/limunimi/sample

### Notebooks
For learning to use the package, you can refer to the interactive
notebooks in the [notebooks](https://github.com/limunimi/sample/tree/master/notebooks) folder
