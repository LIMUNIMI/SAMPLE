# SAMPLE: a Python Package for the Spectral Analysis of Modal Sounds
Scripts related to the CIM '22 paper _"SAMPLE: a Python Package for the Spectral Analysis of Modal Sounds"_.

We recommend [installing in a virtual environment](../../README.md#install).
To install the script dependencies, you can either run
```
pip install .[scripts-cim22]
```
or, to install all development dependencies
```
pip install -r requirements.txt
```
If you are using conda, you may have to add the `sample` source as a package in developer mode
```
conda develop .
```

## Performance
The script [`performance.py`](performance.py) runs the benchmarking script for the cochleagram implementation based on strided convolution. To print the script options, run
```
python scripts/cim22/performance.py --help
```

We recommend saving results to file, specifying an output path with the `--output` option. This benchmarking evaluates numerous cases and can take several hours.

## Autotuning
The script [`autotuning.py`](autotuning.py) produces the plot the automatic optimization figure. To print the script options, run
```
python scripts/cim22/autotuning.py --help
```

We recommend specifying the `--tqdm` option for a progressbar. This script will take one or two minutes with default arguments.
