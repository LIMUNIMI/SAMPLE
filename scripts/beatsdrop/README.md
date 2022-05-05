# BeatsDROP
Scripts related to the paper _"Acoustic Beats and Where To Find Them: Theory of Uneven Beats and Applications to Modal Parameters Estimate"_.

We recommend installing in a virtual environment. For how to create virtual environments, please, refer to the official Python documentation for [`venv`](https://docs.python.org/3/library/venv.html) or [`conda`](https://docs.conda.io).

To install the script dependencies, you can either run
```
pip install .[scripts-beatsdrop]
```
or, to install all development dependencies
```
pip install -r requirements.txt
```
If you are using conda, you may have to add the `sample` source as a package in developer mode
```
conda develop .
```

## Evaluation
The script [`evaluation.py`](evaluation.py) runs the evaluation protocol for the BeatsDROP algorithm. To print the script options, run
```
python scripts/beatsdrop/evaluation.py --help
```

### Running
For running, we recommend
 - explicitly setting the number of parallel jobs with the `--n-jobs` option
 - enabling the progressbar with the `--tqdm` switch
 - saving results to file, specifying an output path with the `--output` option

For example
```
python scripts/beatsdrop/evaluation.py \
  -O evaluation/evaluation \
  --wav evaluation/dataset \
  --checkpoint 100 \
  --tqdm \
  -j 10
```
runs on 10 jobs and saves the following files
```
evaluation
├── dataset                    % Folder of synthesized WAV files
├── evaluation_rankresult.dat  % Pickled file for autorank result object
├── evaluation.csv             % CSV file with ground truth and results for each WAV file
└── evaluation_report.tex      % LaTeX report of statistical comparisons
```

Run time with 10 parallel jobs is less than 15 minutes on an Intel® Core™ i9-9820X CPU at 3.30GHz.
