# BeatsDROP
Scripts related to the paper _"Acoustic Beats and Where To Find Them: Theory of Uneven Beats and Applications to Modal Parameters Estimate"_

## Evaluation
The script [`evaluation.py`](evaluation.py) runs the evaluation protocol for the BeatsDROP algorithm. To print the script options, run
```
python scripts/beatsdrop/evaluation.py --help
```
```
usage: evaluation.py [-h] [-O PATH] [-l LEVEL] [-j N] [-n N] [--alpha P]
                     [--frequentist] [--n-samples N_SAMPLES] [--tqdm]
                     [--no-resume] [--checkpoint PERIOD] [--wav PATH]
                     [--log-exception PATH] [--install]

Evaluation script for BeatsDROP

options:
  -h, --help            show this help message and exit
  -O PATH, --output PATH
                        Output base path for results
  -l LEVEL, --log-level LEVEL
                        Set the log level. Default is 'INFO'
  -j N, --n-jobs N      The number of worker processes to use. By default, the
                        number returned by os.cpu_count() is used.
  -n N, --n-cases N     The number of tests to perform
  --alpha P             The threshold for statistical significance
  --frequentist         Perform frequentist tests (instead of Bayesian)
  --n-samples N_SAMPLES
                        Number of samples used to estimate the posterior
                        probabilities with the Bayesian signed rank test
  --tqdm                Use tqdm progressbar
  --no-resume           Do not load previous results, but recompute everything
  --checkpoint PERIOD   Period for saving checkpoints (the number of tests to
                        do before saving another checkpoint)
  --wav PATH            Folder for writing wav files for test cases
  --log-exception PATH  Folder for writing logs for test failure, instead of
                        raising an exception
  --install             Install dependencies (no experiment will be run)
```
### Dependencies
We recommend installing in a virtual environment. For how to create virtual environments, please, refer to the official Python documentation for [`venv`](https://docs.python.org/3/library/venv.html).

To install the script dependencies, including the `sample` package, you can run
```
python scripts/beatsdrop/evaluation.py --install
```

### Running
For running, we recommend
 - explicitly setting the number of parallel jobs with the `--n-jobs` option
 - enabling the progressbar with the `--tqdm` switch
 - saving results to file, specifying an output path with the `--output` option

For example
```
python scripts/beatsdrop/evaluation.py -O evaluation/evaluation -j 10 --tqdm --checkpoint 100 --wav evaluation/dataset
```
runs on 10 jobs and saves the following files
```
evaluation
├── dataset                    % Folder of synthesized WAV files
├── evaluation_rankresult.dat  % Pickled file for autorank result object
├── evaluation.csv             % CSV file with ground truth and results for each test
└── evaluation_report.tex      % LaTeX report of statistical comparisons
```

Run time with 10 parallel jobs is less than 15 minutes on an Intel® Core™ i9-9820X CPU at 3.30GHz.
