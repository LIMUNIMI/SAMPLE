import setuptools
from sample import __version__ as version


with open("README.md", "r") as f:
  readme = f.read()


setuptools.setup(
  name="lim-sample",
  version=version,
  author="Marco Tiraboschi",
  author_email="marco.tiraboschi@unimi.it",
  maintainer="Marco Tiraboschi",
  maintainer_email="marco.tiraboschi@unimi.it",
  description="Package for the SAMPLE method",
  long_description=readme,
  long_description_content_type="text/markdown",
  url="https://github.com/LIMUNIMI/SAMPLE",
  packages=setuptools.find_packages(include=["sample", "sample.*"]),
  include_package_data=True,
  setup_requires=[
    "wheel",
  ],
  install_requires=[
    "numpy",
    "scipy",
    "scikit-learn",
  ],
  extras_require={
    "plots": [
      "matplotlib",
    ],
    "notebooks": [
      "jupyter",
      "matplotlib",
      "librosa",
      "more-itertools",
    ],
    "test": [
      "cython",
      "more-itertools",
    ],
    "docs": [
      "sphinx",
      "sphinx_rtd_theme",
      "m2r2",
      "recommonmark",
      "matplotlib",
    ],
    "codecheck": [
      "pylint",
    ],
    "packaging": [
      "twine",
    ]
  },
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
)
