import setuptools


with open("README.md", "r") as f:
  readme = f.read()
with open("sample/__init__.py", "r") as f:
  version = f.read().split("__version__ = \"", 1)[-1].split("\"", 1)[0]


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
  url="https://github.com/limunimi/sample",
  packages=setuptools.find_packages(include=["sample", "sample.*"]),
  include_package_data=True,
  setup_requires=[
    "wheel",
  ],
  install_requires=[
    "chromatictools",
    "numpy",
    "scipy",
    "scikit-learn",
  ],
  extras_require={
    "gui": [
      "ttkthemes",
      "librosa",
      "pygame",
      "matplotlib",
      "Pillow",
      "throttle",
    ],
    "plots": [
      "matplotlib",
      "Pillow",
    ],
    "notebooks": [
      "jupyter",
      "matplotlib",
      "librosa",
      "more-itertools",
      "requests",
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
    ],
    "lint": [
      "pylint",
    ],
    "cov": [
      "coverage",
    ],
    "packaging": [
      "twine",
    ],
    "installer": [
      "pyinstaller",
    ]
  },
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires=">=3.6",
)
