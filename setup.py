import setuptools
from sample import __version__ as version


with open("README.md", "r") as f:
  readme = f.read()


setuptools.setup(
  name="sample",
  version=version,
  author="Marco Tiraboschi",
  author_email="marco.tiraboschi@unimi.it",
  maintainer="Marco Tiraboschi",
  maintainer_email="marco.tiraboschi@unimi.it",
  description="Package for the SAMPLE method",
  long_description=readme,
  long_description_content_type="text/markdown",
  url="https://github.com/ChromaticIsobar/SAMPLE",
  packages=setuptools.find_packages(),
  include_package_data=True,
  setup_requires=[
    "wheel",
  ],
  install_requires=[
    "numpy",
    "scipy",
    "scikit-learn",
  ],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
)
