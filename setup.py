"""Setup script for PyPI package"""
import setuptools

with open("README.md", mode="r", encoding="utf-8") as f:
  readme = f.read()
with open("sample/__init__.py", mode="r", encoding="utf-8") as f:
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
        "paragraph",
        "scikit-learn",
        "scikit-optimize",
        "scipy",
        "tqdm",
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
            "ipython",
            "jupyter",
            "matplotlib",
            "librosa",
            "more-itertools",
            "requests",
        ],
        "scripts-beatsdrop": [
            "matplotlib",
            "autorank",
            "pandas",
        ],
        "scripts-cim22": ["matplotlib",],
        "test": [
            "ipython",
            "cython",
            "more-itertools",
        ],
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
            "m2r2",
            "recommonmark",
        ],
        "style": [
            "pylint",
            "yapf",
        ],
        "cov": ["coverage",],
        "packaging": ["twine",],
        "installer": ["pyinstaller<5.1",]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
