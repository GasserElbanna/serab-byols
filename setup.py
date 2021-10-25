#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="serab_byols",
    description="Holistic Evaluation of Audio Representations (HEAR) 2021",
    author="Logitech",
    author_email="",
    url="",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "",
        "Source Code": "",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires="=3.8",
    install_requires=[
        "librosa==0.8.1",
        # otherwise librosa breaks
        "numba==0.48",
        # tf 2.6.0
        "numpy==1.19.2",
        "torchaudio==0.9.0",
        "torch",
        # otherwise librosa breaks
        "numba==0.48",
    ],
    # extras_require={
    #     "test": [
    #         "pytest",
    #         "pytest-cov",
    #         "pytest-env",
    #     ],
    #     "dev": [
    #         "pre-commit",
    #         "black",  # Used in pre-commit hooks
    #         "pytest",
    #         "pytest-cov",
    #         "pytest-env",
    #     ],
    # },
)