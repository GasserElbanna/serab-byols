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
    python_requires="==3.8.10",
    install_requires=[
        "librosa==0.8.1",
        "numba==0.48",
        "numpy==1.19.2",
        "torch",
        "torchaudio==0.9.0",
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