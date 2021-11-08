#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="serab_byols",
    description="Data-driven Audio Representation 2021",
    author="Logitech",
    author_email="",
    url="",
    license="MIT License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "",
        "Source Code": "",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=[
        "librosa==0.8.1",
        "numba==0.48",
        "numpy==1.19.2",
        "torch",
        "torchaudio==0.9.0",
        "tqdm==4.61.1",
    ],
)