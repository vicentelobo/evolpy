#!/usr/bin/env python

##############################################################################
##  evolpy: Python library for minimization through evolutionary algorithms
##
##  Written by Vicente Lobo (vicenteclobo@gmail.com) 
##############################################################################


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EvolPy",
    version="0.0.1",
    author="Vicente Lobo",
    author_email="vicenteclobo@gmail.com",
    description="Python library for minimization through evolutionary algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vicentelobo/evolpy",
    packages=['evolpy'],
    install_requires=['numpy>=1.17.4', 'tqdm>=4.48.0'],
    python_requires='>=3.6',
)