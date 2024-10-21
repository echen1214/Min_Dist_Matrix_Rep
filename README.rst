======================================================
Minimum Distance Matrix Representation (MDMR) Platform
======================================================

.. image:: https://zenodo.org/badge/313663037.svg
  :target: https://doi.org/10.5281/zenodo.13964938

.. .. image:: https://img.shields.io/pypi/v/dist_analy.svg
..         :target: https://pypi.python.org/pypi/dist_analy

.. .. image:: https://img.shields.io/travis/echen1214/dist_analy.svg
..         :target: https://travis-ci.com/echen1214/dist_analy

.. .. image:: https://readthedocs.org/projects/dist-analy/badge/?version=latest
..         :target: https://dist-analy.readthedocs.io/en/latest/?badge=latest
..         :alt: Documentation Status

.. .. image:: https://pyup.io/repos/github/echen1214/dist_analy/shield.svg
..      :target: https://pyup.io/repos/github/echen1214/dist_analy/
..      :alt: Updates

The Minimum Distance Matrix Representation (MDMR) platform for use in within Jupyter notebooks \
calculates distance matrices for a set of structures and run analyses that \
classify the protein conformational and ligand binding mode ensembles. \
Further analyses identifies important intra-protein interactions. \
This is the accompanying GitHub for the paper \
**Can Deep Learning Blind Docking Methods be used to Predict Allosteric Compounds?**

* Free software: MIT license

Features
--------
* Calculates minimum distance matrices on receptor conformations or ligand binding modes
* Runs PCA and HDBSCAN on distance matrices
* Calculates nSMD and accompanying histograms
* Uses altair for interactive plots and py3Dmol for visualizing structures \
  within the notebooks

Installation
------------

Download and install a `ClustalOmega <http://www.clustal.org/omega/>`_ binary into a local directory and symlink the binary to the command line.
For example Mac users can do ::

  wget http://www.clustal.org/omega/clustal-omega-1.2.3-macosx
  mv clustal-omega-1.2.3-macosx clustalo
  sudo chmod u+x clustalo
  sudo ln -s /{path}/{to}/clustalo /usr/local/bin/
  ## this will be different with zsh shell

Similarly, a Windows user can do ::

  wget http://www.clustal.org/omega/clustal-omega-1.2.2-win64.zip -outfile clustal.zip
  tar -xf clustal.zip
  cd clustal-omega-1.2.2-win64
  mklink clustalo .\clustalo.exe
  set PATH=%PATH%;%cd%

Create conda environment for this package ::

  conda create --name dist_analy python=3.9
  conda activate dist_analy
  conda install mdtraj -c conda-forge
  git clone https://github.com/echen1214/dist_analy
  cd dist_analy
  pip install -e .
  conda deactivate

This package is intended to be used with `Jupyter notebooks <https://jupyter.org/install>`_. You can use the conda environment you just created within the Jupyter notebook ::

  pip install jupyterlab
  pip install notebook
  conda install -c conda-forge  nb_conda_kernels

Tutorial
--------

You can run ``tutorial/Tutorial1_CDK2.ipynb`` with the data provided straight from the GitHub repository. This tutorial walks you \
through the downloading, processing, and calculating the distance matrices of CDK2 structures. Then you perform downstream analyses \
with PCA and HDBSCAN of the receptor-only and ligand-only plots, and nSMD calculations and histograms. 
 
The self- and cross-docking benchmarking analyses can be found ``tutorial/Tutorial2_self_cross_docking.ipynb``. To run the analyses \
first download and untar the poses ``zenodo_tar.tar.gz`` from the `Zenodo <https://doi.org/10.5281/zenodo.13964938>`_ database. The tutorial then performs the calculations of the main self- \
and cross-docking results of the paper.

Requirements
------------
::

  python
  numpy
  scikit-learn
  matplotlib
  biopython
  prody
  py3Dmol
  ipykernel
  altair
  rcsbsearchapi
  anytree
  pypdb
  spyrmsd
  meeko
  AlphaSpace2
  mdtraj

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
