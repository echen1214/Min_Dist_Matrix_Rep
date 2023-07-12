==========
dist_analy
==========


.. image:: https://img.shields.io/pypi/v/dist_analy.svg
        :target: https://pypi.python.org/pypi/dist_analy

.. image:: https://img.shields.io/travis/echen1214/dist_analy.svg
        :target: https://travis-ci.com/echen1214/dist_analy

.. image:: https://readthedocs.org/projects/dist-analy/badge/?version=latest
        :target: https://dist-analy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/echen1214/dist_analy/shield.svg
     :target: https://pyup.io/repos/github/echen1214/dist_analy/
     :alt: Updates



Calculates distance matrices for a set of structures and run analyses that classify the conformational ensemble and identifies important interactions

This package is still in development

* Free software: MIT license
* Documentation: https://dist-analy.readthedocs.io.


Features
--------
-Gets PDBs

* TODO

Requirements
--------
::

  python
  numpy
  sklearn
  mdtraj
  biopython
  prody
  AlphaSpace2

Installation
--------

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

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
