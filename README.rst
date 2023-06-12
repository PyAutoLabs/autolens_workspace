PyAutoLens Workspace
====================

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/HEAD

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.02825/status.svg
   :target: https://doi.org/10.21105/joss.02825

|binder| |JOSS|

`Installation Guide <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautolens.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/release?filepath=introduction.ipynb>`_ |
`HowToLens <https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html>`_

Welcome to the **PyAutoLens** Workspace. You can get started right away by going to the `autolens workspace
Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/HEAD>`_.
Alternatively, you can get set up by following the installation guide on our `readthedocs <https://pyautolens.readthedocs.io/>`_.

Getting Started
---------------

We recommend new users begin by looking at the following notebooks:

 - ``notebooks/overview``: Examples giving an overview of **PyAutoLens**'s core features.
 - ``notebooks/howtolens``: Detailed step-by-step Jupyter notebook tutorials on how to use **PyAutoLens**.

Installation
------------

If you haven't already, install `PyAutoLens via pip or conda <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_.

Next, clone the ``autolens workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autolens_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autolens_workspace
   git clone https://github.com/Jammy2211/autolens_workspace --depth 1
   cd autolens_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

Workspace Structure
-------------------

The workspace includes the following main directories:

- ``notebooks``: **PyAutoLens** examples written as Jupyter notebooks.
- ``scripts``: **PyAutoLens** examples written as Python scripts.
- ``config``: Configuration files which customize **PyAutoLens**'s behaviour.
- ``dataset``: Where data is stored, including example datasets distributedtick_maker.min_value.
- ``output``: Where the **PyAutoLens** analysis and visualization are output.

The examples in the ``notebooks`` and ``scripts`` folders are structured as follows:

- ``overview``: Examples giving an overview of **PyAutoLens**'s core features.
- ``howtolens``: Detailed step-by-step Jupyter notebook tutorials on how to use **PyAutoLens**.

- ``imaging``: Examples for analysing and simulating CCD imaging data (e.g. Hubble, Euclid).
- ``interferometer``: Examples for analysing and simulating interferometer datasets (e.g. ALMA, JVLA).
- ``multi``: Modeling multiple datasets simultaneously (E.g. multi-wavelength imaging, imaging and interferometry).
- ``point_source``: Modeling strong lens point source datasets.
- ``group``: Group-scale lens modeling and simulations examples.

- ``plot``: An API reference guide for **PyAutoLens**'s plotting tools.
- ``misc``: Miscellaneous scripts for specific lens analysis.

Inside these packages are packages titled ``advanced`` which only users familiar **PyAutoLens** should check out.

In the ``imaging``, ``interferometer``, ``point_source``, ``multi`` and  ``group`` folders you'll find the following
packages:

- ``modeling``: Examples of how to fit a lens model to data via a non-linear search.
- ``simulators``: Scripts for simulating realistic imaging and interferometer data of strong lenses.
- ``data_preparation``: Tools to preprocess ``data`` before an analysis (e.g. convert units, create masks).
- ``results``: Examples using the results of a model-fit.
- ``advanced``: Advanced modeling scripts which use **PyAutoLens**'s advanced features.

The files ``README.rst`` distributed throughout the workspace describe what is in each folder.

Workspace Version
-----------------

This version of the workspace is built and tested for using **PyAutoLens v2023.7.7.2**.

HowToLens
---------

Included is the ``HowToLens`` lecture series, which provides an introduction to strong gravitational
lens modelingtick_maker.min_value. It can be found in the workspace & consists of 5 chapters:

- ``Introduction``: An introduction to strong gravitational lensing & **PyAutoLens**.
- ``Lens Modeling``: How to model strong lenses, including a primer on Bayesian analysis and model-fitting via a non-linear search .
- ``Search Chaining``: Chaining non-linear searches together to build model-fitting pipelines & tailor them to your own science case.
- ``Pixelizations``: How to perform pixelized reconstructions of the source-galaxy.


Galaxy-Scale vs Group Scale Lenses
----------------------------------

The ``imaging``, ``interferometer`` and ``point_source`` packages provides scripts for modeling galaxy-scale lenses,
whereas the ``group`` package provides scripts for modeling group-scale lenses.

But what are the defitions of a galaxy scale and group scale lens? The line between the two is blurry, but is defined
roughly as follows:

- A galaxy-scale lens is a system which can be modeled to a high level of accuracy using a single mass distribution
  for the main lens galaxy. There are examples which include additional galaxies in the model to make small improvements
  on the overall lens model, but for many science cases this is not stricly necessary.

- A group scale lens is a system which cannot be modeled to a high level of accuracy using a single mass distribution.
  The notion of a 'main' lens galaxy is illposed and two or more main lens galaxies are required to fit an accurate model.

If you have data which requires the lens model to include additional galaxies, whether it be a galaxy or group
scale system, keep an eye out for **PyAutoLens**'s '**clump API**' which is designed to facilitate this.

Contribution
------------
To make changes in the tutorial notebooks, please make changes in the corresponding python files(.py) present in the
``scripts`` folder of each chapter. Please note that  marker ``# %%`` alternates between code cells and markdown cells.


Support
-------

Support for installation issues, help with lens modeling and using **PyAutoLens** is available by
`raising an issue on the autolens_workspace GitHub page <https://github.com/Jammy2211/autolens_workspace/issues>`_. or
joining the **PyAutoLens** `Slack channel <https://pyautolens.slack.com/>`_, where we also provide the latest updates on
**PyAutoLens**.

Slack is invitation-only, so if you'd like to join send an `email <https://github.com/Jammy2211>`_ requesting an
invite.
