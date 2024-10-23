PyAutoLens Workspace
====================

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/HEAD

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.02825/status.svg
   :target: https://doi.org/10.21105/joss.02825

|binder| |JOSS|

`Installation Guide <https://pyautolens.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautolens.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/release?filepath=start_here.ipynb>`_ |
`HowToLens <https://pyautolens.readthedocs.io/en/latest/howtolens/howtolens.html>`_

.. image:: https://github.com/Jammy2211/PyAutoLogo/blob/main/gifs/pyautolens.gif?raw=true
  :width: 900

Welcome to the **PyAutoLens** Workspace!

Getting Started
---------------

You can get set up on your personal computer by following the installation guide on
our `readthedocs <https://pyautolens.readthedocs.io/>`_.

Alternatively, you can try **PyAutoLens** out in a web browser by going to the `autolens workspace
Binder <https://mybinder.org/v2/gh/Jammy2211/autolens_workspace/release?filepath=start_here.ipynb>`_.

New Users
---------

New users should read the ``autolens_workspace/start_here.ipynb`` notebook, which will give you a concise
overview of **PyAutoLens**'s core features and API.

Then checkout the `new user starting guide <https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html>`_ to navigate the workspace for your science case.

HowToLens
---------

For experienced scientists, the workspace examples will be easy to follow. Concepts surrounding strong lensing were
already familiar and the statistical techniques used for fitting and modeling already understood.

For those less familiar with these concepts (e.g. undergraduate students, new PhD students or interested members of the
public), things may have been less clear and a slower more detailed explanation of each concept would be beneficial.

The **HowToLens** Jupyter Notebook lectures are provide exactly this. They are a 3+ chapter guide which thoroughly
take you through the core concepts of strong lensing, teach you the principles of the statistical techniques
used in modeling and ultimately will allow you to undertake scientific research like a professional astronomer.

To complete thoroughly, they'll probably take 2-4 days, so you may want try moving ahead to the examples but can
go back to these lectures if you find them hard to follow.

If this sounds like it suits you, checkout the ``autolens_workspace/notebooks/howtolens`` package.

Workspace Structure
-------------------

The workspace includes the following main directories:

- ``notebooks``: **PyAutoLens** examples written as Jupyter notebooks.
- ``scripts``: **PyAutoLens** examples written as Python scripts.
- ``config``: Configuration files which customize **PyAutoLens**'s behaviour.
- ``dataset``: Where data is stored, including example datasets distributed.
- ``output``: Where the **PyAutoLens** analysis and visualization are output.
- ``slam``: The Source, Light and Mass (SLaM) pipelines to model strong lens imaging **(Advanced)**.

The examples in the ``notebooks`` and ``scripts`` folders are structured as follows:

- ``howtolens``: The **HowToLens** lectures which teach inexperienced scientists what strong lensing is how to use **PyAutoLens**.

- ``simulators``: Simulating example strong lens datasets fitted throughout examples.
- ``modeling``: Fitting simple lens models to data.
- ``data_preparation``: Preparing real data before PyAutoLens analysis.

- ``results``: How to use the results of a **PyAutoLens** model-fit (includes the ``database``).
- ``plot``: How to plot lensing quantities and results.

- ``features``: Features for detailed modeling and analysis of strong lenses (e.g. Multi Gaussian Expansion, Pixelizations).

Inside the ``simulators``, ``modeling`` and ``data_preparation`` packages are separate packages for each
unique dataset type:

- ``imaging``: Examples for galaxy scale strong lenses observed with CCD imaging (e.g. Hubble, Euclid).
- ``interferometer``: Examples for galaxy scale strong lenses observed with an interferometer (e.g. ALMA, JVLA).
- ``point_source``: Examples for strong lens point source datasets.
- ``group``: Examples for group scale strong lenses with.

The ``advanced`` package contains examples which use **PyAutoLens**'s advanced features, such as the SLaM pipelines,
which only experienced users should check out.

The files ``README.rst`` distributed throughout the workspace describe what is in each folder.

Workspace Version
-----------------

This version of the workspace is built and tested for using **PyAutoLens v2024.9.21.2**.

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
scale system, keep an eye out for **PyAutoLens**'s '**extra galaxies API**' which is designed to facilitate this.