The ``modeling`` folder contains example scripts showing how to fit a lens model to multiple imaging datasets:

Notes
-----

The ``multi`` package extends the ``imaging`` package and readers should refer to the ``imaging`` package for
descriptions of how to customize the non-linear search, the fit settings, etc.

These scripts show how to perform lens modeling but only give a brief overview of how to analyse
and interpret the results a lens model fit. A full guide to result analysis is given at ``autolens_workspace/*/imaging/results``.

Files (Beginner)
----------------

- ``start_here.py``: A simple example illustrating how to fit a lens model to multiple datasets.

Folders (Beginner)
------------------

- ``features``: Example modeling scripts for multiple datasets for more experienced users.
- ``customize``: Customize aspects of a model-fit, (e.g. priors, the imaging mask).
- ``searches``: Using other non-linear searches (E.g. MCMC, maximum likelihood estimators).
