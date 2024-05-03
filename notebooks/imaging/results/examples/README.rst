The ``results/examples`` folder contains example scripts for using the results of a **PyAutoLens** model-fit.

Files (Beginner)
----------------

- ``samples.py``: Non-linear search model results (parameter estimates, errors, etc.).
- ``samples_via_aggregator.py``: Loading samples via the aggregator.
- ``queries.py``: Query the database to get certain modeling results (e.g. all models where `effective_radius > 1.0`).
- ``models.py``: Inspect the models in the database (e.g. visualize the galaxy images).
- ``data_fitting.py``: Inspect the data-fitting results in the database (e.g. visualize the residuals).
- ``galaxies_fit.py``:  Inspecting tracers, galaxies (e.g. their light profiles) and fits (e.g. model-images, chi-squareds, likelihoods).

Folders (Advanced)
------------------

- ``advanced``: Advanced result creation and manipulation (e.g. dark matter subhalo analysis).