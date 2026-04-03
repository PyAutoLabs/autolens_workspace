The ``results/examples`` folder contains example scripts for using the results of a **PyAutoLens** model-fit.

Files
-----

- ``samples``: Non-linear search model results (parameter estimates, posteriors, errors, etc.).
- ``samples_via_aggregator``: Loading samples via the aggregator, which is more efficient for large libraries of results.
- ``queries``: Query the database to get certain modeling results (e.g. all models where ``einstein_radius > 1.0``).
- ``models``: Inspect the models in the database (e.g. visualize the galaxy images).
- ``data_fitting``: Reperform fitting of the data using the results to test their robustness (e.g. visualize the for different models residuals).
- ``galaxies_fit``:  Inspecting individual galaxies (e.g. their light profiles) and fits (e.g. model-images, chi-squareds, likelihoods).
- ``interferometer``: Analysing the results of an interferometer fit.