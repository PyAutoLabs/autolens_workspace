The ``guides`` folder contains guides explaining how specific aspects of **PyAutoLens** work.

These scripts are intended to help users understand how **PyAutoLens** works.

Files
-----

The following files in the ``guides`` folder are important, providing a full overview of the core API fundamental to
most calculations.

- ``data_structures``: How the NumPy arrays containing results are structured and the API for using them.
- ``tracer`` Performing ray-tracing and lensing calculations.
- ``galaxies`` Creating and using galaxies and their mass and light profiles.

Folders
-------

- ``units``: Unit conversions (e.g. angles in arcseconds, flux units to ab magnitude, Cosmology).
- ``results``: How to use the results of a **PyAutoLens** model-fit (includes the ``database``).
- ``modeling``: How to customize model-fitting (e.g. non-linear search settings).
- ``log_likelihood_function``: A step-by-step visual guide to how likelihoods are evaluated in **PyAutoLens**.
- ``hpc``: How to run model-fits on high-performance computing clusters.
- ``plot``: How to plot lensing quantities and results.
- ``advanced``: Advanced guides for a variety of features.