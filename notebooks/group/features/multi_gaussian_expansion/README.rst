The ``multi_gaussian_expansion`` folder contains example scripts showing how to perform group-scale strong lens analysis using a Multi Gaussian Expansion (MGE).

An MGE decomposes each galaxy's light into ~10-30+ Gaussians whose intensities are solved via linear algebra. This is especially powerful for group-scale lenses because adding extra galaxies does not increase the number of non-linear parameters in the model.

Files
-----

The following example scripts illustrate group-scale lens modeling using MGE:

- ``modeling``: Lens modeling using MGE for all galaxies (main lens, extras, and source).
- ``simulator``: Simulating the group-scale strong lens dataset used by these examples.
- ``fit``: Fit the group lens model and compute quantities like the residuals, chi squared and likelihood.
- ``source_science``: Performing source science calculations (total flux, magnification) with an MGE source model.
- ``likelihood_function``: A step-by-step guide of the group-scale likelihood function with MGE.
- ``slam``: Using the Source, Light and Mass (SLAM) pipeline for group-scale MGE modeling.

Results
-------

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/guides/results``.
