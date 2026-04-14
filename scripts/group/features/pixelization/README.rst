The ``pixelization`` folder contains example scripts showing how to perform group-scale strong lens analysis using a pixelized source reconstruction.

Files
-----

The following example scripts illustrating group-scale lens modeling where:

- ``modeling``: Group-scale lens modeling using a pixelized source reconstruction (Delaunay mesh with AdaptSplit regularization).
- ``fit``: Fit a group-scale lens with a pixelized source and compute quantities like the residuals, chi squared and likelihood.
- ``likelihood_function``: A step-by-step guide of the pixelized source likelihood function for group-scale lenses.
- ``slam``: Using the SLaM (Source, Light and Mass) pipeline for group-scale lens modeling with a pixelized source.
- ``adaptive``: Advanced pixelization features which adapt the mesh and regularization to the source, including how adapt_data is constructed from group lens-light subtraction.
- ``delaunay``: Using a Delaunay mesh (instead of a rectangular mesh) for group-scale source reconstruction.
- ``cpu_fast_modeling``: How to speed up pixelized source modeling for group-scale lenses using CPUs with numba and sparse operators.
- ``source_science``: Performing source science calculations (total flux, magnification) using a pixelized source reconstruction for group-scale lenses.

Results
-------

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/guides/results``.
