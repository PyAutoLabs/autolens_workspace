The ``pixelization`` folder contains example scripts showing how to perform analysis using a pixelized source reconstruction.

Files
-----

The following example scripts illustrating lens modeling where:

- ``modeling``: Lens modeling using a pixelized source reconstruction.
- ``fit``: Fit a pixelized source and compute quantities like the residuals, chi squared and likelihood.
- ``likelihood_function``: A step-by-step guide of the pixelized source likelihood function.
- ``adaptive``: Advanced pixelization features which adapt the mesh and regularization to the source being reconstructed.
- ``slam``: Using the Source, Light and Mass (SLAM) pipeline to perform lens modeling using pixelized source reconstruction.
- ``delaunay``: Using a Delaunay mesh (instead of a rectangular mesh) for the source reconstruction.
- ``source_reconstruction``: How to output a source reconstruction from a pixelized source model fit. to a .csv which can be loaded to analyse the source without autolens.
- ``many_visibilities_preparation``: How to prepare the linear algbera for datasets with many visibilities for efficient memory usage and run times.

Results
-------

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/guides/results``.