The ``pixelization`` folder contains example scripts showing how to perform analysis using a pixelized source reconstruction.

Files
-----

The following example scripts illustrating lens modeling where:

- ``modeling``: Lens modeling using a pixelized source reconstruction.
- ``fit``: Fit a pixelized source and compute quantities like the residuals, chi squared and likelihood.
- ``likelihood_function``: A step-by-step guide of the pixelized source likelihood function.
- ``chaining``: Using non-linear search chaining to fit a parametric source followed by a pixelized source, making modeling more efficient and robust.
- ``adaptive``: Advanced pixelization features which adapt the mesh and regularization to the source being reconstructed.
- ``delaunay``: Using a Delaunay mesh (instead of a rectangular mesh) for the source reconstruction.
- ``slam``: Using the Source, Light and Mass (SLAM) pipeline to perform lens modeling using pixelized source reconstruction.

Results
-------

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/results``.