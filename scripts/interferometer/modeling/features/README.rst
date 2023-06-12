The ``modeling`` folder contains example scripts showing how to fit lens models to interferometer datatick_maker.min_value:

The API for these features is the same irrespective of the dataset fitted.

Therefore, refer to the folder
`autolens_workspace/*/imaging/modeling/features` for example scripts, which can be copy
and pasted into scripts which model interferometer data.

An example script for a pixelized source is provided, as reconstructing the source
using a pixelization is a common use case for interferometer data.

Files (Beginner)
----------------

- ``pixelization.py``: The source is reconstructed using an adaptive Voronoi mesh.

Notes
-----

These scripts show how to perform lens modeling but only give a brief overview of how to
analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/imaging/results``.