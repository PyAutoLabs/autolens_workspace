The ``modeling`` folder contains example scripts showing how to fit lens models to interferometer data:

The API for these features is the same irrespective of the dataset fitted.

Therefore, refer to the folder
`autolens_workspace/*/modeling/imaging/features` for example scripts, which can be copy
and pasted into scripts which model interferometer data.

An example script for lens modeling where the source is reconstructed using a pixelization, because this
is a common use case for interferometer data which requires its own dedicated consideration of how to ensure
the run-times are fast.

Files (Beginner)
----------------

- ``pixelization.py``: The source is reconstructed using a pixelixation which can capture irregular and clumpy morphologies.

Notes
-----

These scripts show how to perform lens modeling but only give a brief overview of how to
analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/interferometer/results``.