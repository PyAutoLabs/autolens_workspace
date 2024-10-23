The ``modeling/features`` folder contains example scripts showing how to fit a lens model to a point source dataset.

The majority of features are the same irrespective of the dataset fitted.

Therefore, refer to the folder
`autolens_workspace/*/modeling/imaging/features` for example scripts, which can be copy
and pasted into scripts which model point source data.

The following example scripts are specific to point source datasets:

Files (Beginner)
----------------

- ``deblending.py``: Deblend the point-source images (e.g. of a lensed quasar) from the lens galaxy light to determine the positions of the point sources and measure the lens galaxy's properties.

Notes
-----

These scripts show how to perform lens modeling but only give a brief overview of how to analyse
and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/imaging/results``.

