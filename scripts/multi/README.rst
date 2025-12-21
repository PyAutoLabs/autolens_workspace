The ``multi`` folder contains example scripts showing how to perform analysis combining multiple datasets, for example:

- Multiple CCD imaging datasets observed at different wavelengths.
- Combining CCD imaging and interferometer datasets.
- Combining point source modeling of a lensed quasar with CCD imaging of its lens and source host galaxy.

These examples are relatively advanced, and users are recommended to be familiar with individual examples of
each example before using the ``multi`` functionality.

For example, if you want to combine CCD imaging and interferometer data, you should learn how to to analyse each
individually first by reading the examples in the ``imaging`` and ``interferometer`` folders.

Start Here
----------

New users should read the ``start_here`` example, which gives an overview of all examples in the folder.

Files
-----

- ``start_here``: A simple example illustrating how to analyse multiple datasets.
- ``modeling``: Detailed example of performing lens modeling of multiple dataset.
- ``simulator``: Detailed example of how to simulate multiple datasets of the same strong lens.
- ``data_preparation``: See ``imaging/data_preparation``, ``interferometer/data_preparation`` and `point_source/data_preparation``,  which contain all tools for preparing multiple datasets.

Folders
-------

- ``features``: Examples illustrating different core features for CCD imaging analysis and lens modeling.

Results
-------

The ``modeling`` example performs lens modeling but only give a brief overview of how to analyse and interpret the
results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/guides/results``.