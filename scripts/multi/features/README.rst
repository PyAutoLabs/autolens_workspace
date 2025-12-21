The ``multi/features`` folder contains example scripts showing how to analyse multiple datasets simultaneously.

This combines all dataset types, so it could be multple imaging datasets (e.g. multi wavelength) observations,
imaging and interferometry, combining point source modeling with extended source modeling of CCD imaging data, etc.

Notes
-----

The ``multi`` package extends other workspace packages and readers should refer to the relevent package for
descriptions of modeling and features. For example, if you are combining imaging and interferometer datasets,
refer to the ``imaging`` and ``interferometer`` packages first before reading their corresponding `multi` examples.

These scripts show how to perform lens modeling but only give a brief overview of how to analyse
and interpret the results a lens model fit. A full guide to result analysis is given at ``autolens_workspace/*/guides/results``.The following example illustrate multi-dataset lens modeling with features that are specific to having multiple datasets:

- ``dataset_offsets``: Datasets may have small offsets due to pointing errors, which can be accounted for in the model.
- ``imaging_and_interferometer``: Imaging and interferometer datasets are fitted simultaneously.
- ``one_by_one``: Multiple datasets are fitted one-by-one in a sequence.
- ``same_wavelength``: Multiple datasets that are observed at the same wavelength are fitted simultaneously.
- ``wavelength_dependence``: A model is fitted where parameters depend on wavelength following a functional form.
- ``pixelization``: The source is reconstructed using an adaptive Voronoi mesh.
- ``slam``: Using the Source, Light and Mass (SLAM) pipeline to perform lens modeling of multiple datasets simultaneously.