The ``advanced/modeling`` folder contains example scripts showing how to fit a lens model to multiple imaging datasets:

Notes
-----

The ``multi`` package extends the ``imaging`` package and readers should refer to the ``imaging`` package for
descriptions of how to customize the non-linear search, the fit settings, etc.

These scripts show how to perform lens modeling but only give a brief overview of how to analyse
and interpret the results a lens model fit. A full guide to result analysis is given at ``autolens_workspace/*/imaging/results``.

Files (Advanced)
----------------

The following example illustrate multi-dataset lens modeling with features that are specific to having multiple datasets:

- ``dataset_offsets.py``: Datasets may have small offsets due to pointing errors, which can be accounted for in the model.
- ``imaging_and_interferometer.py``: Imaging and interferometer datasets are fitted simultaneously.
- ``same_wavelength.py``: Multiple datasets that are observed at the same wavelength are fitted simultaneously.
- ``wavelength_dependence.py``: A model is fitted where parameters depend on wavelength following a functional form.

The following examples illustrate multi-dataset lens modeling with features that are also used for single-dataset lens modeling:

- ``no_lens_light.py``: The foreground lens's light is not present in the data and thus omitted from the model.
- ``linear_light_profiles.py``: The model includes light profiles which use linear algebra to solve for their intensity, reducing model complexity.
- ``pixelization.py``: The source is reconstructed using an adaptive Voronoi mesh.