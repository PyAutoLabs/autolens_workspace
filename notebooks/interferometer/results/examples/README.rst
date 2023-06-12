The ``interferometer/results`` folder contains example scripts for using the results of a **PyAutoLens** model-fit.

The API for inspecting the results of most quantities (e.g. the non-linar samples or inspecting the tracer) is the
same irrespective of the dataset fitted. Therefore, please refer to the folder
`autolens_workspace/*/imaging/results/examples` for example scripts, which can be copy and pasted
into scripts which model interferometer data.

This package does include an example result, which is the result of a fit to the interferometer dataset.

Files (Beginner)
----------------

- ``interferometer``: Analysing the results of an interferometer fit.

Files (In Imaging)
------------------

The following results are available in the ``autolens_workspace/*/imaging/results`` folder, but are fully
application to interferometer data and can therefore have code copy and pasted from them.

- ``samples.py``: Non-linear search lens model results (parameter estimates, errors, etc.).
- ``tracer.py``:  ``Tracer``  modeling results (images, convergences, etc.).
- ``fits.py``:  Fitting results of a model (model-images, chi-squareds, likelihoods, etc.).
- ``galaxies.py``:  Inspecting individual galaxies, light profiles and mass profile in a lens model.
- ``units_and_cosmology.py``: Unit conversions and Cosmological quantities (converting to kiloparsecs, Einstein masses, etc.).
- ``linear``:  Analysing the results of fits using linear light profiles via an inversion.
- ``pixelization``:  Analysing the results of a pixelized source reconstruction via an inversion.