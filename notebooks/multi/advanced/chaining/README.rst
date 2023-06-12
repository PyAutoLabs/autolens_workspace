The ``chaining`` folder contains examples scripts illustrating the non-linear search chaining feature for multiple imaging datasets.

Notes
-----

The ``multi`` extends the ``imaging`` package and readers should refer to the ``imaging`` package for more examples
of how to perform search chaining.

Files (Advanced)
----------------

- ``parametric_to_pixelization.py``: Fit a parametric source model followed by a pixelized source reconstruction.
- ``sie_to_power_law.py``: Fit a singular isothermal elliptical mass model followed by a power-law.

Folders (Advanced)
------------------

- ``pix_adapt``: Example chaining scripts and pipelines which use PyAutoLens's adaptive pixelization features, which adapt the source reconstruction to the source's morphology.
- ``slam``: Analysis scripts using the Source, Light and Mass (SLaM) pipelines to model strong lens imaging.
