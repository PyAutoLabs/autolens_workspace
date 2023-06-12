The ``chaining`` folder contains examples scripts illustrating the non-linear search chaining feature for interferometer datasets.

The API for search chaining is the same irrespective of the dataset fitted. Therefore, please refer to the folder
`autolens_workspace/*/interferometer/chaining` for more example scripts, which can be copy and pasted
into scripts which model interferometer data via chaining.

Folders (Advanced)
------------------

- ``pipelines``: Lens modeling pipelines that use search chaining to fit a model using multiple non-linear searches.
- ``slam``: Analysis scripts using the Source, Light and Mass (SLaM) pipelines to model strong lens imaging.

Files (Advanced)
----------------

- ``parametric_to_pixelization.py``: Fit a parametric source model followed by a pixelized source reconstruction.