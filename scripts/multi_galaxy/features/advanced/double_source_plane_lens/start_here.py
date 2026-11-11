"""
Features: Double Source Plane Lens
==================================

A double source-plane lens contains background sources at two different redshifts behind the foreground
deflectors. Their light is traced through the relevant sequence of planes to predict the observed image.

__Keep the Planes Distinct__

The two sources do not share the same lensing geometry merely because their arcs appear in the same image.
Their redshifts enter the calculation, and a galaxy that is a source on one plane may also deflect light from
a more distant source when its mass is included.

The model must therefore identify each galaxy's plane and distinguish its light from its mass contribution.
The examples retain their particular redshifts and component choices beside the construction of the model.

__Follow the Geometry Before Fitting__

Start with the simulator or direct fit to see which arcs belong to which source. Then read the modeling
example. Chaining and SLaM examples, where provided, show how the information is carried through a staged fit;
they do not replace the need to check the plane assignments.

Multiple deflectors in group or multi-galaxy examples add components within this geometry. Preserve those
distinctions when adapting a single-deflector example.

__In This Folder__

The examples contain multiple main lens galaxies, rather than a single dominant deflector with only small
companions. Follow how the feature is applied to each galaxy and which parameters are shared. A group halo is
not implicit in this setup.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/simulator.ipynb) — Generate the example dataset.
- [chaining.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/chaining.ipynb) — Fit the model through linked searches.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
