"""
Features: Extra Galaxies
========================

Include neighbouring galaxies whose mass or emission affects the lens analysis. A companion can alter the inferred source and main lens even when its own mass is poorly constrained; deciding which contributions matter is part of building the scientific model.

__Scientific Use Cases__

- Account for companion deflections near informative arcs or multiple images.
- Keep companion mass effects separate from blending errors in the measured compact-image observables.
- Test the stability of lens and source inference when the surrounding galaxy population is represented explicitly.

__This Dataset and Lens Regime__

Point-image constraints are finite, so prioritize companions that can materially change the lensing predictions.

![extra galaxies](../../../../images/features/extra_galaxies.png)

Conceptual illustration. Visible companions and schematic mass contributions are illustrative; a companion is not a mass detection merely because it appears in the image.

__Deciding What to Include__

Here the main scientific effect of companions is their deflection of the compact images. Foreground light
is not directly fitted by a positional or flux likelihood; blends matter when extracting the input image
positions and fluxes. Supporting imaging supplies companion locations and, when needed, luminosity information.

Use supporting imaging to establish companion positions. Decide which companions need independent mass
parameters and which can be tied through a population prescription.

__Separate the Contributions__

Inspect predicted image positions and any fitted fluxes separately. If the measurements came from blended
imaging, assess that upstream decomposition as well; a point-source likelihood does not re-fit foreground
light. Masking visible emission in the preparation stage does not remove a companion's deflection.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/extra_galaxies/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/extra_galaxies/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
