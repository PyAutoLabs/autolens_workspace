"""
Features: Extra Galaxies
========================

Include neighbouring galaxies whose mass or emission affects the lens analysis. A companion can alter the inferred source and main lens even when its own mass is poorly constrained; deciding which contributions matter is part of building the scientific model.

__Scientific Use Cases__

- Account for companion deflections near informative arcs or multiple images.
- Separate contaminating light where the fitted observable contains that light.
- Test the stability of lens and source inference when the surrounding galaxy population is represented explicitly.

__This Dataset and Lens Regime__

These examples fit complex visibilities: the lensed sky prediction is Fourier transformed and compared at
the measured baseline coordinates with the supplied noise. A dirty or reconstructed image is useful for
inspection, but does not replace this likelihood. Interpret source structure in the context of visibility
coverage and the assumptions used to model the sky.

![extra galaxies](../../../../images/features/extra_galaxies.png)

Conceptual illustration. Visible companions and schematic mass contributions are illustrative; a companion is not a mass detection merely because it appears in the image. The scene illustrates sky/source structure; the fit here is to visibilities, not the displayed image.

__Deciding What to Include__

There are two separate questions: does a neighbour contribute significant light to the fitted data, and does
its mass matter for the lensing calculation? Masking a galaxy's visible emission does not remove its
gravitational effect. In a galaxy-light analysis, the concern is instead how overlapping emission affects the
decomposition of the target.

The examples make the relevant choice for their data type. They retain the positions, light profiles, mass
components and parameter constraints beside the model-building code; these should not be transferred to a
different field without inspection.

__Separate the Contributions__

A good fit to the combined image does not guarantee that the light has been assigned correctly to each galaxy.
Inspect individual model components and residuals, especially where their emission overlaps. Fixing or sharing
parameters can help constrain a fit, but also introduces assumptions about the components.

Use the simulator to understand the example field, then follow the modeling example to add the neighbours
explicitly. In larger lens systems, distinguish the main deflectors from the additional companion population.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/simulator.ipynb) — Generate the example dataset.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
