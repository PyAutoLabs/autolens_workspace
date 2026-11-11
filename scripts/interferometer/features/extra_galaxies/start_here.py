"""
Features: Extra Galaxies
========================

A target rarely occupies an otherwise empty field. Nearby galaxies may contribute light inside the region
being fitted, and a lensing model may also need to account for their deflection of the light rays.

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

__In This Folder__

These examples fit visibilities. The image-plane light prediction is transformed into the measured
Fourier-domain quantities; a dirty image is an inspection product, not a substitute for that likelihood. Keep
the visibility sampling, noise and model-specific response choices in view when adapting an imaging example.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/simulator.ipynb) — Generate the example dataset.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
