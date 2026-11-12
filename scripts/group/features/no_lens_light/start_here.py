"""
Features: No Lens Light
=======================

Fit the lens mass and background source when foreground galaxy emission is negligible in the fitted pixels or has already been subtracted. Omitting foreground light leaves all required gravitational deflectors in the model.

__Scientific Use Cases__

- Analyse foreground-faint observations without introducing unnecessary light components.
- Model prepared lens-light-subtracted data while testing sensitivity to subtraction residuals.
- Concentrate on source morphology and lens mass when foreground emission is adequately accounted for.

For source-science measurements, continue to [source_science.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/source_science.ipynb).
Interpret the combined reconstructed light together with the lens and reconstruction assumptions.

__This Dataset and Lens Regime__

This folder treats group-scale imaging with individual member galaxies. Distinguish the main deflectors,
additional members and any explicit common group halo. The light decomposition and the split of mass among
components can affect the inferred source; a good combined fit does not guarantee that each component is
independently constrained.

![no lens light](../../../../images/features/no_lens_light.png)

Conceptual illustration. The dashed contours represent an assumed deflecting mass, not visible light or a measured mass map. The schematic illustrates the feature; retain all deflectors in this group example.

__Light and Mass Are Different Parts of the Model__

Omitting a foreground light profile does not remove the foreground galaxy's deflection. The mass model must
still map the background source into the observed arcs. The examples isolate this distinction by fitting data
without a foreground-light contribution.

Check why the light is absent in your own data. A simulation with no foreground emission is a simpler case
than an observation from which a fitted foreground model has been subtracted. In the latter case, imperfect
subtraction can leave residual structure that a source model may try to explain.

__Following the Examples__

The simulator defines the example's emission and deflectors. The modeling example then constructs the mass and
source-light model, while any SLaM example carries the same premise through its stages. The omission of
foreground light remains explicit in each recipe so that the script can still be understood on its own.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/no_lens_light/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/no_lens_light/simulator.ipynb) — Generate the example dataset.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/no_lens_light/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
