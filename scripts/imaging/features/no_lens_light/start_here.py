"""
Features: No Lens Light
=======================

Some lensing datasets contain no visible foreground lens light. This may be a property of the observation or a
consequence of subtracting a foreground-light model before the lens analysis.

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

__In This Folder__

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF. The
foreground light, deflecting mass and background source play different roles; retain the particular component
choices stated in each recipe.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/no_lens_light/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/no_lens_light/simulator.ipynb) — Generate the example dataset.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/no_lens_light/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
