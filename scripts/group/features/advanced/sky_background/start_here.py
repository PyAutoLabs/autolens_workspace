"""
Features: Sky Background
========================

The background contributes light across an image in addition to the galaxy or lens being studied. If a
background contribution remains in the data, it can be included in the model rather than attributed to the
target's emission.

__Why the Outskirts Are Sensitive__

Bright central light can be easy to distinguish from the background, while faint extended emission may be
comparable to it. A change in the assumed sky level can therefore change the inferred outer profile even when
the central fit appears satisfactory.

The background and the extended light model should be considered together. A flexible galaxy model may absorb
an incorrect background, while a freely fitted background can absorb genuine low-surface-brightness emission.

__Using the Examples__

The simulator establishes the background used in the demonstration. The direct fit and modeling examples show
how it enters the analysis and which parameters are fixed or inferred. Keep the data units, mask and
background parameterization explicit when adapting them.

Inspect residuals across the fitted region and consider how strongly the available background pixels constrain
the sky. The example's chosen prior and field of view are part of its setup, not universal choices for every
image.

__In This Folder__

The examples contain a group-scale lens with several galaxy components. A shared group halo, where present, is
an explicit model choice. Keep the individual deflectors, companion population and source components distinct
when following the recipe.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/sky_background/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/sky_background/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/sky_background/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
