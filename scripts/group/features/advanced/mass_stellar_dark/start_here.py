"""
Features: Mass Stellar Dark
===========================

A decomposed mass model separates the mass associated with stellar light from a dark-halo contribution. It
provides an alternative to describing the total mass with a single profile.

__From Light to Stellar Mass__

A stellar mass component uses a light distribution together with a mass-to-light ratio or related
normalization. The light model therefore becomes part of the mass-model assumptions: an inadequate light
description can affect the inferred stellar contribution.

The dark halo supplies another mass component. Lensing responds to their combined deflection, so a
satisfactory total model does not necessarily imply that the stellar and dark contributions are individually
well determined.

__What Remains Specific to the Example__

The examples state how the stellar mass follows the light, which mass-to-light ratios are shared or free, and
how the halo is parameterized. Multi-galaxy and group examples make these decisions for more than one
deflector.

Use the direct fit to inspect the components, then follow the modeling or chaining example. Interpret the
decomposition together with its priors and assumptions rather than treating each fitted component as an
independent measurement.

__In This Folder__

The examples contain a group-scale lens with several galaxy components. A shared group halo, where present, is
an explicit model choice. Keep the individual deflectors, companion population and source components distinct
when following the recipe.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/mass_stellar_dark/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/mass_stellar_dark/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/mass_stellar_dark/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/mass_stellar_dark/simulator.ipynb) — Generate the example dataset.
- [chaining.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/mass_stellar_dark/chaining.ipynb) — Fit the model through linked searches.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/mass_stellar_dark/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
