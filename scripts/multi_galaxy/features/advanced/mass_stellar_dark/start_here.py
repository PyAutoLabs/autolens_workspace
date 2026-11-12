"""
Features: Mass Stellar Dark
===========================

Investigate how the lensing mass is divided between stars and dark matter by linking a stellar mass component to the observed light and fitting an additional dark-halo component. The lensing constraints act on their combined deflection.

__Scientific Use Cases__

- Infer a stellar mass-to-light normalization under an explicit light and halo model.
- Explore the stellar and dark contributions over the radii constrained by the lensed images.
- Compare decomposed models with total-mass descriptions and test sensitivity to priors and light modeling.

__This Dataset and Lens Regime__

This folder treats imaging with multiple important deflectors. Include all of them in the ray tracing and
keep their light separate from their mass contributions. The combined deflection can be better constrained
than the allocation of mass between galaxies; examine that allocation when interpreting source properties
or small residual structures.

![mass stellar dark](../../../../../images/features/stellar_dark_illustration.png)

Conceptual illustration. The stellar and dark components are a model decomposition; lensing responds to their combined mass. The schematic illustrates the feature; retain all deflectors in this multi-galaxy example.

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


__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/mass_stellar_dark/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/mass_stellar_dark/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/mass_stellar_dark/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/mass_stellar_dark/simulator.ipynb) — Generate the example dataset.
- [chaining.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/mass_stellar_dark/chaining.ipynb) — Fit the model through linked searches.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/mass_stellar_dark/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
