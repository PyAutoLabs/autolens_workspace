"""
Features: Sky Background
========================

Fit an uncertain sky level alongside the galaxies and lensed source, especially when faint extended light is important. The background and galaxy outskirts can trade off, affecting inferred sizes, profile shapes and total light.

__Scientific Use Cases__

- Account for a residual offset after sky subtraction.
- Measure extended galaxy light while allowing uncertainty in the background level.
- Fit simulated or observed imaging that retains an additive sky component.

__This Dataset and Lens Regime__

This folder treats imaging with multiple important deflectors. Include all of them in the ray tracing and
keep their light separate from their mass contributions. The combined deflection can be better constrained
than the allocation of mass between galaxies; examine that allocation when interpreting source properties
or small residual structures.

![sky background](../../../../../images/features/sky_background.png)

Conceptual illustration. The layers are additive model components, not physical planes. Removing the mean background does not remove sky photon noise. The schematic illustrates the feature; retain all deflectors in this multi-galaxy example.

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


__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/sky_background/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/sky_background/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/sky_background/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
