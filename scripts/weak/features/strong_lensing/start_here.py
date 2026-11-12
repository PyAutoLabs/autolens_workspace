"""
Features: Strong Lensing
========================

Connect central strong-lensing constraints with the wider weak-lensing field to study a mass distribution over a larger range of radii. Each observable retains its own uncertainty model while constraining shared mass components.

__Scientific Use Cases__

- Investigate the radial mass distribution from central arcs to larger angular scales.
- Test whether a common mass model can explain both strong and weak lensing.
- Build toward analyses of observed cluster fields, including the separate A2744 workflow supplied here.

__This Dataset and Lens Regime__

The controlled simulation and the observed A2744 example have different preparation requirements. Follow
each script's coordinate conventions, background-source selection, uncertainty treatment and mass assumptions;
the simulation is not a complete model of a real cluster field.

![strong lensing](../../../../images/features/strong_weak_lensing.png)

Conceptual illustration. The panels illustrate different angular scales. Weak-lensing information is statistical, and the dashed mass contours are schematic.

__Different Data, Shared Mass__

The strong-lensing image and the weak-lensing measurements have different observables and uncertainties. Their
analyses should preserve those differences while referring to a consistent geometry and mass parameterization.

The two regimes also cover different angular scales. Coordinate conventions, field coverage and the components
included in the mass model matter when joining their constraints.

__Simulated and Observed Examples__

Start with the simulator and direct fit to understand the combined calculation. The modeling example
introduces inference, while the A2744 example is a separate observed-system workflow with its own preparation
and assumptions.

Keep the real-data preparation and dataset-specific reasoning in that example. A model suitable for the
controlled simulation is not automatically a complete description of a real cluster field.


__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/features/strong_lensing/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/features/strong_lensing/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/features/strong_lensing/simulator.ipynb) — Generate the example dataset.
- [a2744.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/features/strong_lensing/a2744.ipynb) — Follow the observed A2744 example.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
