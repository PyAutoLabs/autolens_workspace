"""
Features: Strong Lensing
========================

Strong and weak lensing probe different parts of a mass distribution. A combined analysis connects central
strongly lensed features with the wider weak-lensing field through a shared mass model.

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

__In This Folder__

The simulated examples and the observed-system example have different data-preparation requirements. Follow
the specific script’s coordinate conventions, data selection and model assumptions.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/features/strong_lensing/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/features/strong_lensing/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/features/strong_lensing/simulator.ipynb) — Generate the example dataset.
- [a2744.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/features/strong_lensing/a2744.ipynb) — Follow the observed A2744 example.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
