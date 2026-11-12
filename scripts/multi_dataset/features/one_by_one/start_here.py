"""
Features: One By One
====================

Use one observation to initialize the analysis of another, or compare datasets through separate fits before a joint analysis. Sequential modeling is useful for organizing multi-band lens and source inference, provided the transferred information is made explicit.

__Scientific Use Cases__

- Initialize a difficult fit from a dataset that constrains useful starting values.
- Measure emission in a weaker band conditional on a chosen earlier lens model.
- Diagnose disagreements between datasets before deciding which parameters to share in a simultaneous fit.

__This Dataset and Lens Regime__

Decide explicitly which parameters are shared, which vary by dataset and whether the fits are simultaneous
or sequential. Each dataset retains its own response and uncertainty treatment. Inspect individual residuals
alongside the combined result; sharing a lens does not require identical emission in every observation.

![one by one](../../../../images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic. The interferometer panel shows sampling locations, not visibility amplitudes or phases; cube slices are spectral channels. These are conceptual dataset representations, not measurements.

__What Is Carried Forward?__

A result can inform initial values, priors or fixed parameters. These choices have different consequences:
fixing a parameter to an earlier estimate does not propagate its uncertainty in the same way as a joint fit.

Read precisely what the example transfers and what it fits again. The order of the datasets can matter when
early decisions constrain the later analysis.

__When to Compare with a Joint Fit__

Sequential fitting can make a complicated analysis easier to organize, particularly when one observation
supplies a useful starting model. If the datasets offer complementary constraints on shared parameters, a
simultaneous analysis may be needed to represent that information together.

The modeling script keeps its transfer and refitting steps explicit so you can follow how the later result
depends on the earlier one.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/one_by_one/modeling.ipynb) — Fit the model with a non-linear search.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
