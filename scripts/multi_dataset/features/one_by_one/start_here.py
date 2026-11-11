"""
Features: One By One
====================

Multi-dataset modeling does not always require fitting every observation simultaneously. A sequential approach
fits one dataset first, then uses the result to guide the analysis of another.

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

__In This Folder__

These examples connect several observations. Read which parameters are shared, which are dataset-specific and
whether the fits are simultaneous or sequential. The per-dataset likelihoods still retain their own response
and noise treatment.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/one_by_one/modeling.ipynb) — Fit the model with a non-linear search.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
