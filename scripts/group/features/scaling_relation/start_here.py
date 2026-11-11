"""
Features: Scaling Relation
==========================

A scaling relation connects the properties of a galaxy population through a shared rule, rather than giving
every member a completely independent model. In these lensing examples, member luminosities help determine
their mass normalizations.

__Why Share a Relation?__

With many companion galaxies, independent mass parameters can create a large search space even when the data
provide limited information about each member. A scaling relation reduces that freedom by expressing the
population in terms of an anchor and a specified dependence on luminosity.

This exchanges independent parameters for assumptions about the population. The anchor, exponent and treatment
of individual members determine what the model can vary. A relation with all of those choices fixed has a
different parameter count from one that fits its normalization or slope.

__Read the Local Parameterization__

The examples explain which galaxy anchors the relation and how each companion is tied to it. Keep those
details in the modeling recipe: different lens scales do not necessarily use the same anchor. Where a
luminosity-fitting example is provided, it explains how the input light measurements are obtained.

Check whether a particularly influential companion needs its own treatment rather than forcing it to follow
the same population relation.

__In This Folder__

The examples contain a group-scale lens with several galaxy components. A shared group halo, where present, is
an explicit model choice. Keep the individual deflectors, companion population and source components distinct
when following the recipe.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/simulator.ipynb) — Generate the example dataset.
- [modeling_for_luminosities.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/modeling_for_luminosities.ipynb) — Fit the luminosities used by the relation.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
