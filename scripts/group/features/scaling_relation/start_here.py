"""
Features: Scaling Relation
==========================

Represent a companion population through a luminosity-based mass prescription when the lensing data cannot constrain every galaxy independently. The relation reduces independent freedom by adding an explicit population assumption.

__Scientific Use Cases__

- Include the collective deflection of numerous faint companions.
- Reserve individual mass freedom for influential galaxies while tying weaker contributors to a shared relation.
- Test how population assumptions affect lens mass and source inference, rather than interpreting every tied mass as an independent measurement.

__This Dataset and Lens Regime__

The group recipe uses truncated dPIE member profiles. Its scaling tier fits a reference velocity-dispersion
normalization rather than anchoring every companion directly to the main galaxy's fitted Einstein radius.
In the current recipe, sigma scales as (L/L_ref)**0.25 and r_cut as (L/L_ref)**0.7, with fixed reference
truncation radius and zero core radius. These exponents and truncation choices are assumptions, not values
that every group must obey. The individually bounded tier and the scaling tier remain distinct.

![scaling relation](../../../../images/features/scaling_relation.png)

Conceptual illustration. The mass symbols describe an assumed population relation, not measured halo boundaries or independent mass detections. The schematic illustrates the feature; retain all deflectors in this group example.

__Why Share a Relation?__

With many companion galaxies, independent mass parameters can create a large search space even when the data
provide limited information about each member. A scaling relation reduces that freedom by expressing the
population in terms of an anchor and a specified dependence on luminosity.

This exchanges independent parameters for assumptions about the population. The anchor, exponent and treatment
of individual members determine what the model can vary. A relation with all of those choices fixed has a
different parameter count from one that fits its normalization or slope.

__Read the Local Parameterization__

Read the reference luminosity, fitted velocity-dispersion normalization and fixed truncation scaling in the
local recipe. The luminosity-fitting example explains how the population inputs are obtained. This group
parameterization differs from a main-galaxy Einstein-radius anchor.

Check whether a particularly influential companion needs its own treatment rather than forcing it to follow
the same population relation.


__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/simulator.ipynb) — Generate the example dataset.
- [modeling_for_luminosities.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/modeling_for_luminosities.ipynb) — Fit the luminosities used by the relation.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
