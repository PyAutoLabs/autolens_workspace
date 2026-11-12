"""
Features: Scaling Relation
==========================

Represent a companion population through a luminosity-based mass prescription when the lensing data cannot constrain every galaxy independently. The relation reduces independent freedom by adding an explicit population assumption.

__Scientific Use Cases__

- Include the collective deflection of numerous faint companions.
- Reserve individual mass freedom for influential galaxies while tying weaker contributors to a shared relation.
- Test how population assumptions affect lens mass and source inference, rather than interpreting every tied mass as an independent measurement.

__This Dataset and Lens Regime__

The recipe takes companion luminosities from supporting imaging and ties their Einstein radii to the
main-lens anchor through (L_i/L_anchor)**0.5. The point dataset cannot measure the foreground luminosity
ratios itself. Positions and any included fluxes provide limited constraints, so compare the tied population
with individually fitted companions only where the data justify that freedom.

![scaling relation](../../../../images/features/scaling_relation.png)

Conceptual illustration. The mass symbols describe an assumed population relation, not measured halo boundaries or independent mass detections.

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


__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/scaling_relation/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/scaling_relation/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/scaling_relation/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/scaling_relation/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
