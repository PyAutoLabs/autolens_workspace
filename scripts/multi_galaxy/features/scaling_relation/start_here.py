"""
Features: Scaling Relation
==========================

Represent a companion population through a luminosity-based mass prescription when the lensing data cannot constrain every galaxy independently. The relation reduces independent freedom by adding an explicit population assumption.

__Scientific Use Cases__

- Include the collective deflection of numerous faint companions.
- Reserve individual mass freedom for influential galaxies while tying weaker contributors to a shared relation.
- Test how population assumptions affect lens mass and source inference, rather than interpreting every tied mass as an independent measurement.

__This Dataset and Lens Regime__

Here the brightest main deflector anchors the companion relation:
einstein_radius_i = einstein_radius_brightest * (L_i/L_brightest)**0.5.
The other main deflectors retain their own masses. This is a relation for the companion tier, not a rule that
forces every dominant galaxy to share one mass-to-light prescription. Fixed luminosity ratios and slope do
not propagate luminosity errors or intrinsic population scatter automatically.

![scaling relation](../../../../images/features/scaling_relation.png)

Conceptual illustration. The mass symbols describe an assumed population relation, not measured halo boundaries or independent mass detections. The schematic illustrates the feature; retain all deflectors in this multi-galaxy example.

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

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/scaling_relation/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/scaling_relation/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/scaling_relation/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/scaling_relation/simulator.ipynb) — Generate the example dataset.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/scaling_relation/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
