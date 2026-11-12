"""
Features: Scaling Relation
==========================

Represent a companion population through a luminosity-based mass prescription when the lensing data cannot constrain every galaxy independently. The relation reduces independent freedom by adding an explicit population assumption.

__Scientific Use Cases__

- Include the collective deflection of numerous faint companions.
- Reserve individual mass freedom for influential galaxies while tying weaker contributors to a shared relation.
- Test how population assumptions affect lens mass and source inference, rather than interpreting every tied mass as an independent measurement.

__This Dataset and Lens Regime__

The companion luminosities come from supporting imaging, not from an assumption that foreground stellar
light is visible in the interferometer band. The recipe ties companion Einstein radii to the main-lens
anchor through (L_i/L_anchor)**0.5. Their gravitational effect remains even when their emission is negligible
in the visibilities. Check whether any bright radio or submillimetre companion also needs an emission model.

These examples fit complex visibilities: the lensed sky prediction is Fourier transformed and compared at
the measured baseline coordinates with the supplied noise. A dirty or reconstructed image is useful for
inspection, but does not replace this likelihood. Interpret source structure in the context of visibility
coverage and the assumptions used to model the sky.

![scaling relation](../../../../images/features/scaling_relation.png)

Conceptual illustration. The mass symbols describe an assumed population relation, not measured halo boundaries or independent mass detections. The scene illustrates sky/source structure; the fit here is to visibilities, not the displayed image.

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

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/scaling_relation/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/scaling_relation/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/scaling_relation/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/scaling_relation/simulator.ipynb) — Generate the example dataset.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/scaling_relation/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
