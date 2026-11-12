"""
Features: Scaling Relation
==========================

A strong-lens model may need the collective gravitational contribution of many companion galaxies even when
the data cannot constrain every companion mass independently. Scaling relations connect their mass
normalizations to measured luminosities, representing the population through shared assumptions instead of
a separate unconstrained mass for every member.

__Scientific Use Cases__

Use this feature to account for a surrounding galaxy population when reconstructing the source and inferring
the main lens's mass. It is useful when a few companions strongly influence the arcs but many fainter galaxies
are only weakly constrained individually. Modeling those two populations differently keeps the important
individual freedoms while giving the fainter population a collective mass prescription.

The scientific trade-off is explicit: a smaller search space comes from a stronger population assumption.
The resulting companion masses are tied predictions of the adopted relation, not independent measurements
of each galaxy's mass.

![Galaxy luminosities inform a shared population mass model](../../../../images/features/scaling_relation.png)

Conceptual illustration: the mass symbols represent an assumed luminosity-based relation. They are not
visible dark matter, measured halo boundaries or individual mass detections.

__Two Companion Tiers__

The modeling example distinguishes two populations using their preparation files:

- The bounded tier, listed in `extra_galaxies_centres.json`, retains a free Einstein radius for each galaxy,
  within a luminosity-derived upper bound. Use this treatment for influential companions that merit individual freedom.
- The scaling tier, listed in `scaling_galaxies_centres.json`, ties each Einstein radius to a common relation.
  This represents companions whose individual masses the data constrain less strongly.

The file assignment is a modeling choice, not an automatic scientific classification. A companion close to an
arc may need individual treatment even if it is faint. Inspect residuals and test whether moving influential
members between treatments changes the lens or source conclusions.

__The Main Lens Anchors This Example__

The galaxy-scale recipe uses the relation:

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

Here the anchor is the main lens galaxy itself. Its Einstein radius is already fitted, and the luminosity
ratios and exponent are specified inputs. As the proposed main-lens mass changes, the tied companion Einstein
radii change with it.

This scaling tier therefore adds zero independent mass parameters in this particular recipe. It does not
mean that the entire model has no free parameters: the main lens, source, bounded companions and any fitted
light properties still contribute to the model. A version that fits a separate normalization or exponent
would have a different parameter count.

The relation ties Einstein radii, which describe lensing strength; it should not be read as a measurement of
each galaxy's total halo mass. The example uses untruncated isothermal mass profiles, so comparisons with other
mass definitions require an explicit profile and aperture convention.

__Population Assumptions and Luminosity Measurements__

Anchoring on the main lens assumes that it belongs on the same adopted relation as the tied companions.
A fixed exponent and deterministic relation also restrict the allowed differences between galaxies. The
relation is useful precisely because it supplies information not independently measured for every companion.

Use consistent luminosity measurements for the anchor and members. Blended light, different measurement
apertures or populations with different mass-to-light properties can affect those ratios. Read the luminosity
preparation and CSV-loading sections alongside the recipe, and retain the documented units and conventions.
The supplied relation does not automatically propagate all luminosity measurement errors or intrinsic scatter.

Do not infer a common redshift from proximity on the sky. The galaxies must be modeled at appropriate
redshifts, and a relation calibrated for one lensing geometry cannot be transferred unchanged to another
without checking its interpretation.

__Group and Cluster Normalizations__

The group and cluster workflows use a different convention: a separately fitted Einstein radius at a fixed
reference magnitude. That adds a shared normalization parameter and avoids selecting a particular member
as the anchor. Their mass-profile choices can also include truncation appropriate to the adopted environment.

These conventions are not interchangeable labels. Before transferring a recipe, check the anchor or reference
luminosity, exponent, redshifts, profile family and which members are tied. Use the group or cluster workflows
when their mass configuration better represents the system.

__Inspecting the Fit__

Compare the simulated or inferred contributions of the main lens, individually fitted companions and scaling
population. A good fit does not show that every tied mass is correct. Assess whether plausible changes to the
relation or treatment of influential members alter the scientific result.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/scaling_relation/fit.ipynb) — Inspect a fit with the scaling-relation construction.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/scaling_relation/modeling.ipynb) — Follow the bounded and scaling tiers, main-lens anchor, parameter count and CSV interface.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/scaling_relation/likelihood_function.ipynb) — Follow how the scaling model enters the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/scaling_relation/simulator.ipynb) — Generate a known population for exploring the example's assumptions.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/scaling_relation/slam.ipynb) — Carry the scaling-relation model through a staged fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the
workspace root.
"""
