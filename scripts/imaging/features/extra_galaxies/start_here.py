"""
Features: Extra Galaxies
========================

Nearby galaxies can affect a strong-lens analysis in two distinct ways: their light can overlap the arcs,
and their mass can contribute to the deflection of the background source. The `extra_galaxies` feature lets
you account for these neighbours when reconstructing the source and inferring the main lens's mass distribution.

__Scientific Use Cases__

Use this feature when a companion's light blends with the lensed emission, when a nearby galaxy's gravity
may influence the arcs, or when testing whether visible neighbouring structure can explain an apparent
lensing anomaly. Modeling these contributions helps distinguish source structure from foreground light
and gives the lens model a way to account for mass beyond the main galaxy.

This is also relevant to source science: light from a neighbour should not be reconstructed as a bright
clump in the background galaxy, and an unmodeled deflection can change the inferred source morphology.
The aim is to recover the lens and source while accounting for their surroundings, not merely to obtain
a visually clean image.

The examples focus on a galaxy-scale lens with a small number of additional galaxies. Decide separately
whether each neighbour needs a light model, a mass model, or both.

![Nearby galaxies contribute overlapping light and additional deflection](../../../../images/features/extra_galaxies.png)

Conceptual illustration: the two panels emphasize the light and gravitational contributions of the same
neighbouring galaxies. Both effects can be present together; this is not a fitted dataset or a numerical
prediction. Any dashed arc is a schematic reference for the main lens alone.

__Light and Mass Are Separate Decisions__

A neighbour's emission matters when it contributes appreciably to the pixels being fitted, including light
spread by the point-spread function (PSF). Its mass matters when its deflection changes the lensed source
image enough to affect the scientific measurement. These need not be the same set of galaxies.

Masking a galaxy's visible emission does not remove its gravitational effect. You can suppress its light
in the fitted data while retaining a mass component, or model its light without adding a mass component
when that is justified. The appropriate choice depends on the field and the required accuracy.

Projected proximity does not establish that a neighbour lies at the main lens redshift. The supplied example
places the additional galaxies at that redshift; use appropriate redshift information when adapting it to
real data. The `los_halos` feature instead explores simulated populations of dark halos across the line of sight.

__Masking, Noise Scaling or Explicit Modeling__

The modeling example contrasts two approaches. First, it suppresses the contribution of pixels containing
neighbouring emission. It then reloads the original dataset and explicitly fits the extra galaxies' light
and mass together with the main lens and source.

For the first approach, the example uses noise scaling rather than cutting holes out of the image mask.
Within the selected regions, data values are set to zero and noise values are made large, giving those pixels
negligible weight in the likelihood while retaining their coordinates. This can be useful for pixelized
source reconstructions, where removing regions entirely can introduce discontinuities in the sampling.
It does not recover the source information hidden in those downweighted pixels.

Explicit light modeling is useful when neighbouring emission overlaps source pixels you want to retain.
The profiles are compared with the data after PSF convolution, so their extended light contributes to the
same image as the main lens and lensed source. Including their mass additionally accounts for their deflections.
The first example omits neighbour mass; do not assume that choice is adequate for another system simply
because its light has been suppressed.

__Galaxy Centres and Model Complexity__

The data-preparation examples provide a mask of neighbouring emission and measured galaxy centres. The
modeling example fixes each extra galaxy's light and mass centres to these positions, fits its light with a
multi-Gaussian expansion, and uses a spherical isothermal mass profile.

Position constraints help prevent a profile intended for a neighbour from moving onto a lensed arc or
absorbing part of the main lens. They also reduce the number of free parameters. However, fixed centres,
profile choices and mass priors are assumptions: inspect their suitability for the data and test sensitivity
to them where they could affect the result. The parameter limits supplied in a recipe are not universal
physical limits for companion galaxies.

__Inspect the Separate Contributions__

A good fit to the combined image does not guarantee that the light has been assigned correctly to each
galaxy. Inspect individual light components, the reconstructed source and the residuals, especially where
emission overlaps. Compare reasonable treatments of the neighbours when assessing the robustness of a
scientific measurement.

Additional profiles increase model complexity and computation. The SLaM example carries the extra galaxies
through staged source, light and mass fits, fixing selected components between stages. Follow those choices
alongside the code: fixing a component does not propagate all of its earlier uncertainty into later stages.

__Choosing the Scale of the Model__

This folder is intended for a main galaxy-scale lens with a small number of neighbours. For several
co-dominant deflectors, explore the multi-galaxy examples. For group or cluster systems, use the corresponding
modeling workflows, which account for their different mass structure and companion populations.

The group examples also introduce shared luminosity scaling relations, reducing the need to fit every
companion mass independently. Such relations add population assumptions; use them because they suit the
system, rather than treating them solely as a way to make a fit faster.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/extra_galaxies/modeling.ipynb) — Compare noise scaling with explicit modeling of neighbouring light and mass, using measured galaxy centres.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/extra_galaxies/simulator.ipynb) — Generate a known lens field containing two extra galaxies and the preparation files used by the examples.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/extra_galaxies/slam.ipynb) — Carry extra galaxies through staged source, light and mass fits, including a pixelized source reconstruction.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the
workspace root.
"""
