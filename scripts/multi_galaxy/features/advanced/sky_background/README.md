The `sky_background` folder contains example scripts for fitting a multi-galaxy strong lens whose data still
contains the **sky background** — the diffuse emission from the atmosphere, zodiacal light and unresolved sources
in the field.

# What this folder is for

The sky is usually subtracted during data reduction. If that subtraction were perfect there would be nothing to
model, but it rarely is, and what is left over is a constant offset across the image.

A residual sky is flat, and the faint outskirts of a galaxy's light profile are nearly flat too. If the sky is
not in the model, the light profiles take it — extending their outskirts to cover a level that has nothing to do
with the galaxies. Fitting it costs one parameter and removes that degeneracy at the source.

The sky is a property of the **dataset**, not of anything being lensed, so it is composed as a `DatasetModel`
that sits alongside `galaxies` rather than inside it.

# The multi-galaxy difference

The sky is a single number for the whole image, and there are two bright galaxies in that image whose faint
outskirts overlap. That makes it a **shared** systematic: a mis-estimated sky does not perturb one galaxy's light
model and leave the other alone — both absorb it, in proportion to how much of the image each one's outskirts
cover. What moves is the ratio between the two galaxies' luminosities.

At galaxy scale a badly handled sky costs you one galaxy's outer profile. Here it costs you the comparison
between two.

The `simple` dataset has its sky subtracted, so `simulator.py` here writes its own: the same pair, same mass
profiles, same source, simulated with `subtract_background_sky=False`.

# Files

- `simulator`: Simulating the pair with the sky left in the data.
- `modeling`: Fitting the sky as a free `DatasetModel` parameter.
- `fit`: The same model with the sky fixed to its true value, alongside the same fit with the sky omitted.

# Related

- `multi_galaxy/modeling`: the same lens on sky-subtracted data.
- `imaging/features/advanced/sky_background`: the galaxy-scale walkthrough.
- `group/features/advanced/sky_background`: the same feature at group scale, where a wider mask gives the sky
  more pixels to be constrained by.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
