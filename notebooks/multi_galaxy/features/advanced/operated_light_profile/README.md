The `operated_light_profile` folder contains example scripts for fitting a multi-galaxy strong lens where **each
co-dominant deflector hosts a compact nuclear point source** — an AGN — modelled with an operated light profile.

# What an operated profile is

An operated light profile is one assumed to have already been convolved with the PSF, so it is not convolved
again during the fit. That is what an unresolved nucleus is in the data: the recorded image of a point source is
the PSF itself, scaled by the source's brightness. Convolving it a second time smears it to roughly twice the PSF
width, which no combination of the other model parameters can undo.

# The multi-galaxy difference

Each deflector gets its own point source, and the two are independent — a pair of interacting galaxies is not
required to host equally active nuclei.

That independence is why they are worth modelling separately. Light that is not in the model does not vanish; it
is absorbed by whichever component can absorb it. With two co-dominant deflectors sitting on top of each other,
an unmodelled nucleus in one of them is absorbed asymmetrically, and what distorts is the ratio between the two
galaxies' luminosities.

The `simple` dataset's deflectors have no nuclei, so `simulator.py` here writes its own dataset: the same pair,
same mass profiles, same source, with one point source added per deflector.

# Files

- `simulator`: Simulating the pair with a nuclear point source in each deflector.
- `modeling`: Fitting an MGE plus an operated point source per deflector.

# Related

- `multi_galaxy/modeling`: the same lens without the nuclear point sources.
- `multi_galaxy/features/advanced/shapelets`: a different way of giving a deflector's light more freedom.
- `multi_galaxy/slam.py`: the SLaM baseline, whose stages already carry a `point` slot per deflector.
- `imaging/features/advanced/operated_light_profile`: the galaxy-scale walkthrough of the operated profiles.
- `group/features/advanced/operated_light_profile`: the same feature at group scale, where the tier below the
  main galaxies can carry operated profiles too.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
