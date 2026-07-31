The `advanced` folder contains example scripts for the more specialised multi-galaxy features — techniques that
are useful for particular kinds of data or particular science goals, rather than part of the default model.

Each folder changes one thing about the model of `multi_galaxy/modeling.py`, and says what that change means when
the lens has two co-dominant deflectors rather than one lens galaxy.

# Files

- `operated_light_profile`: Light profiles assumed to be already convolved with the PSF, for compact nuclear
  emission such as an AGN in either deflector.
- `shapelets`: A shapelet basis for a galaxy's light, flexible enough to describe disturbed and asymmetric
  morphology.
- `sky_background`: Fitting the uniform sky level alongside the lens model, rather than assuming it is zero.
- `double_source_plane_lens`: Two source galaxies at different redshifts behind the pair, appearing as two
  distinct Einstein rings. The second ring's images sample the deflection field where the first's do not, which
  is what constrains the mass split.
- `mass_stellar_dark`: Each deflector's mass decomposed into a stellar component and a dark halo. The two
  galaxies' mass-to-light ratios are near-degenerate, and tying them is what makes the decomposition tractable.
- `subhalo`: Detecting a dark matter subhalo. The detection floor is set by the smooth model, whose mass split
  is degenerate — so a false positive can be a mis-split rather than a perturber.

# Scope

Every folder in `group/features/advanced` is now present here. Two things in `imaging/features/advanced` are
deliberately absent: `potential_correction` and `los_halos`, which have no `group/features/advanced` equivalent
and are outside this package's scope.

Subhalo **sensitivity mapping** is likewise not covered — `subhalo` is detection only, the same boundary group
draws.

# Related

- `multi_galaxy/features`: the non-advanced feature folders, which change the default model in ways most
  multi-galaxy analyses will want.
- `multi_galaxy/modeling`: the default model every folder here starts from.
- `group/features/advanced` and `imaging/features/advanced`: the same features at group and galaxy scale, with the
  fuller API walkthroughs these scripts cross-link rather than repeat.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
