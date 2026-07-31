The `double_source_plane_lens` folder contains example scripts for fitting a multi-galaxy strong lens with **two
source galaxies at different redshifts** behind the same pair of co-dominant deflectors. They appear as two
distinct Einstein rings in the image plane.

# What this folder is for

A double source-plane lens (DSPL) is usually sought out for cosmology: the ratio of angular diameter distances
between the deflector and the two sources depends on the cosmological model, and a DSPL measures that ratio
directly.

In the multi-galaxy regime it does something else as well, and the two should not be confused.

# The multi-galaxy difference

The regime's standing problem is that the data constrains the *total* deflection of the pair well and the
*split* between the two deflectors much less well. A second source plane helps with that split — but not
because of the extra redshift.

Both sources sit behind the same pair, so the deflection field the second source sees is the first source's
field scaled by a geometric factor. A scaling says nothing about how mass is divided between the two galaxies,
and the degeneracy scales straight through it.

What helps is that the second source sits at a **different sky position**, so its ring's images land elsewhere
in the image plane. The mass split is a statement about the spatial structure of the deflection field, and a
second set of images measures that field where the first source's images do not reach.

This is why `simulator.py` offsets the second source rather than placing it behind the first, why `modeling.py`
insists the mask contain both rings, and why `slam.py`'s mass stage is the one that pays off.

# Files

- `simulator`: Simulating the pair with a second source plane behind the first.
- `modeling`: Fitting both source planes in one search — a tutorial that cheats on its priors.
- `fit`: The multi-plane tracer inspected directly, without a search.
- `likelihood_function`: The DSPL likelihood step by step, including both traces.
- `chaining`: The honest two-search version, with a tight mask for the first ring and a full mask for both.
- `slam`: The full pipeline — six stages rather than the baseline's five.

For real data use `chaining` or `slam`. `modeling` initialises its priors at the true simulator values and is a
tutorial only.

# Related

- `multi_galaxy/modeling`: the two-plane multi-galaxy lens these scripts extend.
- `multi_galaxy/slam.py`: the SLaM baseline, whose stages this folder's `slam` diffs against.
- `imaging/features/advanced/double_source_plane_lens`: the single-deflector DSPL walkthrough, with the fuller
  multi-plane ray-tracing API tour.
- `group/features/advanced/double_source_plane_lens`: the same feature at group scale.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
