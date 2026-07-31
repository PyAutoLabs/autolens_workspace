The `shapelets` folder contains example scripts for fitting a multi-galaxy strong lens where the **source** is
reconstructed with a basis of shapelets — orthogonal basis functions well suited to irregular, asymmetric
morphology.

Shapelets are described in full in Refregier (2003), MNRAS 338, 35 (arXiv:astro-ph/0105178).

# What this folder is for

A source that is not a smooth ellipse cannot be described by a Sersic, and the residuals a Sersic leaves are
absorbed by the mass model. A shapelet basis can describe clumps and asymmetry, and every shapelet's intensity is
solved by linear algebra — so the extra flexibility costs no non-linear parameters. The whole basis shares one
centre, one ellipticity and one `beta`, which is why a basis of tens of shapelets costs four sampled parameters
between them.

Shapelets need the linear solve to allow negative intensities, which is what
`al.Settings(use_positive_only_solver=False)` does in both scripts. Orders above the first are negative over part
of their extent; that is how a sum of them describes anything other than a bump.

# Why the deflectors stay MGE

The multi-galaxy regime is populated by interacting systems, and tidally disturbed light is the norm rather than
the exception — a natural argument for putting a flexible basis on the *deflectors* too. The API supports it, and
these scripts do not do it.

`imaging/features/advanced/shapelets/modeling.py` says why under its own `__Lens Shapelets__` section: the model
is not established in the literature, and for massive early-type galaxies — which both deflectors here are — an
MGE is faster and gives better results. The disturbed-deflector case is answered by
`multi_galaxy/features/multi_gaussian_expansion`, whose dataset has exactly that twisted, two-component light.

# Files

- `modeling`: Fitting a shapelet source with a non-linear search.
- `fit`: The same composition without a search, where the basis and its solved intensities can be inspected.

Both reuse the `simple` dataset — the shapelets describe the source, and `simple`'s source is the one every other
script in the package fits.

# Related

- `multi_galaxy/modeling`: the same lens with an MGE source.
- `multi_galaxy/features/pixelization`: a free-form source mesh, for structure a centred basis cannot describe.
- `multi_galaxy/features/multi_gaussian_expansion`: the answer to disturbed deflector morphology.
- `imaging/features/advanced/shapelets`: the galaxy-scale walkthrough, including the Cartesian basis and the
  basis-regularization options.
- `group/features/advanced/shapelets`: the same feature at group scale.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
