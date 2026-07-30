The `linear_light_profiles` folder contains example scripts for fitting a multi-galaxy strong lens where each
light profile's `intensity` is solved analytically by linear algebra rather than sampled by the non-linear search.

# What This Folder Is For

`multi_galaxy/modeling.py` already uses linear light profiles — its MGE bases are built from `lp_linear`
Gaussians. So this folder is not introducing the feature to the package. It isolates it, using a single
`lp_linear.Sersic` per galaxy so the linear solve is visible without the MGE composition API in the way.

# The multi-galaxy difference

At galaxy scale, linear light profiles are described as a dimensionality win: `intensity` leaves the search, and
its degeneracy with `effective_radius` and `sersic_index` goes with it. For the model fitted here that is 29 free
parameters down to 26 — a real but modest saving.

What makes the feature worth its own folder in this regime is the **coupling between the two deflectors**. The
linear system solves for all intensities at once, and its curvature matrix `F` measures how much each profile's
light overlaps every other's. Measured on the `simple` dataset at full resolution:

    normalized F = [[1.0000  0.2955  0.1353]     lens_0
                    [0.2955  1.0000  0.1130]     lens_1
                    [0.1353  0.1130  1.0000]]    source

The largest off-diagonal term in the whole system is **0.296, between the two lens galaxies** — more than twice
either galaxy's coupling to the source. The solve is not mostly separating lens light from source light, as the
galaxy-scale picture suggests. It is mostly separating the two deflectors from each other.

That has a measurable consequence, which `fit.py` demonstrates. Mis-specify `lens_0`'s `effective_radius` (0.6 →
0.8) and leave `lens_1` untouched, and `lens_1`'s solved intensity still moves by 5.2% — the flux ratio between
the two galaxies goes from 1.20 to 0.81, a 33% error, in the galaxy whose model was never wrong.

The flux ratio is frequently the measurement. This is why `multi_galaxy/slam.py` treats its lens-light stage as
load-bearing rather than cosmetic, and it has no galaxy-scale equivalent: with one lens galaxy there is nothing
for the solver to redistribute flux to.

# Files

- `modeling`: Fitting the model with a linear `Sersic` per deflector, and extracting the solved intensities.
- `fit`: The same composition without a search, where the solve can be checked against the simulator's truth
  (it recovers 1.2 and 1.0 to within 0.1%) and its sensitivity to shape measured.
- `likelihood_function`: The step of the likelihood the linear solve replaces, and where the 0.296 above comes
  from.
- `slam`: The Source, Light and Mass pipeline using linear Sersics instead of MGEs — and when that is, and is not,
  the right choice.

# Caveat

A single `Sersic` per galaxy is not the light model to fit to real multi-galaxy data. The systems this regime
describes are often interacting pairs, whose disturbed morphology a symmetric profile cannot represent, and whose
residuals land where the two deflectors are most strongly coupled. Use `multi_gaussian_expansion` in production;
use this folder to understand what the solve is doing.

# Related

- `multi_galaxy/modeling`: the package's default model, which already uses linear profiles via an MGE.
- `imaging/features/multi_gaussian_expansion`: many linear profiles per galaxy.
- `imaging/features/linear_light_profiles`: the galaxy-scale walkthrough, with the fuller API tour and the full
  derivation of the inversion matrices.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
