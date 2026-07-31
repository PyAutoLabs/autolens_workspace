The `mass_stellar_dark` folder contains example scripts for fitting a multi-galaxy strong lens where **each
co-dominant deflector's mass is decomposed into a stellar component and a dark matter halo**, rather than
described by a single total mass profile.

# What this folder is for

A total mass profile measures how much mass sits inside the Einstein radius and nothing about what it is made
of. A decomposition measures the stellar and dark components separately, which is what a stellar-mass estimate,
an initial mass function constraint or a halo concentration needs.

The stellar component is an `lmp.Sersic` — a light **and** mass profile, where the same `intensity`,
`effective_radius` and `sersic_index` produce both the light you subtract and the mass you deflect with, tied
together by a `mass_to_light_ratio`. The dark component is an `NFWSph` halo with no light.

# The choice this folder is about

With one lens galaxy the mass-to-light ratio is one parameter and there is nothing to decide. With two
co-dominant deflectors it is a modelling choice: fit a ratio per galaxy, or tie them to a single shared value.

**Tying is what makes the decomposition tractable here.** The two galaxies' ratios are near-degenerate with each
other — stellar mass can be moved from one galaxy to the other while the total deflection barely changes, which
is the multi-galaxy mass-split degeneracy reappearing inside the stellar component. A shared ratio removes that
direction from the parameter space rather than leaving the search to find its way along it.

The cost is an assumption: that both galaxies have the same stellar populations. For an interacting pair of
early-types that is defensible. For a pair with visibly different colours it is not, and then you fit the ratios
separately and accept the wider posteriors.

The dark halos are never tied. A shared stellar population is a defensible assumption; equal dark masses is
close to asserting the mass split this regime exists to measure.

`simulator.py` gives the two galaxies **different** mass-to-light ratios on purpose, so the tied model has
something to be wrong about.

# Files

- `simulator`: Simulating the pair with decomposed stellar and dark mass.
- `modeling`: Fitting the decomposition, with the ratios tied — and how to untie them.
- `fit`: The decomposed deflection field, component by component, and how the components separate with radius.
- `likelihood_function`: The decomposed likelihood step by step.
- `chaining`: A total mass model first, the decomposition second.
- `slam`: The full pipeline, whose terminal stage is `MASS LIGHT DARK` rather than `MASS TOTAL`.

# Related

- `multi_galaxy/modeling`: the total-mass multi-galaxy lens these scripts decompose.
- `multi_galaxy/slam.py`: the SLaM baseline, whose `MASS TOTAL` stage this folder's `slam` replaces.
- `imaging/features/advanced/mass_stellar_dark`: the single-deflector decomposition.
- `group/features/advanced/mass_stellar_dark`: the same feature at group scale.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
