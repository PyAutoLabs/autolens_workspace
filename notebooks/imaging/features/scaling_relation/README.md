The `scaling_relation` folder contains example scripts showing how to include a large population of foreground
galaxies in a lens model by tying their masses to the main lens galaxy's, rather than freeing each one.

Each member's Einstein radius follows a Faber-Jackson relation anchored on the main lens:

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

`einstein_radius_anchor` is the main lens's own `einstein_radius`, which the model already fits, so the tier adds
**zero free parameters** no matter how many galaxies it holds. Pair it with the `extra_galaxies` tier for the
brighter companions that do warrant individual freedom.

The mass profiles here are **untruncated**: truncation encodes tidal stripping by a host halo's potential, and a
galaxy-scale lens has no host halo. The truncated `dPIEMass` form of this tier belongs to the group- and
cluster-scale workflows (`group/features/group_halo`, `cluster/modeling`), where a host potential does exist.

# Files

- `simulator`: Simulating a lens with two tiers of companions, with truth masses derived from the relation.
- `modeling`: Lens modeling with the tier tied to the main lens, assuming measured luminosities.
- `fit`: The same composition without a search, showing the per-galaxy deflection sum.
- `likelihood_function`: The one step of the likelihood a scaling relation changes.
- `slam`: The Source, Light and Mass pipeline — **where the luminosities are measured**.

The first four scripts take luminosities as given, because that is what makes them readable. `slam` is the
production path: its light stage fits an MGE to every galaxy and integrates it to a luminosity.

# Related

- `imaging/features/extra_galaxies`: companions modelled with full individual freedom.
- `multi_galaxy/features/scaling_relation`: the same relation with the anchor chosen as the brightest of several
  co-dominant deflectors.
- `group/features/scaling_relation` and `cluster/modeling`: the reference-magnitude (Lenstool `mag0`)
  normalisation, which costs one free parameter and does not assume a single anchoring galaxy.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
