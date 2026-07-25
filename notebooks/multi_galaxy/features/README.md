The `features` folder extends the multi-galaxy lens model of `multi_galaxy/start_here` and
`multi_galaxy/modeling`, which by design uses **only main lens galaxies** — one free light and mass model per
co-dominant deflector, nothing else.

Two extensions add galaxies *below* co-dominance, using the same tiered API that becomes the default at group
and cluster scale:

- **Extra galaxies** — perturbers near (but not part of) the co-dominant system, added with restricted freedom
  (centres fixed to the observed light, free Einstein radius). See
  `imaging/features/extra_galaxies` for the worked example; the API is identical here, applied on top of N main
  galaxies instead of one.

- **Scaling galaxies** — a population of faint galaxies whose masses are tied to their luminosities through a
  scaling relation, so the model dimensionality does not grow with the population. See
  `group/features/scaling_relation` for the worked example. Two things differ at multi-galaxy scale, and the
  prose of any example you adapt should say so:

  1. **Use untruncated profiles (isothermals, not dPIE).** Truncation of a member's mass profile encodes tidal
     stripping by a host halo's potential. A multi-galaxy lens has no host halo, so there is no physical
     motivation for truncation — truncated dPIE members belong to the group and cluster regimes.
  2. **Expect the tier to be "a load of galaxies far from the lens".** With no host halo there is no bound
     member population; the scaling tier here is typically distant, individually-negligible galaxies whose
     collective contribution is a weak correction. It is supported and sometimes worthwhile, but — unlike at
     group/cluster scale — it is not a standard ingredient of the model.

Standard single-galaxy features (pixelized source reconstructions, linear light profiles, MGE variations) apply
to multi-galaxy lenses unchanged — see `imaging/features` and swap the single lens galaxy for the `lens_0`,
`lens_1`, ... loop of this package.
