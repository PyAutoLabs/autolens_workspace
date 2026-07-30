The `features` folder extends the multi-galaxy lens model of `multi_galaxy/start_here` and
`multi_galaxy/modeling`, which by design uses **only main lens galaxies** — one free light and mass model per
co-dominant deflector, nothing else.

Two extensions add galaxies *below* co-dominance, using the same tiered API that becomes the default at group
and cluster scale:

- **Extra galaxies** — perturbers near (but not part of) the co-dominant system, added with restricted freedom
  (centres fixed to the observed light, free but capped Einstein radius). The worked example lives here, in
  `features/extra_galaxies` (simulator + modeling); `imaging/features/extra_galaxies` gives the fuller API
  walkthrough at galaxy scale, including the SLaM variant.

  Note the division of labour with the core scripts. `multi_galaxy/simulator.py` puts a **massless** contaminant
  in the `simple` dataset and `start_here` / `modeling` / `fit` / `likelihood_function` demonstrate the
  `__Extra Galaxies Noise Scaling__` step which scales its light out of the fit. That is the right lever when
  light is the only problem. The example here gives its perturbers **mass as well as light**, where noise
  scaling is no longer sufficient — on that dataset, removing only the extra galaxies' mass shifts the model
  image by up to 7.6 sigma, none of it in the pixels the mask covers. What is left for the model to decide is
  the tier question: co-dominant deflector, perturber, or scaling-relation member.

- **Scaling galaxies** — a population of faint galaxies whose masses are tied to their luminosities through a
  scaling relation, so the model dimensionality does not grow with the population. The worked example lives
  here, in `features/scaling_galaxies` (simulator + modeling), using the untruncated-isothermal relation
  (`einstein_radius ∝ L^0.5`); `group/features/scaling_relation` gives the fuller API walkthrough. Two framing
  points matter here, and both examples' prose says so:

  1. **Untruncated profiles are the physically right choice at this scale, not a simplification.** Truncation
     of a member's mass profile encodes tidal stripping by a host halo's potential. A multi-galaxy lens has no
     host halo, so there is no physical motivation for truncation. (The truncated dPIE variant of the scaling
     tier — physically motivated where a host potential exists — appears in the group regime's Lenstool-style
     workflow, `group/features/group_halo`, and is the default at cluster scale.)
  2. **Expect the tier to be "a load of galaxies far from the lens".** With no host halo there is no bound
     member population; the scaling tier here is typically distant, individually-negligible galaxies whose
     collective contribution is a weak correction. It is supported and sometimes worthwhile, but — unlike at
     group/cluster scale — it is not a standard ingredient of the model.

Standard single-galaxy features (pixelized source reconstructions, linear light profiles, MGE variations) apply
to multi-galaxy lenses unchanged — see `imaging/features` and swap the single lens galaxy for the `lens_0`,
`lens_1`, ... loop of this package.
