The `features` folder extends the multi-galaxy lens model of `multi_galaxy/start_here` and
`multi_galaxy/modeling`, which by design uses **only main lens galaxies** — one free light and mass model per
co-dominant deflector, an MGE for each, and nothing else.

Each folder here changes one thing about that model, and says what changes when the lens has two co-dominant
deflectors rather than one lens galaxy.

# Model variations

- **`linear_light_profiles`** — a single linear `Sersic` per galaxy instead of an MGE, isolating the linear
  intensity solve (an MGE is already built from linear profiles, so this is not introducing the feature — it is
  showing it without the MGE composition API in the way). The regime content is the curvature matrix: the
  strongest coupling in the linear system is between the two **deflectors** (0.296) rather than between either
  and the source (0.135, 0.113), so their intensities — and the flux ratio between them — are solved jointly and
  are sensitive to each other's shape parameters. Mis-specifying one galaxy's `effective_radius` moves the
  *other* galaxy's solved intensity by 5.2%.

- **`no_lens_light`** — the co-dominant deflectors have no visible light. At galaxy scale this is close to a free
  win. Here it costs you the deflectors' positions: there are two of them, neither at the origin, and with the
  light gone nothing in the image marks where they are. The mass centres become model parameters informed from
  outside the image, so the parameter saving is 4 rather than the 8 the light removed — and the 4 that came back
  are degenerate with the mass split.

- **`multi_gaussian_expansion`** — one MGE basis per co-dominant deflector, with the over-sampling applied at
  every deflector's centre. The MGE is already the default in `multi_galaxy/modeling`, so this folder is not
  introducing it; it is the variations walkthrough, on a dataset whose deflectors have twisted, two-component
  light.

- **`pixelization`** — the source reconstructed on a mesh of pixels solved by linear algebra, instead of an
  analytic profile. Covers the rectangular and Delaunay meshes, the adaptive mesh and regularization schemes,
  the CPU sparse-operator route, and the SLaM pipeline's pixelization choices.

# Galaxy tiers

The core scripts have no galaxy tiers — every deflector is co-dominant. Two extensions add galaxies *below*
co-dominance, using the same tiered API that becomes the default at group and cluster scale:

- **`extra_galaxies`** — perturbers near (but not part of) the co-dominant system, added with restricted freedom
  (centres fixed to the observed light, free but capped Einstein radius), plus the SLaM pipeline that schedules
  that freedom across the stages.

  Note the division of labour with the core scripts. `multi_galaxy/simulator.py` puts a **massless** contaminant
  in the `simple` dataset and `start_here` / `modeling` / `fit` / `likelihood_function` demonstrate the
  `__Extra Galaxies Noise Scaling__` step which scales its light out of the fit. That is the right lever when
  light is the only problem. The example here gives its perturbers **mass as well as light**, where noise
  scaling is no longer sufficient — on that dataset, removing only the extra galaxies' mass shifts the model
  image by up to 7.6 sigma, none of it in the pixels the mask covers. What is left for the model to decide is
  the tier question: co-dominant deflector, perturber, or scaling-relation member.

- **`scaling_relation`** — a population of faint galaxies whose masses are tied to their luminosities through a
  scaling relation, so the model dimensionality does not grow with the population. Uses the
  untruncated-isothermal relation (`einstein_radius ∝ L^0.5`) **anchored on the brightest co-dominant deflector**,
  which makes the whole tier cost zero free parameters. The anchor is called simply "the brightest galaxy" at this
  scale — a multi-galaxy lens is not a bound system, so there is no brightest *cluster* or *group* galaxy to speak
  of; `group/features/scaling_relation` uses the BCG/BGG terminology.

  `imaging/features/scaling_relation` gives the fuller walkthrough with a single-lens anchor, and
  `group/features/scaling_relation` the reference-magnitude (Lenstool `mag0`) alternative that costs one parameter
  instead. Two framing points matter here, and both examples' prose says so:

  1. **Untruncated profiles are the physically right choice at this scale, not a simplification.** Truncation
     of a member's mass profile encodes tidal stripping by a host halo's potential. A multi-galaxy lens has no
     host halo, so there is no physical motivation for truncation. (The truncated dPIE variant of the scaling
     tier — physically motivated where a host potential exists — appears in the group regime's Lenstool-style
     workflow, `group/features/group_halo`, and is the default at cluster scale.)
  2. **Expect the tier to be "a load of galaxies far from the lens".** With no host halo there is no bound
     member population; the scaling tier here is typically distant, individually-negligible galaxies whose
     collective contribution is a weak correction. It is supported and sometimes worthwhile, but — unlike at
     group/cluster scale — it is not a standard ingredient of the model.

# Why there is no `group_halo` here

`group/features/group_halo` is the group regime's signature tutorial: whether a lens model includes a group-scale
dark-matter halo is an explicit modelling choice, tested by fitting the same data with and without it.

There is no analogue at this scale, and the absence is definitional rather than an omission. A multi-galaxy lens
is *defined* as a system of co-dominant deflectors with **no** dominant host halo — that is precisely what
separates this rung of the ladder from `group/`. If your system does have one, you are looking at a group, and
`group/` is the package you want.

The same fact propagates through the rest of the model, which is why it is worth stating rather than leaving
implicit: it is why the scaling tier above is untruncated, and why truncated `dPIEMass` members belong to
`group/features/group_halo` and are the cluster-scale default.

# SLaM

`multi_galaxy/slam.py` is the regime's SLaM baseline — the five-stage production pipeline for a multi-galaxy lens,
generalized over any number of co-dominant deflectors. Each feature folder's `slam.py` documents only its
difference from that script, which in turn documents only its difference from `guides/modeling/slam_start_here`.

`imaging/` has no top-level `slam.py`, because at galaxy scale the guide's composition is already the one you
want. Here it is not: every stage loops over deflectors, carries a separate `shear_galaxy`, anchors mass centres
before releasing them, and scales its live points with the deflector count.

# Advanced

`advanced/` holds the more specialised features — useful for particular kinds of data or particular science
goals, rather than part of the default model:

- **`advanced/operated_light_profile`** — a PSF-convolved point source per deflector, for a lens whose galaxies
  host AGN. An unmodelled nucleus in one of two deflectors is absorbed asymmetrically by the two light models.

- **`advanced/shapelets`** — a shapelet basis for the source, describing clumps and asymmetry at the cost of no
  extra non-linear parameters. The deflectors keep their MGEs, for the reason that folder's README gives.

- **`advanced/sky_background`** — the residual sky fitted as a `DatasetModel` rather than assumed zero. One sky
  across an image with two bright galaxies is a shared systematic that both light models absorb.

- **`advanced/double_source_plane_lens`** — two source galaxies at different redshifts, appearing as two
  distinct Einstein rings. The second ring helps with the mass split, but because its images sample the
  deflection field at different sky positions, not because of the extra redshift — the extra redshift is what
  constrains cosmology.

- **`advanced/mass_stellar_dark`** — each deflector's mass decomposed into stellar and dark components. With
  two galaxies the mass-to-light ratios are near-degenerate with each other, so tying them across the pair is
  what makes the decomposition tractable.

- **`advanced/subhalo`** — detecting a dark matter subhalo. Detection is a comparison against the smooth
  model, and this regime's smooth model has a degenerate mass split — which produces residuals of the same
  character as a perturber, so the comparison model has to carry every deflector.

See `advanced/README.md` for the full inventory.

# Scope

This package now covers every folder in `group/features` except `group_halo`, which has no analogue at this
scale for the reason given above, plus `extra_galaxies` and `scaling_relation`.

`potential_correction` and `los_halos` exist only in `imaging/features/advanced` and are deliberately not part
of this package.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
