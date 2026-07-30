The `scaling_relation` folder contains example scripts showing how to add a population of faint, distant galaxies to
a multi-galaxy lens model by tying their masses to the brightest co-dominant deflector, rather than freeing
each one.

    einstein_radius_i = einstein_radius_brightest * (L_i / L_brightest) ** 0.5

`einstein_radius_brightest` is the brightest galaxy's own `einstein_radius`, which the model already fits, so the tier
adds **zero free parameters** however many galaxies it holds.

The anchor is called simply "the brightest galaxy" throughout these scripts. A multi-galaxy lens is a handful of
co-dominant deflectors, not a bound system, so there is no *brightest cluster galaxy* to speak of. At group and
cluster scale the same anchor is the BCG (brightest cluster galaxy) or BGG (brightest group galaxy) — see
`group/features/scaling_relation`, which uses that terminology.

Two things distinguish this tier at multi-galaxy scale:

1. **The anchor has to be identified, not assumed.** With two or more co-dominant deflectors the relation needs the
   brightest, found by `argmax` over the measured luminosities — not whichever galaxy is listed first in
   `main_lens_centres.json`.

2. **The tier is not a standard ingredient here.** With no host halo there is no bound member population, so this is
   "a load of galaxies far from the lens" — a percent-level shear correction. It is supported and sometimes
   worthwhile, but fit without it first. At group and cluster scale the same tier becomes standard, and tidally
   truncated.

Mass profiles here are **untruncated**: truncation encodes tidal stripping by a host halo's potential, which a
multi-galaxy lens lacks by definition. The truncated `dPIEMass` variant belongs to `group/features/group_halo` and is
the cluster-scale default.

# Files

- `simulator`: Simulating the co-dominant pair plus five distant tied galaxies, truths derived from the relation.
- `modeling`: Lens modeling with the tier tied to the brightest galaxy, assuming measured luminosities.
- `fit`: The same composition without a search, including how much the tier actually contributes.
- `likelihood_function`: The one step of the likelihood the tier changes.
- `slam`: The Source, Light and Mass pipeline — **where the luminosities are measured**, using two masks.

The first four scripts take luminosities as given, because that is what makes them readable. `slam` is the production
path, and at this scale the measurement needs a second, enlarged mask: the tier sits outside the mask used to fit the
lensed source, and you cannot measure the luminosity of a galaxy you have masked away.

# Related

- `imaging/features/scaling_relation`: the fuller walkthrough of the relation, with a single-lens anchor.
- `group/features/scaling_relation`: the reference-magnitude (Lenstool `mag0`) normalisation, which costs one free
  parameter and does not assume a single anchoring galaxy.
- `multi_galaxy/modeling`: the co-dominant-pair model this extends.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
