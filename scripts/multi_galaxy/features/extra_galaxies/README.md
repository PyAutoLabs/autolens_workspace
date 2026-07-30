The `extra_galaxies` folder contains example scripts showing how to add an **extra galaxies tier** to a
multi-galaxy lens model: perturbers near the co-dominant pair, carried in the model with their own light and
mass and their centres fixed to the observed light.

# Relation To The Core Scripts

The `multi_galaxy` package already handles extra galaxies one way. `multi_galaxy/simulator.py` puts a single
faint contaminant in the `simple` dataset, and `start_here`, `modeling`, `fit` and `likelihood_function` all
demonstrate the `__Extra Galaxies Noise Scaling__` step which scales its light out of the fit. That contaminant
has **no mass**, deliberately, so the arcs stay clean and every example can treat the field as a pure
two-deflector lens.

This folder is the other half. Its dataset gives the extra galaxies **mass as well as light**, and once that is
true noise scaling is not sufficient — scaling away the contaminated pixels removes the light, but the mass is
still deflecting the source. On this dataset, removing only the extra galaxies' mass changes the model image by
up to 7.6 sigma in the worst pixel and by more than 3 sigma across 226 pixels, none of them in the region the
mask covers.

# The Tier Question

So the question these scripts answer is not "how do I remove them" but **which tier does a galaxy belong in**:

- **Main lens galaxies** — free light and mass, free centres. Co-dominant deflectors which set the image
  configuration.
- **Extra galaxies** — light and mass with centres fixed and `einstein_radius` capped. Perturbers on a
  configuration the main galaxies already set. This folder.
- **Scaling galaxies** — a population whose masses follow a luminosity relation, anchored on the brightest
  co-dominant deflector so the tier costs **zero** free parameters no matter how many galaxies it holds. See
  `features/scaling_relation`.

`multi_galaxy/simulator.py` states the test: does the galaxy contribute significantly to the *lensing*, not to
the *light*? `modeling.py` here works through what each mistake actually costs, in both directions.

# Files

The following example scripts illustrate multi-galaxy lens modeling where:

- `modeling`: Lens modeling using a model which includes an extra galaxies tier with light and mass.
- `simulator`: Simulating a co-dominant pair with two massive perturbers surrounding it.

# Results

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
