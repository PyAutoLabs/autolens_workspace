The `multi_gaussian_expansion` folder contains example scripts for fitting a multi-galaxy strong lens where **each
co-dominant deflector has its own MGE** — a basis of ~10-30 Gaussians whose intensities are solved by linear
algebra.

# What this folder is for

`multi_galaxy/modeling.py` already fits an MGE. It is the package default, composed through the
`al.model_util.mge_model_from` helper which deliberately hides the API. So this folder is not introducing the
feature. It does three things the core script does not:

1. **Fits a dataset that needs one.** `simulator.py` here writes a dataset whose deflectors each have two offset,
   differently-rotated light components — isophotal twists and a radially varying ellipticity that a single
   elliptical Sersic provably cannot represent. The `simple` dataset's deflectors are single Sersics, so an MGE
   fitted to it has nothing to demonstrate.
2. **Shows the basis-composition API** the helper hides, so you can vary it.
3. **Documents what two MGEs in one model do to the linear system** — the regime-specific part.

The dataset choice is not contrived. This regime is populated by interacting systems; the package's reference
pair, SDSS J1011+0143, is a merger, and tidally disturbed morphology is the norm.

# The measurement that justifies the MGE

At fixed light centres, without a non-linear search:

| Light model per deflector | log likelihood |
|---|---|
| single linear `Sersic`, at each galaxy's **true bulge shape** | ~ −289,000 |
| MGE, 10 Gaussians | ~ −4,600 |
| MGE, 20 Gaussians | ~ −4,490 |
| MGE, 30 Gaussians | ~ −4,480 |
| the simulator's truth tracer | ~ +28,000 |

Two significant figures, because the simulator adds unseeded Poisson noise — every value moves a percent or two on
re-simulation. The *gaps* do not. The Sersic is given every advantage (its centre, axis ratio, position angle,
effective radius and Sersic index are the true values of that galaxy's bulge) and is still ~284,000 worse than a
10-Gaussian MGE.

Twenty Gaussians is the right basis size here: 10 → 20 gains ~130, 20 → 30 gains ~8, which is smaller than the
scatter between two re-simulations.

# The multi-galaxy difference

Giving each galaxy more freedom to describe its own light also gives the pair more freedom to trade light between
them. With 20 Gaussians per deflector the curvature matrix `F` is 41 × 41, and its normalized couplings are:

| Block | mean \|C\| | max \|C\| |
|---|---|---|
| within one deflector's basis | 0.459 | 1.0000 |
| **between the two deflectors** | 0.119 | **0.9877** |
| deflector to source | 0.098 | 0.384 |

Unlike the log likelihoods, these are stable to four decimal places across re-simulations — `F` depends on the
model geometry and the noise map, not the noise draw.

The within-basis correlation of 1.0000 is expected and harmless: nobody interprets an individual Gaussian, only
their sum. The **0.9877 between the two galaxies is not harmless**, because that sum *is* interpreted — it is each
galaxy's luminosity, and the ratio between them is frequently the measurement.

Compare the single-profile case in `multi_galaxy/features/linear_light_profiles`, which measures the same coupling
at **0.296**. Twenty profiles per galaxy gave the pair 400 ways to trade light.

The practical consequence: `F`'s condition number is ~1e24, and solving the system with a naive positive-negative
solver returns **21 of 41 intensities negative**. That is the "ringing" PyAutoLens's positive-only solver exists to
prevent — and at this scale it would not merely be unphysical within one galaxy, it would move flux between two
galaxies whose relative brightness someone intends to quote.

# Files

- `simulator`: Simulating the pair with twisted, two-component light.
- `modeling`: Fitting one MGE per deflector, and choosing the basis size.
- `fit`: The same composition without a search — where the Sersic comparison and the basis-size scan are run.
- `likelihood_function`: The 41 × 41 curvature matrix and its block structure.
- `source_science`: Integrating each basis to a luminosity, the flux ratio, and the source's magnification.
- `slam`: The Source, Light and Mass pipeline on this dataset.

# Related

- `multi_galaxy/modeling`: the package default, which already uses an MGE.
- `multi_galaxy/features/linear_light_profiles`: one profile per galaxy, and where the 0.296 comes from.
- `multi_galaxy/slam.py`: the SLaM baseline, whose MGE stages this folder's dataset motivates.
- `imaging/features/multi_gaussian_expansion`: the galaxy-scale walkthrough, with the fuller API tour.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
