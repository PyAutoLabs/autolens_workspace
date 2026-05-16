The `mass_stellar_dark` folder contains example scripts for analysing **group-scale** strong lenses where each
main lens galaxy's mass is decomposed into a stellar component (tied to its light via a mass-to-light ratio)
and a separately-parameterized dark matter halo. This decomposition is the standard tool for studying
mass-to-light variation across a group environment, since each galaxy's stellar vs dark contribution is
constrained independently by the data.

The group `lens_dict` model-composition API is used throughout: one `lens_i` entry per main lens galaxy centre,
loaded from a `main_lens_centres.json` file written by the simulator. Each `lens_i` carries an `lmp.Sersic`
bulge (light + stellar mass coupled by `mass_to_light_ratio`) and an `NFWSph` dark matter halo. An
`ExternalShear` is attached to `lens_0` only, representing the group-wide shear field.

# Files

- `simulator`: Simulating a group-scale decomposed-mass lens system (two main lens galaxies at z=0.5, one
  source at z=1.0). Each main lens galaxy has its own `lmp.Sersic` + `NFWSph`; `lens_0` additionally carries an
  `ExternalShear`.
- `fit`: Standalone `Tracer` + `FitImaging` example without invoking a non-linear search — useful for
  understanding the per-galaxy deflection decomposition and the group `lens_dict` composition. Includes an
  assertion that the hand-summed `sum_i (alpha_stellar_i + alpha_dark_i) + alpha_shear` matches the tracer's
  total deflection.
- `modeling`: Tutorial single-search fit using Nautilus. **This script "cheats" by initialising priors at the
  true simulator values and is not suitable for real data.** Use `chaining.py` or `slam.py` for real fits.
- `chaining`: Two chained non-linear searches — search 1 fits each main lens galaxy's bulge as a pure
  `lp.Sersic` light profile (no stellar-mass coupling, no dark NFW), search 2 reintroduces the stellar-mass
  coupling via `lmp.Sersic` with priors carried from search 1, adds per-galaxy `NFWSph` dark halos and the
  external shear. This is the practical workflow for fitting a group decomposed-mass lens.
- `slam`: Full SLaM (Source, Light and Mass) pipeline ending in a pixelized source reconstruction. The
  recommended workflow for production-quality modeling. Uses the `MASS_LIGHT_DARK` SLaM pipeline with the lens
  plane composed via `lens_dict` and the per-galaxy decomposition constructed manually (the canonical
  single-lens `mass_light_dark_from` helper does not support multi-galaxy lens planes).
- `likelihood_function`: Step-by-step description of the additional likelihood-function steps specific to a
  group-scale decomposed mass model — namely the per-galaxy deflection sum
  `alpha_lens = sum_i ((M/L)_i * alpha_light_i + alpha_NFW_i) + alpha_shear`.

# Background

For background on the per-galaxy decomposition mechanics (the M/L coupling, the stellar + dark deflection
sum), see the single-lens-galaxy decomposed-mass example at
`autolens_workspace/scripts/imaging/features/advanced/mass_stellar_dark/`.

For background on the group `lens_dict` model-composition convention, the GUI for locating main lens centres,
and the `main_lens_centres.json` workflow, see `autolens_workspace/scripts/group/start_here.py`.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit. A
full guide to result analysis is given at `autolens_workspace/*/guides/results`.
