The `double_einstein_ring` folder contains example scripts for analysing **group-scale** double-source-plane
lenses (DSPLs) — strong lens systems where two source galaxies, at two different redshifts, are observed in
addition to multiple foreground main lens galaxies at the lens-plane redshift.

The group `lens_dict` model-composition API is used throughout: one `lens_i` entry per main lens galaxy centre,
loaded from a `main_lens_centres.json` file written by the simulator.

# Files

- `simulator`: Simulating a group-scale DSPL system (two main lens galaxies at z=0.5, two sources at z=1.0 and
  z=2.0).
- `fit`: Standalone `Tracer` + `FitImaging` example without invoking a non-linear search — useful for
  understanding the multi-plane API and the group `lens_dict` composition.
- `modeling`: Tutorial single-search fit using Nautilus. **This script "cheats" by initialising priors at the
  true simulator values and is not suitable for real data.** Use `chaining.py` or `slam.py` for real fits.
- `chaining`: Two chained non-linear searches — search 1 fits the main lens galaxies + `source_0` with a
  smaller mask, search 2 introduces `source_1` and `source_0`'s mass with a larger mask. This is the practical
  workflow for fitting a group DSPL.
- `slam`: Full SLaM (Source, Light and Mass) pipeline ending in pixelized source reconstructions for both
  source planes. The recommended workflow for production-quality modeling.
- `likelihood_function`: Step-by-step description of the additional likelihood-function steps specific to a
  group-scale double Einstein ring (multi-plane deflection chain across multiple main lens galaxies).

# Background

For background on the multi-plane physics, the cosmological sensitivity (`beta_01` distance ratio), and the
imaging-data API, see the single-lens-galaxy double Einstein ring example at
`autolens_workspace/scripts/imaging/features/advanced/double_einstein_ring/`.

For background on the group `lens_dict` model-composition convention, see
`autolens_workspace/scripts/group/start_here.py`.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit. A
full guide to result analysis is given at `autolens_workspace/*/guides/results`.
