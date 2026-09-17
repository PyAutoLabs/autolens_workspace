The `multi_galaxy` folder contains example scripts showing how to analyse multi-galaxy strong lenses:
systems of individually galaxy-scale deflectors where **two or more galaxies of comparable mass both
contribute significantly to the lensing** of a single background source, with no dominant group- or cluster-scale dark matter halo.

# The Three Regimes Above Galaxy Scale

PyAutoLens organises lenses above the single-galaxy scale into a ladder of three regimes. Every group and
cluster is a multi-galaxy system, but not vice versa. What changes as you climb is first the mass model, then
the entire analysis strategy:

| Regime | Mass model | Source(s) | Analysis | Lens light |
|---|---|---|---|---|
| `multi_galaxy` | One free SIE/EPL per co-dominant deflector (+ shear); **no host halo, untruncated profiles** | One extended source, pixel-level reconstruction | `AnalysisImaging` (CCD pixels) | Modeled (MGE) |
| `group` | Main + extra + scaling tiers; host halo an **explicit choice** (truncated dPIE members in the Lenstool-style `group_halo` workflow) | One extended source, pixel-level reconstruction | `AnalysisImaging` (CCD pixels) | Modeled (MGE) |
| `cluster` | Host halo(s) + many truncated members on scaling relations | **Many** point sources at many redshifts, multi-plane | `AnalysisPoint` + factor graph (image positions) | Not modeled |

The multi-galaxy regime is the base rung: the only new concept relative to `imaging/` is "more than one main
lens galaxy". The extended-source analysis workflow is completely unchanged.

# Scientific Motivation

`start_here` fits **SDSS J1011+0143** (Shu et al. 2016, arXiv:1602.02927) itself: a merging pair of early-type
galaxies (~4.2 kpc separation, z=0.331) lensing a z=2.701 Lyman-alpha emitter into a ~1.8" Einstein cross. The
data are the archival HST ACS/WFC F814W imaging Shu et al. modelled (programme 10831, 2088 s, 0.05"/pixel),
bundled as a small cutout in `dataset/multi_galaxy/sdssj1011+0143/` — whose `README.md` and `prep.py` record and
reproduce its preparation from the MAST frame. Its two-SIE + shear model measured offsets of up to ~1.7 kpc
between each galaxy's mass and light — a probe of dark-matter physics only a multi-deflector model can deliver.

Every other example in this folder uses `simple`, a simulated look-alike of the same system produced by
`simulator.py`, so they run anywhere with no download and with the truth known. Other well-studied multi-galaxy
lenses include B1608+656 (two interacting deflectors, time-delay cosmography), PS J0630-1201 (five-image quasar
from a dual-SIE lens) and 2M1310-1714 (a galaxy pair inside a ~2.9" Einstein ring).

# Start Here

New users should read the `start_here` example, which gives an overview of all examples in the folder.

# Files

- `start_here`: A simple example illustrating how to analyse multi-galaxy strong lenses, fitting the real HST
  ACS/WFC F814W imaging of SDSS J1011+0143.
- `modeling`: Detailed example of performing lens modeling of a multi-galaxy strong lens.
- `simulator`: Detailed example of how to simulate a multi-galaxy strong lens; it writes the `simple` dataset
  every example in this folder other than `start_here` loads.
- `simulator_sample`: How to simulate a sample of multi-galaxy strong lenses, drawing random co-dominant pairs.
- `fit`: An anatomy of the multi-galaxy fit — how each co-dominant deflector contributes to the summed
  deflection field, and every quantity `FitImaging` computes.
- `plot`: How to plot a multi-galaxy `Imaging` dataset and a `FitImaging` fit; identical to `imaging/plot.py`.
- `source_science`: Source science calculations (total flux, magnification) behind two co-dominant deflectors.
- `likelihood_function`: A step-by-step guide of the multi-galaxy likelihood function, including the deflection
  summation which defines the regime.
- `data_preparation`: See `imaging/data_preparation`, which has all tools for preparing CCD imaging data, and
  `dataset/multi_galaxy/sdssj1011+0143/prep.py` for a worked end-to-end example turning a MAST drizzled frame
  into the files `start_here` loads; the centre-input GUI in `group/start_here` writes the
  `main_lens_centres.json` file this package loads.

# Folders

- `features`: Extensions of the multi-galaxy model — extra galaxies, scaling galaxies (untruncated), pixelized
  source reconstructions.

# Results

The `modeling` example performs lens modeling but only gives a brief overview of how to analyse and interpret
the results of a lens model fit. A full guide is given at `autolens_workspace/*/guides/results`.
