# PyAutoLens for Lenstool Users

If you model galaxy clusters with **Lenstool**, this folder shows you — with a real, published
cluster — how to do the same analysis in **PyAutoLens**, and what you gain on the other side.

The worked example is **SMACS J0723.3−7327**, the first JWST cluster, using the Lenstool model of
[Mahler et al. 2023](https://arxiv.org/abs/2207.07101) whose complete workflow (`best.par`,
`input.par`, `arcs.dat`, `galcat.cat`) is
[public](https://github.com/guillaumemahler/SMACS0723-mahler2022). Two scripts:

| Script | What it does |
|---|---|
| `data.py` | Downloads the published model files + a RELICS HST image; converts them to the PyAutoLens cluster CSV formats, recording every unit and sign convention. |
| `modeling.py` | **Reconstructs** the published best-fit (149 dPIE potentials, read straight from `best.par`), **verifies** it reproduces the observed multiple images, and composes the **refit** — the same free parameters and priors as `input.par` — for a from-scratch PyAutoLens fit you can compare with the paper's Table 3. |

## The dictionary

| Lenstool | PyAutoLens | Notes |
|---|---|---|
| `potentiel` with `profil 81` | `al.mp.dPIEMass` / `al.mp.dPIEMassSph` | Same code lineage: PyAutoLens's dPIE is ported from Lenstool's C source and validated against it numerically (`autolens_workspace_test/scripts/cluster/lenstool_parity.py`). |
| `x_centre`, `y_centre` | `centre=(y, x)` | Same relative-arcsec frame; **x is positive toward West** (see below). |
| `ellipticite` | `ellipticity` | Lenstool's mass ellipticity (a²−b²)/(a²+b²); PyAutoLens converts internally to its (1−q)/(1+q) exactly as `set_lens.c` does. |
| `angle_pos` | `angle_pos` | Degrees, counter-clockwise. |
| `core_radius`, `cut_radius` | `r_core`, `r_cut` | Arcsec. For the `_kpc` variants divide by `cosmology.kpc_per_arcsec_from(redshift=z_lens)`. |
| `v_disp` | `sigma` | **The fiducial σ_LT, not the physical central dispersion** — σ₀ = √(3/2)·σ_LT (Elíasdóttir et al. 2007, App. A). Quote `.par` values unchanged; never feed a measured stellar dispersion here. |
| `potfile` (`mag0`, `sigma`, `cutkpc`, `vdslope 4`, `slope 4`) | shared priors + derived parameters | σᵢ = σ\*·(L/L₀)^0.25, r_cut,ᵢ = r_cut\*·(L/L₀)^β, L/L₀ = 10^(0.4(mag0−m)) — the reference-anchored scaling relation. Lenstool's classic `slope 4` gives β = 0.5; `scripts/cluster/` now defaults to the modern tied exponent β = 1 + γ − 2α = 0.7 (γ = 0.2, Bergamini et al. 2019). |
| `arcs.dat` | `point_datasets.csv` → `al.list_from_csv` | One `PointDataset` per system; redshifts per system. |
| `sigposArcsec` | `positions_noise` column | Identical positional chi-squared definition. |
| source-plane optimization | `al.FitPositionsSource` | Lenstool's default likelihood. |
| image-plane optimization | `al.FitPositionsImagePair*` | The rigorous (and expensive) version; see `scripts/cluster/likelihood_function.py`. |
| `best.par` | the max-likelihood instance of the fit | And `bayes.dat` ↔ the Nautilus samples. |
| cosmology block (`H0 70`, `omegaM 0.3`) | `al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)` | Pass it everywhere — PyAutoLens defaults to Planck15. |

## Conventions verified against the data (not just documented)

- **Coordinate frame**: Lenstool's relative frame has x = −ΔRA·cos δ₀·3600 (positive West),
  y = +Δδ·3600. `data.py` verifies this by matching `galcat.cat` positions to their `potential`
  sections in `best.par` to milliarcseconds.
- **σ convention and ellipticity conversion**: unit-tested in PyAutoGalaxy against the Lenstool C
  source, and re-checked numerically by the 6-leg parity script in `autolens_workspace_test`.
- **End-to-end**: `modeling.py`'s Verification I ray-traces the 60 observed images through the
  reconstructed 149-profile tracer — every system collapses to a compact source-plane group
  (median RMS 0.07″, all below 0.29″ — consistent with the published image-plane RMS of 0.32″
  through typical magnifications), and the optional Verification II forward-solve checks the
  image-plane RMS directly. One convention this exercise pinned down the honest way: PyAutoLens's
  multi-plane tracer normalizes profile deflections to its **final** plane, so `dPIEMass`
  must be given the highest source redshift in the system as `redshift_source`.

## What's different (and why you might care)

- **Sources are sampled, not eliminated.** Lenstool removes source positions analytically inside
  its optimizer; PyAutoLens samples them (or, with `FitPositionsImagePair*`, solves the forward
  problem per likelihood call). You see the joint posterior, including source–mass covariances.
- **The likelihood is yours to choose.** Source-plane χ² reproduces Lenstool; image-plane χ² with
  proper handling of image multiplicity is one line away, and the full likelihood is documented
  step by step (`scripts/cluster/likelihood_function.py`).
- **Everything after the point-source fit is in the same framework**: extended-source (arc)
  modeling with pixelized sources, JAX acceleration of the point solver, Bayesian evidence for
  model comparison, and the cluster visualization tools (per-source-plane critical curves and
  caustics — `aplt.plot_critical_curves`) used throughout these scripts.

## Attribution

The SMACS J0723 model files are by Mahler et al. (2023), downloaded at runtime from their public
repository — cite them (and RELICS, Coe et al. 2019, for the imaging) in any work that uses this
example. The dPIE profile implementation in PyAutoLens derives from Lenstool's C code (Kneib,
Jullo et al.); see the profile docstrings for references.
