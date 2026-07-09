"""
Real Data: Weak Lensing of Abell 2744
=====================================

This script fits a dark-matter halo mass model to a **real weak-lensing shape catalogue** of the merging
galaxy cluster Abell 2744 ("Pandora's Cluster", z = 0.308) — the first PyAutoLens weak-lensing analysis of
real sky data, and the capstone of the weak-lensing example series.

__The Data & Its Provenance__

The catalogue is the Abell 2744 galaxy shape catalogue shipped with the public pyRRG weak-lensing shape
measurement code (https://github.com/davidharvey1986/pyRRG, `jwst` branch), whose JWST application to this
cluster is described in Harvey & Massey 2024 (MNRAS 529, 802, arXiv:2401.16478). It contains 6,585 sources
with SExtractor photometry, RRG shape moments and per-galaxy ellipticity measurements `(e1, e2)` with
uncertainties.

**Honest framing:** the file ships inside pyRRG's star/galaxy-classification training data, and we apply our
own quality cuts below rather than the exact selections of the published analysis — so this example is a
*real-data demonstration* whose results we sanity-check against published Abell 2744 analyses, not a
metrology-grade reproduction of any single paper. That is the right expectation for a first weak-lensing
look at any new catalogue.

__What this script shows__

- Downloading and loading a real shape catalogue (FITS binary table) and converting RA/Dec to the
  tangent-plane arc-second coordinates PyAutoLens uses.
- The standard weak-lensing quality cuts every real catalogue needs.
- Building a `WeakDataset` flagged `is_reduced` — real surveys measure galaxy ellipticities, i.e. the
  *reduced* shear g = gamma / (1 - kappa), and `FitWeak` computes the matching model quantity.
- A model-independent Kaiser-Squires mass map — Abell 2744 is a famous merger, and the map's structure is
  the first sanity check.
- Fitting a spherical NFW dark-matter halo with a Nautilus search and reading the tangential-shear profile
  in which cluster weak-lensing results are usually shown.

__Contents__

- **Catalogue Download:** Fetch the public catalogue (cached on disk after the first run).
- **Catalogue Load & Projection:** RA/Dec to tangent-plane arc-seconds about the cluster centre.
- **Quality Cuts:** The standard selections that turn raw shapes into a usable shear sample.
- **Weak Dataset:** Build the reduced-shear `WeakDataset`.
- **Mass Map:** Model-independent Kaiser-Squires reconstruction.
- **Model & Search:** Spherical NFW halo, Nautilus.
- **Result:** Tangential-shear profile, and how the numbers compare to the literature.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path

import numpy as np

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Catalogue Download__

The catalogue is fetched once from the public pyRRG repository (pinned to a specific commit for
reproducibility) and cached in the dataset folder. We do not redistribute the file with the workspace —
provenance stays with the pyRRG project.
"""
dataset_path = Path("dataset") / "weak" / "a2744_pyrrg"
catalogue_path = dataset_path / "abell2744_galaxies.fits"

CATALOGUE_URL = (
    "https://raw.githubusercontent.com/davidharvey1986/pyRRG/"
    "0ccc29fb4513137da61b1afb632ca492093bd609/"
    "trainStarGalClass/TrainingData/abell2744_galaxies.fits"
)

if not catalogue_path.exists():
    import urllib.request

    dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading A2744 catalogue from pyRRG (one-off, ~3 MB) ...")
    urllib.request.urlretrieve(CATALOGUE_URL, catalogue_path)

"""
__Catalogue Load & Projection__

The table stores sky positions as RA/Dec in degrees. PyAutoLens works in tangent-plane arc-second offsets
`(y, x)` about a chosen centre, so we project about the cluster core (the catalogue's density peak,
consistent with the BCG region used by published analyses):

 - `x = (RA - RA0) * cos(Dec0) * 3600` (arc-seconds East)
 - `y = (Dec - Dec0) * 3600` (arc-seconds North)

A note on conventions: whether East points left or right on the sky is a *parity* choice that rotates or
mirrors the shear components' frame. The tangential shear — the quantity our fit constrains — is invariant
under this mirror (only the B-mode cross component flips sign), so the halo-profile fit below is robust to
it. Precision studies of shear *systematics* must track the convention carefully.
"""
from astropy.io import fits as astropy_fits

with astropy_fits.open(catalogue_path) as hdul:
    table = hdul[1].data

ra = np.asarray(table["ra"], dtype=float)
dec = np.asarray(table["dec"], dtype=float)
e1 = np.asarray(table["e1"], dtype=float)
e2 = np.asarray(table["e2"], dtype=float)
e1_err = np.asarray(table["e1_err"], dtype=float)
e2_err = np.asarray(table["e2_err"], dtype=float)

ra_centre, dec_centre = 3.5875, -30.3972  # A2744 core (J2000 degrees)

x = (ra - ra_centre) * np.cos(np.deg2rad(dec_centre)) * 3600.0
y = (dec - dec_centre) * 3600.0
radii = np.sqrt(x**2.0 + y**2.0)

print(f"catalogue sources : {len(ra)}")

"""
__Quality Cuts__

Raw shape catalogues always contain unusable measurements — blends, noise detections, objects whose moments
diverged. The cuts below are the standard minimum for any weak-lensing sample:

 - finite, physical ellipticities: |e1|, |e2| < 1 (a galaxy ellipticity cannot exceed 1; the raw catalogue
   contains outliers far beyond this from failed moment measurements).
 - measured uncertainties in a sane range: 0 < e_err < 0.4 per component.
 - a radial window 10" < r < 130": inside ~10" we are in the strong-lensing core where cluster members
   dominate and the weak-lensing (linear shear) approximation is worst; ~130" is the edge of this
   catalogue's contiguous coverage.

On this catalogue the cuts are severe: only ~1,600 of the 6,585 sources have measured shapes at all, and
the physical-ellipticity cut trims those to ~400 — this is training-data-grade depth, an order of magnitude
shallower than the selections behind published A2744 analyses. Keep that in mind when reading the results.
"""
finite = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(e1_err) & np.isfinite(e2_err)
physical = (np.abs(e1) < 1.0) & (np.abs(e2) < 1.0)
well_measured = (e1_err > 0.0) & (e1_err < 0.4) & (e2_err > 0.0) & (e2_err < 0.4)
radial = (radii > 10.0) & (radii < 130.0)

use = finite & physical & well_measured & radial

print(f"after quality cuts : {use.sum()}")

"""
__Weak Dataset__

The per-galaxy noise combines the intrinsic shape dispersion (each galaxy has a random unlensed ellipticity;
sigma_int ~ 0.25 per component is the standard value) with the measurement uncertainty, in quadrature.

`from_arrays` builds the `WeakDataset`; `is_reduced=True` (the loader default) records that these are
measured ellipticities — reduced shear — so `FitWeak` will compare them against the model's
g = gamma / (1 - kappa), not the bare shear.
"""
sigma_int = 0.25

noise = np.sqrt(sigma_int**2.0 + 0.5 * (e1_err[use] ** 2.0 + e2_err[use] ** 2.0))

dataset = al.WeakDataset.from_arrays(
    positions=np.stack([y[use], x[use]], axis=1),
    gamma_1=e1[use],
    gamma_2=e2[use],
    noise_map=list(noise),
    is_reduced=True,
    name="a2744_pyrrg",
)

print(dataset.info)

aplt.subplot_weak_dataset(dataset=dataset, output_path=dataset_path, output_format="png")

"""
__Mass Map__

Before any model is fitted, the Kaiser-Squires inversion gives a model-independent mass map. Abell 2744 is
one of the most disturbed clusters known — published lensing maps (Merten et al. 2011; Medezinski et al.
2016; Harvey & Massey 2024) show multiple substructures around the main core from an ongoing merger — so we
should *not* expect a clean single peak, and the structure in this map is the first indication the
catalogue's shear signal is real.
"""
aplt.plot_convergence_map(
    shear_yx=dataset.shear_yx,
    shape_native=(30, 30),
    smoothing_sigma_pixels=1.5,
    output_path=dataset_path,
    output_format="png",
)

"""
__Model & Search__

The model is a spherical NFW dark-matter halo — the standard first-order description of a cluster halo and
deliberately simple for a merging system (the published analyses use multiple halos; a single NFW measures
the dominant mass concentration).

The halo centre gets Gaussian priors of width 10" about the projected cluster core, and the fit assumes a
single effective source plane at z = 1.0 behind the z = 0.308 cluster (the catalogue provides no per-galaxy
redshifts; this is the standard effective-depth approximation and its choice rescales the inferred halo
normalisation).
"""
mass = af.Model(al.mp.NFWSph)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=10.0)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=10.0)

lens = af.Model(al.Galaxy, redshift=0.308, mass=mass)

source = af.Model(al.Galaxy, redshift=1.0)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

analysis = al.AnalysisWeak(dataset=dataset)

search = af.Nautilus(
    path_prefix=Path("weak") / "real_data",
    name="a2744_nfw",
    unique_tag="a2744_pyrrg",
    n_live=100,
    iterations_per_quick_update=10000,
)

print(
    """
    The non-linear search has begun running — a few hundred galaxies and a 4-parameter model,
    so expect minutes, not hours.
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

The tangential-shear profile is the standard presentation of a cluster weak-lensing measurement: binned
data (with the cross-component B-mode null test) against the maximum-likelihood NFW curve.

What this shallow sample can and cannot show — read the numbers with the sample size in mind:

 - With only ~400 usable shapes, the overall tangential-shear detection is *marginal* (a weighted mean
   gamma_t of ~0.02 at ~1.5 sigma), though it behaves exactly as a real lensing signal should: it
   concentrates at small radii (gamma_t is several times larger inside 40" than outside) and the
   cross-component B-mode is consistent with zero.
 - The NFW posterior is correspondingly broad: the halo normalisation and scale radius are each uncertain
   at the tens-of-percent-to-factors level, and the centre is only loosely pinned near the projected core.
   That *is* the honest result of fitting ~400 galaxies around one cluster — published A2744 analyses
   (Medezinski et al. 2016 quote a virial mass ~2 x 10^15 solar masses; Harvey & Massey 2024 map the
   merger's substructure) rest on samples an order of magnitude deeper with survey-grade calibration.
 - What the example therefore demonstrates is the *workflow* on real sky data — download, projection,
   quality cuts, reduced-shear dataset, model-independent map, likelihood fit, profile — which transfers
   unchanged to a deep catalogue via `WeakDataset.from_fits` / `from_csv`.
"""
print(result.info)

aplt.subplot_fit_weak(
    fit=result.max_log_likelihood_fit, output_path=dataset_path, output_format="png"
)

aplt.plot_shear_profile(
    result.max_log_likelihood_fit,
    centre=(0.0, 0.0),
    bins=8,
    output_path=dataset_path,
    output_format="png",
)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

The weak-lensing series is now complete end to end: simulator, visualization, fit, modeling, likelihood
guide, combined strong+weak analysis — and real sky data. From here:

 - Replace the single NFW with a multi-halo model (`scripts/cluster`) to chase A2744's substructures.
 - Combine this shear catalogue with the cluster's strong-lensing constraints, exactly as in
   `scripts/weak/features/strong_lensing` — the hybrid approach of Niemiec et al. 2020.
 - Swap in your own survey's catalogue via `WeakDataset.from_fits` / `from_csv`.
"""
