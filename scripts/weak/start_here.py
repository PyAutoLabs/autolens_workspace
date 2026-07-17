"""
Start Here: Weak Lensing
========================

Gravitational lensing does not stop at the spectacular arcs and multiple images of strong lensing.
Every galaxy behind a massive structure is slightly sheared by its gravity — a percent-level
distortion invisible in any single galaxy but measurable statistically across thousands. This is
**weak lensing**, and it traces mass on scales far beyond the strong-lensing core.

This script shows you how to fit a weak-lensing shear catalogue using **PyAutoLens** with as little
setup as possible. In about 15 minutes you'll be able to point the code at your own catalogue and
fit your first dark-matter halo.

We model **real data**: the galaxy shape catalogue of **Abell 2744** ("Pandora's Cluster",
z = 0.308), one of the most massive and disturbed merging clusters known. The catalogue ships with
the public pyRRG weak-lensing shape measurement code (Harvey & Massey 2024, MNRAS 529, 802), whose
JWST application to this cluster it supports. It contains 6,585 sources with per-galaxy ellipticity
measurements `(e1, e2)` and uncertainties.

The workflow below — download, projection to arc-second coordinates, quality cuts, reduced-shear
dataset, model-independent mass map, likelihood fit, tangential-shear profile — transfers unchanged
to any survey catalogue.

__Contents__

- **JAX:** JAX acceleration for fast GPU/CPU model-fitting.
- **Google Colab Setup:** The introduction `start_here` examples are available on Google Colab, which allows you to run them.
- **Imports:** Import the required Python libraries.
- **Catalogue Download:** Fetch the public A2744 catalogue (cached on disk after the first run).
- **Catalogue Load & Projection:** RA/Dec to tangent-plane arc-seconds about the cluster centre.
- **Quality Cuts:** The standard selections that turn raw shapes into a usable shear sample.
- **Weak Dataset:** Build the reduced-shear `WeakDataset`.
- **Mass Map:** Model-independent Kaiser-Squires reconstruction.
- **Model:** Compose the dark-matter halo model fitted to the data.
- **Model Fit:** Perform the model-fit using the search and analysis.
- **Result:** The tangential-shear profile and how to read it.
- **Model Your Own Catalogue:** Adapting the script to your own survey data.
- **Simulator:** Simulating your own weak-lensing shear catalogues.
- **Wrap Up:** Summary of the script and next steps.

__JAX__

PyAutoLens can run weak-lensing model-fits on JAX via `al.AnalysisWeak(dataset, use_jax=True)`,
where the search driver wraps the likelihood in `jax.vmap(jax.jit(...))` so batches of parameter
vectors evaluate in parallel on a GPU. This example keeps the default NumPy path: a weak-lensing
likelihood — a few hundred to a few hundred thousand shear residuals — is light compared to
imaging, so this fit completes in minutes even on CPU (and visualization of JAX-path weak fits is
currently blocked by PyAutoLens#614).

For the broader JAX principles (when you write `@jax.jit` yourself, the return-type contract, how
to opt out for debugging), see the `__JAX__` section of the top-level
`autolens_workspace/start_here.py`.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoLens installation.

The code below sets up your environment if you are using Google Colab, including installing autolens and downloading
files required to run the notebook. If you are running this script not in Colab (e.g. locally on your own computer),
running the code will still check correctly that your environment is set up and ready to go.
"""

import subprocess
import sys

try:
    import google.colab

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autoconf", "--no-deps"]
    )
except ImportError:
    pass

from autoconf import setup_colab

setup_colab.for_autolens(
    raise_error_if_not_gpu=False  # Switch to False for CPU Google Colab
)

"""
__Imports__

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Catalogue Download__

The catalogue is fetched once from the public pyRRG repository (pinned to a specific commit for
reproducibility) and cached in the dataset folder. We do not redistribute the file with the
workspace — provenance stays with the pyRRG project.
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

The table stores sky positions as RA/Dec in degrees. PyAutoLens works in tangent-plane arc-second
offsets `(y, x)` about a chosen centre, so we project about the cluster core (the catalogue's
density peak, consistent with the BCG region used by published analyses):

 - `x = (RA - RA0) * cos(Dec0) * 3600` (arc-seconds East)
 - `y = (Dec - Dec0) * 3600` (arc-seconds North)

A note on conventions: whether East points left or right on the sky is a *parity* choice that
rotates or mirrors the shear components' frame. The tangential shear — the quantity our fit
constrains — is invariant under this mirror (only the B-mode cross component flips sign), so the
halo-profile fit below is robust to it. Precision studies of shear *systematics* must track the
convention carefully.
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

Raw shape catalogues always contain unusable measurements — blends, noise detections, objects whose
moments diverged. The cuts below are the standard minimum for any weak-lensing sample:

 - finite, physical ellipticities: |e1|, |e2| < 1 (a galaxy ellipticity cannot exceed 1; the raw
   catalogue contains outliers far beyond this from failed moment measurements).
 - measured uncertainties in a sane range: 0 < e_err < 0.4 per component.
 - a radial window 10" < r < 130": inside ~10" we are in the strong-lensing core where cluster
   members dominate and the weak-lensing (linear shear) approximation is worst; ~130" is the edge
   of this catalogue's contiguous coverage.

On this catalogue the cuts are severe: only ~1,600 of the 6,585 sources have measured shapes at
all, and the physical-ellipticity cut trims those to ~400 — this is training-data-grade depth, an
order of magnitude shallower than the selections behind published A2744 analyses. Keep that in
mind when reading the results.
"""
finite = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(e1_err) & np.isfinite(e2_err)
physical = (np.abs(e1) < 1.0) & (np.abs(e2) < 1.0)
well_measured = (e1_err > 0.0) & (e1_err < 0.4) & (e2_err > 0.0) & (e2_err < 0.4)
radial = (radii > 10.0) & (radii < 130.0)

use = finite & physical & well_measured & radial

print(f"after quality cuts : {use.sum()}")

"""
__Weak Dataset__

The per-galaxy noise combines the intrinsic shape dispersion (each galaxy has a random unlensed
ellipticity; sigma_int ~ 0.25 per component is the standard value) with the measurement
uncertainty, in quadrature.

`from_arrays` builds the `WeakDataset`; `is_reduced=True` (the loader default) records that these
are measured ellipticities — reduced shear — so `FitWeak` will compare them against the model's
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

aplt.subplot_weak_dataset(dataset=dataset)

"""
__Mass Map__

Before any model is fitted, the Kaiser-Squires inversion gives a model-independent mass map.
Abell 2744 is one of the most disturbed clusters known — published lensing maps (Merten et al.
2011; Medezinski et al. 2016; Harvey & Massey 2024) show multiple substructures around the main
core from an ongoing merger — so we should *not* expect a clean single peak, and the structure in
this map is the first indication the catalogue's shear signal is real.
"""
aplt.plot_convergence_map(
    shear_yx=dataset.shear_yx,
    shape_native=(30, 30),
    smoothing_sigma_pixels=1.5,
)

"""
__Model__

The model is a spherical NFW dark-matter halo — the standard first-order description of a cluster
halo and deliberately simple for a merging system (the published analyses use multiple halos; a
single NFW measures the dominant mass concentration).

The halo centre gets Gaussian priors of width 10" about the projected cluster core, and the fit
assumes a single effective source plane at z = 1.0 behind the z = 0.308 cluster (the catalogue
provides no per-galaxy redshifts; this is the standard effective-depth approximation and its
choice rescales the inferred halo normalisation).
"""
mass = af.Model(al.mp.NFWSph)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=10.0)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=10.0)

lens = af.Model(al.Galaxy, redshift=0.308, mass=mass)

source = af.Model(al.Galaxy, redshift=1.0)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__Model Fit__

We now fit the data with the halo model using the non-linear fitting method and nested sampling
algorithm Nautilus. The `AnalysisWeak` object defines the `log_likelihood_function` that compares
the model's reduced shear at every galaxy position with the measured ellipticities.
"""
# use_jax=False: this fit is small enough that NumPy completes in minutes, and
# visualization of a JAX-path weak fit currently crashes (PyAutoLens#614). The
# JOSS benchmark (autolens_jax_joss/benchmarks/weak.py) times the identical fit
# on the JAX path with visualization disabled.
analysis = al.AnalysisWeak(
    dataset=dataset,
    use_jax=False,
)

search = af.Nautilus(
    path_prefix=Path("weak"),  # The path where results and output are stored.
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag="a2744_pyrrg",  # A unique tag which also defines the folder.
    n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
    iterations_per_quick_update=10000,  # Every N iterations the max likelihood model is output to hard-disk.
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

The tangential-shear profile is the standard presentation of a cluster weak-lensing measurement:
binned data (with the cross-component B-mode null test) against the maximum-likelihood NFW curve.

What this shallow sample can and cannot show — read the numbers with the sample size in mind:

 - With only ~400 usable shapes, the overall tangential-shear detection is *marginal* (a weighted
   mean gamma_t of ~0.02 at ~1.5 sigma), though it behaves exactly as a real lensing signal
   should: it concentrates at small radii and the cross-component B-mode is consistent with zero.
 - The NFW posterior is correspondingly broad — that *is* the honest result of fitting ~400
   galaxies around one cluster. Published A2744 analyses (Medezinski et al. 2016 quote a virial
   mass ~2 x 10^15 solar masses; Harvey & Massey 2024 map the merger's substructure) rest on
   samples an order of magnitude deeper with survey-grade calibration.
 - What the example demonstrates is the *workflow* on real sky data, which transfers unchanged to
   a deep catalogue via `WeakDataset.from_fits` / `from_csv`.
"""
print(result.info)

aplt.subplot_fit_weak(fit=result.max_log_likelihood_fit)

aplt.plot_shear_profile(
    result.max_log_likelihood_fit,
    centre=(0.0, 0.0),
    bins=8,
)

"""
__Model Your Own Catalogue__

If you have your own weak-lensing shape catalogue, you are now ready to fit it by adapting the
code above:

- Load positions and ellipticities via `WeakDataset.from_arrays`, `from_fits` or `from_csv`.
- Project RA/Dec to arc-second offsets about your chosen centre, as above.
- Apply your survey's quality cuts — the ones above are the bare minimum.
- Set `is_reduced=True` for measured ellipticities (reduced shear), the usual case.
- Combine intrinsic shape dispersion and measurement noise in quadrature for the noise-map.

__Simulator__

You can also simulate weak-lensing shear catalogues from any mass model — great for practising the
workflow, testing sensitivity to sample depth, or building training sets. The
`scripts/weak/simulator.py` example shows how: it ray-traces a grid of source-galaxy positions
through a `Tracer` and draws noisy reduced-shear measurements at each.

__Wrap Up__

This script has shown how to fit real weak-lensing data with a dark-matter halo model.

The following locations of the workspace are good places to checkout next:

- `autolens_workspace/*/weak/modeling.py`: the weak-lensing modeling API on simulated data, where
  the truth is known.
- `autolens_workspace/*/weak/likelihood_function.py`: exactly what the weak-lensing likelihood
  computes, step by step.
- `autolens_workspace/*/weak/features/strong_lensing`: combining weak- and strong-lensing
  constraints on a single mass model — the hybrid approach used for clusters like A2744.
- `autolens_workspace/*/weak/real_data/a2744.py`: this cluster again with a fuller discussion of
  the catalogue's provenance and honest framing of its depth.
"""
