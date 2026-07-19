"""
Real Data: Combined Strong + Weak Lensing of Abell 2744
=======================================================

This script constrains a **single cluster-scale mass model** of **Abell 2744** ("Pandora's
Cluster", z = 0.308) using both its **strong-lensing** multiple images and its **weak-lensing**
shear catalogue — real data on both sides, fitted jointly with one non-linear search via
PyAutoFit's factor-graph API.

It combines the two real-data examples of this workspace:

 - The strong-lensing constraints of `cluster/start_here.py`: the 7 gold multiple-image systems
   (25 images, spectroscopic sources from z = 1.69 to 5.66) and cluster-member catalogue from the
   published lens-model inputs of Bergamini et al. 2023 (A&A 670, A60).
 - The weak-lensing constraints of `weak/start_here.py`: the pyRRG galaxy shape catalogue of
   Harvey & Massey 2024 (MNRAS 529, 802).

Crucially, both datasets were prepared in the **same coordinate frame** (arc-second offsets about
the projected cluster core), so one mass model can fit both directly.

__Why combine them?__

 - **Strong lensing** pins the mass distribution exquisitely — but only inside the ~30" region
   where multiple images form.
 - **Weak lensing** measures the mass profile out to hundreds of arcseconds — noisily per galaxy,
   but with ensemble statistical power, and precisely where strong lensing has none.

Joint fitting forces one model to satisfy both regimes — the hybrid approach of Niemiec et al.
2020, who showed sequential fitting biases cluster profiles at 2-3 sigma where a joint fit stays
within ~1 sigma. For the simulated galaxy-scale introduction to this feature, see `modeling.py` in
this folder.

__Contents__

- **Dataset (Strong):** The multiple-image CSVs of `dataset/cluster/a2744/`.
- **Dataset (Weak):** Download and cut the A2744 shape catalogue, as in `weak/start_here.py`.
- **Model:** The four-tier cluster mass model, shared by both regimes.
- **Analysis Factors & Factor Graph:** Point analyses + a weak analysis over one parameter space.
- **Search & Model-Fit:** Nautilus over the shared parameters.
- **Result:** Reading the strong/weak complementarity.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset (Strong)__

The committed Abell 2744 strong-lensing CSVs (see `dataset/cluster/a2744/README.md` for
provenance).
"""
dataset_path = Path("dataset") / "cluster" / "a2744"

dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")

mass_table = al.galaxy_models_from_csv(
    file_path=dataset_path / "mass.csv", family="mass"
)
point_table = al.galaxy_models_from_csv(
    file_path=dataset_path / "point.csv", family="point"
)
scaling_galaxies_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

"""
__Dataset (Weak)__

The same catalogue download, projection and quality cuts as `weak/start_here.py` — see that script
for the full discussion of each step. The projection centre is identical to the strong-lensing
CSVs' centre, which is what lets one mass model serve both datasets.
"""
weak_path = Path("dataset") / "weak" / "a2744_pyrrg"
catalogue_path = weak_path / "abell2744_galaxies.fits"

CATALOGUE_URL = (
    "https://raw.githubusercontent.com/davidharvey1986/pyRRG/"
    "0ccc29fb4513137da61b1afb632ca492093bd609/"
    "trainStarGalClass/TrainingData/abell2744_galaxies.fits"
)

if not catalogue_path.exists():
    import urllib.request

    weak_path.mkdir(parents=True, exist_ok=True)
    print("Downloading A2744 catalogue from pyRRG (one-off, ~3 MB) ...")
    urllib.request.urlretrieve(CATALOGUE_URL, catalogue_path)

from astropy.io import fits as astropy_fits

with astropy_fits.open(catalogue_path) as hdul:
    table = hdul[1].data

ra = np.asarray(table["ra"], dtype=float)
dec = np.asarray(table["dec"], dtype=float)
e1 = np.asarray(table["e1"], dtype=float)
e2 = np.asarray(table["e2"], dtype=float)
e1_err = np.asarray(table["e1_err"], dtype=float)
e2_err = np.asarray(table["e2_err"], dtype=float)

ra_centre, dec_centre = 3.5875, -30.3972  # A2744 core — same centre as the strong CSVs

x = (ra - ra_centre) * np.cos(np.deg2rad(dec_centre)) * 3600.0
y = (dec - dec_centre) * 3600.0
radii = np.sqrt(x**2.0 + y**2.0)

finite = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(e1_err) & np.isfinite(e2_err)
physical = (np.abs(e1) < 1.0) & (np.abs(e2) < 1.0)
well_measured = (e1_err > 0.0) & (e1_err < 0.4) & (e2_err > 0.0) & (e2_err < 0.4)
radial = (radii > 10.0) & (radii < 130.0)
use = finite & physical & well_measured & radial

sigma_int = 0.25
noise = np.sqrt(sigma_int**2.0 + 0.5 * (e1_err[use] ** 2.0 + e2_err[use] ** 2.0))

dataset_weak = al.WeakDataset.from_arrays(
    positions=np.stack([y[use], x[use]], axis=1),
    gamma_1=e1[use],
    gamma_2=e2[use],
    noise_map=list(noise),
    is_reduced=True,
    name="a2744_pyrrg",
)

print(dataset_weak.info)

"""
__Model__

The same four-tier mass model as `cluster/start_here.py` — 2 dPIE BCGs, an NFW host halo, and the
188-member scaling tier. The mass-model components are composed **once**: the strong-lensing
factors and the weak-lensing factor below receive views containing the *same* model objects, so
both regimes constrain the same priors.

The weak-lensing view pairs the mass model with a single effective source plane at z = 1.0 (the
catalogue has no per-galaxy redshifts — the standard effective-depth approximation, as in
`weak/start_here.py`); the strong-lensing view carries the 7 point sources at their spectroscopic
redshifts.
"""
redshift_lens = 0.308

galaxy_models = al.galaxy_af_models_from_csv_tables(mass_table, point_table)

for name in ("lens_0", "lens_1"):
    galaxy_models[name].mass.sigma = af.UniformPrior(lower_limit=50.0, upper_limit=600.0)
    galaxy_models[name].mass.r_core = af.UniformPrior(lower_limit=1.0, upper_limit=15.0)
    galaxy_models[name].mass.r_cut = af.UniformPrior(lower_limit=5.0, upper_limit=40.0)
    galaxy_models[name].mass.H0 = 67.66
    galaxy_models[name].mass.Om0 = 0.30966

galaxy_models["host_halo"].dark.mass_at_200 = af.LogUniformPrior(
    lower_limit=10**14.5, upper_limit=10**16.0
)

for i, dataset in enumerate(dataset_list):
    positions = np.atleast_2d(dataset.positions)
    point_attr = getattr(galaxy_models[f"source_{i}"], f"point_{i}")
    point_attr.centre_0 = af.GaussianPrior(
        mean=float(np.mean(positions[:, 0])), sigma=3.0
    )
    point_attr.centre_1 = af.GaussianPrior(
        mean=float(np.mean(positions[:, 1])), sigma=3.0
    )

scaling_sigma_ref = af.UniformPrior(lower_limit=0.0, upper_limit=300.0)
scaling_sigma_exponent = 0.25
scaling_radius_exponent = 0.5
reference_luminosity = 1.0
scaling_r_core_ref_fixed = 0.158
scaling_r_cut_ref_fixed = 15.8
source_redshift_max = max(float(d.redshift) for d in dataset_list)

scaling_galaxies_list = []
for centre, luminosity in zip(
    scaling_galaxies_table.centres, scaling_galaxies_table.luminosities
):
    luminosity_ratio = luminosity / reference_luminosity

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = tuple(centre)
    mass.sigma = scaling_sigma_ref * luminosity_ratio**scaling_sigma_exponent
    mass.r_core = scaling_r_core_ref_fixed * luminosity_ratio**scaling_radius_exponent
    mass.r_cut = scaling_r_cut_ref_fixed * luminosity_ratio**scaling_radius_exponent
    mass.redshift_object = redshift_lens
    mass.redshift_source = source_redshift_max
    mass.H0 = 67.66
    mass.Om0 = 0.30966

    scaling_galaxies_list.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

scaling_galaxies = af.Collection(scaling_galaxies_list)

# Strong-lensing view: the full multi-plane model.

model_strong = af.Collection(
    galaxies=af.Collection(**galaxy_models),
    scaling_galaxies=scaling_galaxies,
)

# Weak-lensing view: the same mass-model objects (lens_0, lens_1, host_halo and the
# scaling tier are the identical af.Model instances, so their priors are shared),
# with a single empty source galaxy defining the effective z = 1.0 source plane.

model_weak = af.Collection(
    galaxies=af.Collection(
        lens_0=galaxy_models["lens_0"],
        lens_1=galaxy_models["lens_1"],
        host_halo=galaxy_models["host_halo"],
        source_weak=af.Model(al.Galaxy, redshift=1.0),
    ),
    scaling_galaxies=scaling_galaxies,
)

"""
__Analysis Factors & Factor Graph__

One `AnalysisPoint` per strong-lensing source, plus one `AnalysisWeak` for the shear catalogue,
combined into a single factor graph. The total log likelihood is the sum over all factors.
"""
grid = al.Grid2D.uniform(shape_native=(120, 120), pixel_scales=1.0)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

analysis_factor_list = [
    af.AnalysisFactor(
        prior_model=model_strong,
        analysis=al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=True),
    )
    for dataset in dataset_list
]

analysis_factor_list.append(
    af.AnalysisFactor(
        prior_model=model_weak,
        analysis=al.AnalysisWeak(dataset=dataset_weak, use_jax=True),
    )
)

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

print(factor_graph.global_prior_model.info)

"""
__Search & Model-Fit__
"""
search = af.Nautilus(
    path_prefix=Path("weak") / "features",
    name="strong_lensing_a2744",
    unique_tag="a2744",
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=10000,
)

print(
    """
    The non-linear search has begun running.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("The search has finished run — you may now continue the notebook.")

"""
__Result__

One `Result` per factor — the strong-lensing results carry the multiple-image fits, the final
result the weak-lensing shear fit, and all share the same maximum-likelihood mass model.

The number to watch is the host halo's `mass_at_200`: strong lensing alone constrains the core
tightly but extrapolates the outer profile from the NFW form; the shear catalogue measures that
outer profile directly. Compare the joint posterior against a strong-only run
(`cluster/start_here.py`) and a weak-only run (`weak/start_here.py`) to see the complementarity.
"""
for result in result_list:
    print(result.max_log_likelihood_instance)

aplt.subplot_fit_weak(fit=result_list[-1].max_log_likelihood_fit)

"""
__Wrap Up__

A single mass model of a real merging cluster, constrained simultaneously by 25 multiple images
and ~400 weak-lensing shear measurements. From here:

- `weak/features/strong_lensing/modeling.py`: the simulated galaxy-scale introduction to this
  feature, where the truth is known.
- `cluster/start_here.py` / `weak/start_here.py`: each regime on its own.
- Published joint analyses of this cluster use several halos — adding a second NFW halo to
  `mass.csv` is a CSV-level edit.
"""
