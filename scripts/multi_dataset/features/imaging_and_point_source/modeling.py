"""
Modeling: Combined Imaging + Point Source
=========================================

This script jointly fits CCD imaging of a lensed quasar's **extended arcs** and its **point-source
observables** (image positions and time delays) with a **single lens mass model**, using PyAutoFit's
factor-graph API to sample the joint likelihood with one non-linear search.

We model **real data**: the quadruply imaged quasar **RXJ1131-1231** (z_source = 0.658, lensed by an
elliptical at z_lens = 0.295), the same system fitted with point-source observables alone in
`point_source/start_here.py`. Here we add HST imaging of the system, whose spectacular arc — the
quasar's host galaxy stretched around the lens — carries mass-model information the point positions
alone cannot access.

__Why combine them?__

The two datasets constrain the same mass model in complementary ways:

 - **Point-source observables** (positions, time delays) are exquisitely precise constraints at the
   four image locations, and the delays carry absolute-distance information — but four points only
   loosely pin the mass profile's radial slope.

 - **Extended arcs** trace the mass model continuously around the Einstein ring, breaking profile
   degeneracies — this is why time-delay cosmography analyses of this very lens (Suyu et al. 2013)
   model the arcs and the point data together.

__The Data__

 - Imaging: an HST H-band cutout of RXJ1131 fetched from the CDS hips2fits service on first run
   (~160 kB, cached). HiPS pixel values are drizzled/resampled from the original exposures, so this
   is real sky data at demonstration grade: we estimate a constant background RMS noise-map from the
   image edges and adopt a Gaussian PSF, which is sufficient to teach the joint workflow (a
   publication-grade analysis would use the reduced exposures, their weight maps and a measured
   PSF).
 - Point data: the committed `dataset/point_source/rxj1131/point_dataset_with_time_delays.json` —
   HST image positions (Suyu et al. 2013) and COSMOGRAIL time delays (Tewes et al. 2013).

__Contents__

- **Dataset:** Download/load both datasets; build the imaging noise-map and PSF; mask.
- **Quasar Image Masking:** Scale the noise at the four quasar image positions so the imaging fit
  targets the arcs.
- **Model:** One mass model shared by both datasets' analyses.
- **Analysis Factor & Factor Graph:** Combine the imaging and point analyses into one search.
- **Search & Model-Fit:** Nautilus over the shared parameter space.
- **Result:** Reading the joint constraints.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset (Imaging)__

The HST H-band cutout is downloaded once and cached. It is 200 x 200 pixels at 0.06" / pixel
(a 12" field centred on the lens).
"""
dataset_path = Path("dataset") / "multi_dataset" / "rxj1131"
data_fits_path = dataset_path / "data.fits"

HIPS2FITS_URL = (
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
    "?hips=CDS%2FP%2FHST%2FH&ra=172.96446&dec=-12.53293"
    "&width=200&height=200&fov=0.00333&projection=TAN&format=fits"
)

if not data_fits_path.exists():
    import urllib.request

    dataset_path.mkdir(parents=True, exist_ok=True)
    print("Downloading HST H-band image of RXJ1131 (one-off, ~160 kB) ...")
    urllib.request.urlretrieve(HIPS2FITS_URL, data_fits_path)

pixel_scales = 0.06

data = al.Array2D.from_fits(file_path=data_fits_path, pixel_scales=pixel_scales)

# hips2fits cutouts can contain NaNs at coverage edges — zero them.
data = al.Array2D.no_mask(
    values=np.nan_to_num(np.asarray(data.native)), pixel_scales=pixel_scales
)

# Under PYAUTO_SMALL_DATASETS=1 (smoke/CI), centre-crop the downloaded 200x200
# cutout to the 16x16 cap so it stays shape-consistent with the masks and grids
# built below (which honour the same env var) — a no-op in normal runs. Returns
# the updated pixel_scales too, so everything downstream stays consistent.
data, pixel_scales = al.util.dataset.cap_array_2d_for_small_datasets(data, pixel_scales)

"""
__Noise Map & PSF__

hips2fits does not provide a noise-map or PSF, so we construct demonstration-grade versions:

 - The noise-map is the robust RMS of the image's outer 20% border (sigma-clipped), applied
   uniformly — reasonable where the sky dominates, an underestimate on the bright lens galaxy.
 - The PSF is a circular Gaussian of FWHM 0.19" (typical for drizzled WFC3/IR H-band imaging).
"""
data_np = np.nan_to_num(np.asarray(data.native))

border = np.concatenate(
    [
        data_np[:20, :].ravel(),
        data_np[-20:, :].ravel(),
        data_np[:, :20].ravel(),
        data_np[:, -20:].ravel(),
    ]
)
clipped = border[np.abs(border - np.median(border)) < 3.0 * np.std(border)]
background_rms = float(np.std(clipped))

print(f"background RMS estimate: {background_rms:.4f}")

noise_map = al.Array2D.full(
    fill_value=background_rms,
    shape_native=data.shape_native,
    pixel_scales=pixel_scales,
)

psf = al.Convolver.from_gaussian(
    shape_native=(11, 11),
    sigma=0.19 / 2.355,  # FWHM 0.19" -> sigma in arcsec
    pixel_scales=pixel_scales,
)

dataset_imaging = al.Imaging(data=data, noise_map=noise_map, psf=psf)

aplt.subplot_imaging_dataset(dataset=dataset_imaging)

"""
__Dataset (Point Source)__

The positions + time delays dataset committed with the workspace (see
`point_source/start_here.py` for its full provenance).
"""
dataset_point = al.from_json(
    file_path=Path("dataset")
    / "point_source"
    / "rxj1131"
    / "point_dataset_with_time_delays.json",
)

print(dataset_point.info)

"""
__Quasar Image Masking__

The four quasar images are far brighter than the arcs and would dominate the imaging chi-squared;
modeling them in the image plane requires PSF deblending (see
`point_source/features/deblending/modeling.py`). The standard shortcut is to remove them from the imaging
fit: we scale the noise to very large values in a small circle around each observed image position,
so the imaging likelihood is driven by the arcs and lens light. Their information is not lost — the
point-source dataset carries it, at higher precision than the pixels could.
"""
quasar_mask_radius = 0.3

grid_all = al.Grid2D.uniform(shape_native=data.shape_native, pixel_scales=pixel_scales)

quasar_image_circles = np.zeros(data.shape_native, dtype=bool)
for centre in np.asarray(dataset_point.positions):
    distances = np.hypot(
        np.asarray(grid_all.native)[:, :, 0] - centre[0],
        np.asarray(grid_all.native)[:, :, 1] - centre[1],
    )
    quasar_image_circles |= distances < quasar_mask_radius

# `apply_noise_scaling` scales the noise where the mask is False, so the circles
# around the quasar images must be the mask's False pixels.
mask_quasar_images = al.Mask2D(
    mask=np.invert(quasar_image_circles), pixel_scales=pixel_scales
)

dataset_imaging = dataset_imaging.apply_noise_scaling(mask=mask_quasar_images)

"""
__Masking__

A 3.0" circular mask encloses the lens galaxy and the full Einstein ring of arcs.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset_imaging.shape_native,
    pixel_scales=pixel_scales,
    radius=mask_radius,
)

dataset_imaging = dataset_imaging.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset_imaging.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset_imaging = dataset_imaging.apply_over_sampling(
    over_sample_size_lp=over_sample_size
)

aplt.subplot_imaging_dataset(dataset=dataset_imaging)

"""
__Point Solver__

Identical configuration to `point_source/start_here.py`.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Model__

The **mass model is composed once and shared**: the same `Isothermal` + `ExternalShear` model
objects appear in both the imaging model and the point model below, so both analyses constrain the
same priors — the definition of a joint fit.

Each dataset then carries its own light components:

 - The imaging model adds an MGE lens-light bulge and an MGE source (the quasar's host galaxy,
   which forms the arcs).
 - The point model adds the `Point` source whose (y,x) centre the positions and delays constrain.

The quasar sits in its host galaxy, so their source-plane centres are physically coincident; we
keep them as separate free parameters here for simplicity (tying them is a one-line prior
customization).
"""
# Shared mass model:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

# Imaging model (lens light + host-galaxy source):

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)
source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

lens_imaging = af.Model(
    al.Galaxy, redshift=0.295, bulge=lens_bulge, mass=mass, shear=shear
)
source_imaging = af.Model(al.Galaxy, redshift=0.658, bulge=source_bulge)

model_imaging = af.Collection(
    galaxies=af.Collection(lens=lens_imaging, source=source_imaging)
)

# Point model (same mass + shear objects -> same priors):

lens_point = af.Model(al.Galaxy, redshift=0.295, mass=mass, shear=shear)
source_point = af.Model(al.Galaxy, redshift=0.658, point_0=af.Model(al.ps.Point))

model_point = af.Collection(
    galaxies=af.Collection(lens=lens_point, source=source_point)
)

"""
__Analysis Factor & Factor Graph__

One analysis per dataset, each wrapped in an `AnalysisFactor` with its own model view; the factor
graph combines them so a single search samples the shared parameter space. The total log likelihood
is the sum of the imaging and point log likelihoods.
"""
analysis_imaging = al.AnalysisImaging(dataset=dataset_imaging, use_jax=True)
analysis_point = al.AnalysisPoint(dataset=dataset_point, solver=solver, use_jax=True)

factor_imaging = af.AnalysisFactor(prior_model=model_imaging, analysis=analysis_imaging)
factor_point = af.AnalysisFactor(prior_model=model_point, analysis=analysis_point)

factor_graph = af.FactorGraphModel(factor_imaging, factor_point, use_jax=True)

print(factor_graph.global_prior_model.info)

"""
__Search & Model-Fit__
"""
search = af.Nautilus(
    path_prefix=Path("multi_dataset"),
    name="imaging_and_point_source",
    unique_tag="rxj1131",
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=10000,
)

print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed — this could take some time!
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("The search has finished run — you may now continue the notebook.")

"""
__Result__

`search.fit` on a factor graph returns one `Result` per dataset — the imaging result carries the
arc fit, the point result the positions/delays fit, and both share the same maximum-likelihood mass
model.

Compare the mass-model posterior to a positions-only fit (`point_source/start_here.py`): the arcs
tighten the constraints on the mass profile and shear, exactly the complementarity that motivates
joint modeling.
"""
for result in result_list:
    print(result.max_log_likelihood_instance)

aplt.subplot_fit_imaging(fit=result_list[0].max_log_likelihood_fit)

"""
__Wrap Up__

This script combined real HST imaging of RXJ1131's arcs with its point-source positions and time
delays in a single differentiable lens model — the analysis pattern behind time-delay cosmography.

Next steps:

- `point_source/start_here.py`: the same lens with point-source observables alone.
- `point_source/features/deblending/modeling.py`: modeling the quasar images in the pixels instead of
  masking them.
- `multi_dataset/features/imaging_and_interferometer`: the same joint-fit pattern combining imaging and
  interferometer visibilities.
"""
