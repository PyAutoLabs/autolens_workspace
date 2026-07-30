"""
Fits: Multi Galaxy
==================

This guide shows how to fit multi-galaxy strong lens data using the `FitImaging` object, including visualizing and
interpreting its results.

A multi-galaxy lens has two or more galaxies of comparable mass which both contribute significantly to the lensing
of a single background source. The fitting API is **identical** to the galaxy-scale case — a fit neither knows nor
cares how many galaxies deflect the light — so everything in this guide applies equally to `imaging/fit.py`. What
this script adds is the multi-deflector specifics: the deflection field is the **sum** of every deflector's field,
each galaxy's share of it can be inspected separately, and the "planes" of a fit group galaxies by *redshift*, not
one plane per galaxy.

References
----------

This example uses functionality described fully in other examples in the `guides` package:

- `guides/plot`: Using the plotting API (`aplt.plot_array`, `aplt.subplot_fit_imaging`, etc.) to visualize figures.
- `guides/units`: The source code unit conventions (e.g. arc seconds for distances and how to convert to physical units).
- `guides/data_structures`: The bespoke data structures used to store 1D and 2D arrays.

__Contents__

- **Loading Data:** Load the multi-galaxy dataset `simple` from .fits files (auto-simulating if absent).
- **Extra Galaxies Noise Scaling:** Scale the noise of nearby contaminating galaxies so they do not impact the fit.
- **Mask:** Define the 2D mask applied to the dataset for the fit.
- **Galaxies:** Compose the true pair + source by hand.
- **Per-Galaxy Deflections:** Each deflector's contribution to the summed deflection field.
- **Fitting:** Fit the tracer to the dataset and inspect the results.
- **Bad Fit:** A bad lens model will show features in the residual-map and chi-squared map.
- **Fit Quantities:** The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.
- **Figures of Merit:** There are single valued floats which quantify the goodness of fit.
- **Plane Quantities:** The `FitImaging` object has specific quantities which break down each image of each plane.
- **Per-Galaxy Quantities:** Breaking the lens plane down into each individual deflector's model image.
- **Unmasked Quantities:** All of the quantities above are computed using the mask which was used to fit the data.
- **Pixel Counting:** An alternative way to quantify residuals like the lens light residuals is pixel counting.
- **Outputting Results:** You may wish to output certain results to .fits files for later inspection.

__JAX__

This script constructs a `FitImaging` directly from a tracer and dataset
(no Analysis / no non-linear search). The fit itself is JAX-friendly:
any quantities it computes (`fit.model_image`, `fit.residual_map`,
`fit.log_likelihood`, etc.) work on either backend and return arrays
backed by `numpy.ndarray` on the default path or `jax.Array` if you
constructed the upstream objects with `xp=jnp`.

For the standard analysis-driven modeling path — where `AnalysisImaging`
auto-enables `use_jax=True` and the search driver handles the JIT
internally — see `start_here.py` / `modeling.py`. For the advanced path
where you wrap your own `@jax.jit` around `FitImaging` construction, see
`likelihood_function.py`'s `__JAX__` section and the `lens_calc.py` guide.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Loading Data__

We begin by loading the multi-galaxy strong lens dataset `simple` from .fits files, which is the dataset we will
use to demonstrate fitting.

This dataset was simulated using the `multi_galaxy/simulator` example, read through that to have a better
understanding of how the data this example fits was generated.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
The `aplt.subplot_imaging_dataset` contains a subplot which plots all the key properties of the dataset
simultaneously.

This includes the observed image data, RMS noise map, Point Spread Function and other information.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Extra Galaxies Noise Scaling__

Before masking, we must deal with any extra galaxies in the data: nearby galaxies (or foreground stars, or
data-reduction artefacts) whose emission is not associated with the strong lens but blends into the field. If
their light is left in the data it will contaminate the fit and bias the inferred lens model. It is too easy to
skip straight to fitting without checking for these, so we make this step an explicit part of the workflow.

For a multi-galaxy lens there is a judgement here that the galaxy-scale case does not force on you: the image
contains several galaxies and you must decide which are co-dominant *deflectors* (they belong in the tracer below,
with their own mass profiles) and which are *contaminants* (they belong here, removed by noise scaling). The test
is whether the galaxy contributes significantly to the lensing of the source, not whether it is bright or nearby.

To prevent extra galaxies from impacting the fit, we do not mask them entirely from the fit, which would be
analogous to making the circular mask smaller or using a more refined mask. When pixels are masked and removed
entirely from the fit, their coordinates are not used when performing ray-tracing and the light of the lens and
source galaxies in these pixels not evaluated.

Instead, the pixels are kept in the fit, but their data values are scaled to zero and their noise-map values
are increased to very large values. This means that during the fit, these pixels contribute negligibly to the
likelihood, and therefore do not impact the lens model.

This approach is used because for certain types of modeling approaches, like a pixelized source reconstruction,
masking regions of the image in a way that removes their image pixels entirely from the fit can produce
discontinuities in the pixelization. This can lead to unexpected systematics and unsatisfactory results.

The dataset includes a faint extra galaxy, and a `mask_extra_galaxies.fits` covering it is shipped with the
dataset (created by the simulator). If you are fitting your own data with an extra galaxy, you must either:

 - Create a `mask_extra_galaxies.fits` for it using the data-preparation tools (the GUI at the end of
   `multi_galaxy/start_here.py`, the GUI
   `autolens_workspace/*/imaging/data_preparation/gui/mask_extra_galaxies.py`, or the manual
   `autolens_workspace/*/imaging/data_preparation/examples/optional/mask_extra_galaxies.py`), then load it
   as below; or
 - Shrink the circular mask below so the extra galaxy lies outside it and is removed from the fit entirely.

After scaling, the extra galaxy's pixels have their data set to zero and noise-map increased, making their
signal-to-noise effectively zero.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,  # `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We now mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.

We use a ``Mask2D`` object, which for this example is a 3.0" circular mask.

Sizing this mask needs more care than at galaxy scale: the Einstein radius that matters is that of the *combined*
mass distribution (~1.8" here), not either galaxy's individually, and the lensed arcs wrap around the pair as a
whole. A mask sized by eye from one galaxy's light will clip the ring.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

"""
We now combine the imaging dataset with the mask.
"""
dataset = dataset.apply_mask(mask=mask)

"""
We now plot the image with the mask applied, where the image automatically zooms around the mask to make the lensed
source appear bigger.
"""
aplt.plot_array(array=dataset.data, title="Image Data With Mask Applied")

"""
The mask is also used to compute a `Grid2D`, where the (y,x) arc-second coordinates are only computed in unmasked
pixels within the masks' circle.

This grid will be used to perform lensing calculations when fitting the data below.
"""
aplt.plot_grid(grid=dataset.grid, title="Grid2D of Masked Dataset")

"""
__Over Sampling__

We load the centres of the two main lens galaxies and apply adaptive over sampling centred on both of them, so each
deflector's steep central light profile is evaluated accurately without paying that cost across the whole image.

Note how `centre_list` takes as many centres as you give it, so this scales unchanged as you add deflectors.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxies__

Compose the truth: the values the simulator used (see `multi_galaxy/simulator.py`), so the fit below
sits at the likelihood's maximum. Note the small light/mass centre offsets on each galaxy — the
J1011+0143-style science this regime measures.

Note also that the `ExternalShear` is **not** attached to either galaxy. It describes the tidal field of structure
outside the system, so it is a property of the system as a whole, and we hold it in its own galaxy at the system
centre (0.0", 0.0") — exactly as the simulator does. Because the tracer sums every deflection field, this is
numerically identical to attaching it to a deflector; it just stops the shear being mislabelled as a property of
`lens_0`.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__Per-Galaxy Deflections__

Because both deflectors are at one redshift, the total deflection field is simply the sum of each
galaxy's field (plus the shear's). Evaluate each galaxy's field on the masked grid and compare their
magnitudes — co-dominance in numbers: neither field is negligible anywhere near the ring.

If one of these magnitudes were much smaller than the other, this would be a galaxy-scale lens with a
minor perturber and belong in `imaging/` rather than here.
"""
grid = dataset.grids.lp

deflections_0 = lens_0.deflections_yx_2d_from(grid=grid)
deflections_1 = lens_1.deflections_yx_2d_from(grid=grid)

magnitude_0 = float(np.mean(np.linalg.norm(np.asarray(deflections_0.array), axis=-1)))
magnitude_1 = float(np.mean(np.linalg.norm(np.asarray(deflections_1.array), axis=-1)))

print(f'mean |deflection| lens_0 = {magnitude_0:.3f}"  lens_1 = {magnitude_1:.3f}"')
print(
    f"ratio = {magnitude_1 / magnitude_0:.2f}  (co-dominant: neither is a minor perturber)"
)

"""
__Fitting__

We can make a tracer from a collection of light profiles, mass profiles and galaxies.

The combination of light and mass profiles below is the same as those used to generate the simulated dataset we
loaded above. It therefore produces a tracer whose image looks exactly like the dataset.

The tracer sums the two deflectors' deflection fields internally, so from here on nothing about the API depends on
there being two of them.
"""
tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source])

"""
Because the tracer's light and mass profiles are the same used to make the dataset, its image is nearly the same as
the observed image.

However, the tracer's image does appear different to the data, in that its arcs appear a bit thinner. This is
because its image has not been blurred with the telescope optics PSF, which the data has.

[For those not familiar with Astronomy data, the PSF describes how the observed emission of the galaxy is blurred by
the telescope optics when it is observed. It mimics this blurring effect via a 2D convolution operation].
"""
aplt.plot_array(array=tracer.image_2d_from(grid=dataset.grid), title="Tracer Image")

"""
We now use a `FitImaging` object to fit this tracer to the dataset.

The fit creates a `model_image` which we fit the data with, which includes performing the step of blurring the
tracer's image with the imaging dataset's PSF. We can see this by comparing the tracer's image (which isn't PSF
convolved) and the fit's model image (which is).
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.plot_array(array=fit.model_data, title="Model Image")

"""
The fit does a lot more than just blur the tracer's image with the PSF, it also creates the following:

 - The `residual_map`: The `model_image` subtracted from the observed dataset's `data`.
 - The `normalized_residual_map`: The `residual_map` divided by the observed dataset's `noise_map`.
 - The `chi_squared_map`: The `normalized_residual_map` squared.

For a good lens model where the model image and tracer are representative of the strong lens system the
residuals, normalized residuals and chi-squareds are minimized:
"""
aplt.plot_array(array=fit.residual_map, title="Residual Map")
aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")
aplt.plot_array(array=fit.chi_squared_map, title="Chi Squared Map")

"""
A subplot can be plotted which contains all of the above quantities, as well as other information contained in the
tracer such as the source-plane image, a zoom in of the source-plane and a normalized residual map where the
colorbar goes from 1.0 sigma to -1.0 sigma, to highlight regions where the fit is poor.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
The fit also provides us with a ``log_likelihood``, a single value quantifying how good the tracer fitted the
dataset.

Lens modeling (see `multi_galaxy/modeling.py`) effectively tries to maximize this log likelihood value.
"""
print(f"log likelihood at the truth = {float(fit.log_likelihood):.2f}")

"""
__Bad Fit__

A bad lens model will show features in the residual-map and chi-squared map.

The multi-galaxy version of this is instructive, so rather than moving the source we perturb **one** deflector:
below we shift `lens_0`'s mass centre by 0.1" and reduce its Einstein radius, leaving `lens_1` and the source at
their true values.

This is a specifically multi-galaxy failure mode. Because the data constrains the *total* deflection much better
than it constrains the split between the two galaxies, getting one deflector's mass wrong does not produce
residuals localized around that galaxy — it produces residuals in the **arcs**, right across the system. That
degeneracy is exactly what the corner plot in `modeling.py` is there to expose.
"""
lens_0_bad = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.40, 0.38),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=0.7,
    ),
)

tracer_bad = al.Tracer(galaxies=[lens_0_bad, lens_1, shear_galaxy, source])

"""
A new fit using this tracer shows residuals, normalized residuals and chi-squared which are non-zero.
"""
fit_bad = al.FitImaging(dataset=dataset, tracer=tracer_bad)

aplt.subplot_fit_imaging(fit=fit_bad)

"""
We also note that its likelihood decreases.
"""
print(f"log likelihood of the bad fit = {float(fit_bad.log_likelihood):.2f}")

"""
__Fit Quantities__

The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.

There is a `model_image`, which is the image-plane image of the tracer we inspected above blurred with the imaging
data's PSF.

This is the image that is fitted to the data in order to compute the log likelihood and therefore quantify the
goodness-of-fit.
"""
print(fit.model_data.slim)

# The native property provides quantities in 2D NumPy Arrays.
# print(fit.model_data.native)

"""
There are numerous ndarrays showing the goodness of fit:

 - `residual_map`: Residuals = (Data - Model_Data).
 - `normalized_residual_map`: Normalized_Residual = (Data - Model_Data) / Noise
 - `chi_squared_map`: Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
"""
print(fit.residual_map.slim)
print(fit.normalized_residual_map.slim)
print(fit.chi_squared_map.slim)

"""
__Figures of Merit__

There are single valued floats which quantify the goodness of fit:

 - `chi_squared`: The sum of the `chi_squared_map`.

 - `noise_normalization`: The normalizing noise term in the likelihood function
    where [Noise_Term] = sum(log(2*pi*[Noise]**2.0)).

 - `log_likelihood`: The log likelihood value of the fit where [LogLikelihood] = -0.5*[Chi_Squared_Term + Noise_Term].
"""
print(fit.chi_squared)
print(fit.noise_normalization)
print(fit.log_likelihood)

"""
__Plane Quantities__

The `FitImaging` object has specific quantities which break down each image of each plane:

 - `model_images_of_planes_list`: Model-images of each individual plane, which in this example is a model image of
   the lens galaxies and a model image of the lensed source galaxy. Both images are convolved with the imaging's PSF.

 - `subtracted_images_of_planes_list`: Subtracted images of each individual plane, which are the data's image with
   all other plane's model-images subtracted. For example, the first subtracted image has the source galaxy's model
   image subtracted and therefore is of only the lens galaxies' emission. The second subtracted image is of the
   lensed source, with the lens galaxies' light removed.

**This is the one place the multi-galaxy case can surprise you.** A "plane" is a *redshift*, not a galaxy. Both
deflectors here are at z=0.5, so there are only **two** planes for three galaxies, and
`model_images_of_planes_list[0]` contains the **combined** light of `lens_0` and `lens_1`, not one of them. The list
length is the number of distinct redshifts, so adding a third co-dominant deflector at z=0.5 would still give two
entries.

(If your deflectors are at *different* redshifts the system is a compound, multi-plane lens and these lists do
lengthen — one entry per redshift.)
"""
print(f"number of planes = {len(fit.model_images_of_planes_list)}")

print(fit.model_images_of_planes_list[0].slim)
print(fit.model_images_of_planes_list[1].slim)

print(fit.subtracted_images_of_planes_list[0].slim)
print(fit.subtracted_images_of_planes_list[1].slim)

"""
__Per-Galaxy Quantities__

To break the lens plane down into each individual deflector — which is what you actually want for a multi-galaxy
lens — use `galaxy_model_image_dict`. This is keyed by the `Galaxy` objects themselves, so each deflector's own
PSF-convolved model image is available separately even though their light blends together in the data and shares a
plane.

This is how you check that each galaxy's light model is sensible individually, rather than only that their sum
fits.
"""
galaxy_model_images = fit.galaxy_model_image_dict

for name, galaxy in [("lens_0", lens_0), ("lens_1", lens_1), ("source", source)]:
    galaxy_image = galaxy_model_images[galaxy]
    print(f"{name}: model image sum = {float(np.sum(galaxy_image.slim)):.3f}")

"""
__Unmasked Quantities__

All of the quantities above are computed using the mask which was used to fit the data.

The `FitImaging` can also compute the unmasked blurred image of each plane.
"""
print(fit.unmasked_blurred_image.native)
print(fit.unmasked_blurred_image_of_planes_list[0].native)
print(fit.unmasked_blurred_image_of_planes_list[1].native)

"""
__Mask__

We can use the `Mask2D` object to mask regions of one of the fit's maps and estimate quantities of it.

Below, we estimate the average absolute normalized residuals within a 1.0" circular mask, which would inform us of
how accurate the lens light subtraction of a model fit is and if it leaves any significant residuals.

For a multi-galaxy lens a circle at the origin is not necessarily the region you care about, because neither galaxy
is at the origin. We therefore also measure the residuals in a 0.5" circle centred on **each** deflector, which
tells you whether one galaxy's light model is doing worse than the other's.
"""
mask_centre = al.Mask2D.circular(
    shape_native=fit.dataset.shape_native,
    pixel_scales=fit.dataset.pixel_scales,
    radius=1.0,
)

normalized_residuals = fit.normalized_residual_map.apply_mask(mask=mask_centre)

print(f"mean |normalized residual| within 1.0 of centre = {np.mean(np.abs(normalized_residuals.slim)):.3f}")

for i, centre in enumerate(main_lens_centres):
    mask_galaxy = al.Mask2D.circular(
        shape_native=fit.dataset.shape_native,
        pixel_scales=fit.dataset.pixel_scales,
        centre=(float(centre[0]), float(centre[1])),
        radius=0.5,
    )

    normalized_residuals_galaxy = fit.normalized_residual_map.apply_mask(
        mask=mask_galaxy
    )

    print(
        f"lens_{i}: mean |normalized residual| within 0.5 = "
        f"{np.mean(np.abs(normalized_residuals_galaxy.slim)):.3f}"
    )

"""
__Pixel Counting__

An alternative way to quantify residuals like the lens light residuals is pixel counting. For example, we could sum
up the number of pixels whose chi-squared values are above 10 which indicates a poor fit to the data.

Whereas computing the mean above gives the average level of residuals, pixel counting informs us how spatially large
the residuals extend.

We count over the full fitted mask, and compare the good fit to the bad fit from earlier.
"""
print(
    f"pixels with chi-squared > 10 (good fit) = {int(np.sum(fit.chi_squared_map > 10.0))}"
)
print(
    f"pixels with chi-squared > 10 (bad fit)  = {int(np.sum(fit_bad.chi_squared_map > 10.0))}"
)

"""
For a multi-galaxy lens, *where* those pixels are is the interesting part. We only perturbed `lens_0`'s mass, so a
naive expectation is that the damage is localized around `lens_0`. It is not.

Below we split the bad fit's poorly-fitted pixels by radius from the system centre. The overwhelming majority sit
out at the arcs (r > 1.0"), with only a handful within 0.5" of `lens_0` itself — because the data constrains the
*total* deflection, so mis-apportioning mass between the two deflectors shows up in the lensed images rather than
at the galaxy you got wrong. This is the same degeneracy the corner plot in `modeling.py` exposes.
"""
grid_2d = np.asarray(fit.dataset.grid.slim)
bad_pixels = np.asarray(fit_bad.chi_squared_map.slim) > 10.0

radii = np.hypot(grid_2d[:, 0], grid_2d[:, 1])
distance_to_lens_0 = np.hypot(grid_2d[:, 0] - 0.35, grid_2d[:, 1] - 0.25)

print(f"bad-fit pixels total          = {int(np.sum(bad_pixels))}")
print(f"  of which at r > 1.0 (arcs)  = {int(np.sum(bad_pixels & (radii > 1.0)))}")
print(
    f"  of which within 0.5 of lens_0 = {int(np.sum(bad_pixels & (distance_to_lens_0 < 0.5)))}"
)

"""
__Outputting Results__

You may wish to output certain results to .fits files for later inspection.

For example, one could output the lens light subtracted image of the lensed source galaxy to a .fits file such that
we could fit this source-only image again with an independent pipeline. For a multi-galaxy lens this subtracts the
light of **both** deflectors at once, since they share a plane.
"""
lens_subtracted_image = fit.subtracted_images_of_planes_list[1]

aplt.fits_array(
    array=lens_subtracted_image,
    file_path=dataset_path / "lens_subtracted_data.fits",
    overwrite=True,
)

"""
__Wrap Up__

- `imaging/fit.py` — the galaxy-scale version of this guide, identical machinery with one deflector.
- `multi_galaxy/modeling.py` — fitting a model (rather than the truth) to this dataset, and the posterior which
  exposes the two deflectors' mass degeneracy.
- `multi_galaxy/likelihood_function.py` — a step-by-step guide to how the log likelihood printed above is computed.
- `multi_galaxy/features/scaling_galaxies` — adding a far-out scaling tier to the deflection sum.
"""
