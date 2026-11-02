"""
Source Science: Multi Galaxy
============================

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or
galaxies).

Using a source galaxy model, we can compute key quantities such as the magnification, total flux, and intrinsic
size of the source.

This example shows how to perform these calculations for a **multi-galaxy** lens, where two (or more) galaxies of
comparable mass both contribute significantly to the lensing. The API is identical to the galaxy-scale case (see
`imaging/source_science.py`); what changes is the lens itself, and therefore the magnification.

That change is worth being explicit about, because it is the reason multi-galaxy lenses are scientifically
valuable for source science. Two co-dominant deflectors produce a larger and more structured caustic than either
galaxy alone, so a source behind such a system is typically **more highly magnified** than it would be behind a
single galaxy of the same total mass — which is what makes these systems useful for studying faint, high-redshift
sources.

__Contents__

- **Simulated Dataset:** Load and plot the `simple` multi-galaxy dataset (auto-simulating if absent).
- **Mask:** Define the 2D mask applied to the dataset.
- **Source Values:** The true lens and source model used to simulate the dataset.
- **Source Flux:** The total flux of the source galaxy.
- **Source Magnification:** The overall magnification of the source behind two deflectors.
- **Single Deflector Comparison:** How much of the magnification each deflector is responsible for.
- **Tracer:** Computing these quantities from a `max_log_likelihood_tracer`.
- **Magnification At A Point:** The signed magnification `mu = 1 / det(A)` at chosen (y,x) image-plane coordinates, computed via the Hessian.
- **Area Magnification:** Each lensed image's area divided by the area of the source region it comes from.
- **Parametric Source Models:** What you need from lens modeling to do this on real data.
- **Magnification Errors:** Errors on every magnification above, from a quick lens model fit and draws from its posterior (the only slow part of the script, so it runs last).
- **Magnification Uncertainty Map:** The error of the magnification at every image-plane pixel, from the same posterior draws.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Simulated Dataset__

We load and plot the `simple` multi-galaxy example dataset, which is simulated imaging of a close pair of
co-dominant deflectors lensing a single background source. We will use it to demonstrate source science
calculations.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/multi_galaxy/simulator.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We apply a 3.0 arcsecond circular mask and apply it to the `Imaging` object.

Source science calculations are typically performed on masked datasets to ensure only the lensed source is used
in the calculations. For a multi-galaxy lens this mask must enclose the Einstein ring of the *combined* mass
distribution, which wraps around the pair as a whole.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Source Values__

Source science calculations for real lenses are performed using the best-fitting model inferred from a dataset,
and this example demonstrates how to use this below.

However, for simplicity, we demonstrate these calculations using the Sersic source model used to simulate the
dataset, which we refer to as the "true" source model. When analysing real strong lenses, a true underlying model
is not known, but for simulated datasets it is.

This allows us to illustrate the calculations in a way that does not depend on the specific details of the data or
on assumptions about how the lens model is inferred.

The galaxies below correspond to the same tracer used to simulate the `simple` dataset, and therefore represent the
true model — two co-dominant deflectors with comparable Einstein radii (1.0" and 0.8"), an external shear held at
the system centre rather than on either galaxy, and one source.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

lens_galaxy_0 = al.Galaxy(
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

lens_galaxy_1 = al.Galaxy(
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

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

field = al.MassField(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

tracer = al.Tracer(
    galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy], fields=[field]
)

"""
By plotting the image of the tracer, we confirm it looks like the simulated dataset but does not have CCD imaging
features such as noise or blurring from a PSF.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Source Flux__

A key quantity for a source galaxy is its total flux, which can be used to compute magnitudes (see
`autolens_workspace/*/guides/units/flux` for more details on this).

The most simple way to compute the total flux of a light profile is to create a grid of (y,x) coordinates over which
we compute the image of the light profile, and then sum the image.

The units of the light profile `intensity` are the units of the data the light profile was fitted to. In this example
we will assume everything is in electrons per second (`e- s^-1`), which is typical for Hubble Space Telescope imaging
data.

Note that the source's intrinsic flux is a property of the **source**, so it does not depend on how many deflectors
lens it. It is the magnification below that the multi-galaxy nature of the lens changes.
"""
print(f"Source Galaxy's Intensity {source_galaxy.bulge.intensity} e- s^-1")

"""
The total flux, in units of `e- s^-1`, is computed by summing the image of the light profile over all pixels.

Note that we can use a `grid` of any shape and pixel scale here, the important thing is that it is so large
and high enough resolution that it captures all the light from the light profile.

Note that we are using the source galaxy's true light profile, which corresponds to its emission in the
source-plane. For real datasets, we have to infer this via lens modeling.
"""
grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

image = source_galaxy.bulge.image_2d_from(grid=grid)

total_flux = np.sum(image)  # in units e- s^-1 as summed over pixels

print(f"Total Source Flux: {total_flux} e- s^-1")

"""
__Source Magnification__

The overall magnification of the source is estimated as the ratio of total surface brightness in the image-plane and
total surface brightness in the source-plane.

Note that the surface brightness is different to the total flux above, as surface brightness is flux per unit area.
We therefore explicitly mention how area folds into the calculation below.

To ensure the magnification is stable and that we resolve all source emission in both the image-plane and
source-plane we use a very high resolution grid, higher than we used to compute the total flux above.
"""
grid = al.Grid2D.uniform(shape_native=(1000, 1000), pixel_scales=0.03)

"""
We repeat our calculation of the source's total flux in the source-plane using this higher resolution grid, note
that we do not take the area into account, the reason for this is explained below.
"""
image = source_galaxy.bulge.image_2d_from(grid=grid)

total_source_plane_flux = np.sum(image)  # in units e- s^-1 as summed over pixels

"""
We now need the total flux of the lensed source in the image-plane, that is how much flux we measure after
gravitational lensing.

To calculate this, we first ray-trace the grid above from the image-plane to the source-plane using the tracer
and then pass it to the source galaxy's light profile to compute the lensed image.

The tracer sums both deflectors' deflection fields internally, so this single line accounts for the lensing of the
pair as a whole.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

source_plane_grid = traced_grid_list[1]

lensed_source_image = source_galaxy.bulge.image_2d_from(grid=source_plane_grid)

total_image_plane_flux = np.sum(
    lensed_source_image
)  # in units e- s^-1 as summed over pixels

"""
We now take the ratio of the total image-plane flux to source-plane flux to estimate the magnification.

Because both fluxes were computed on grids with the same total area and area per pixel, we do not need to
explicitly account for area in this calculation. This is because the area terms cancel out when taking the ratio.
Were the grid areas different, we would need to include area terms in the calculation.
"""
source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Magnification: {source_magnification}")

"""
__Single Deflector Comparison__

For a multi-galaxy lens it is instructive to ask how much of that magnification each deflector is responsible for.

Below we recompute the magnification four times: with each deflector acting alone, with the full pair, and finally
with the pair plus the external shear. This is a calculation you cannot do for a galaxy-scale lens, and it makes
concrete why the multi-galaxy regime is not simply "a slightly bigger lens galaxy".

The magnification of the pair is **not** the sum of the individual magnifications. Magnification depends
non-linearly on the deflection field (it is set by the Jacobian of the lens mapping), so two co-dominant deflectors
reshape the caustic structure rather than just deepening one galaxy's potential. This is the mechanism behind the
strong magnifications these systems deliver.

The first three cases below **exclude** the external shear, so the only thing varying is which deflectors are
present. A fourth case adds the shear back, which reconciles this comparison with the headline magnification
computed above and shows that the shear is not a negligible part of the total.

Each case computes its own source-plane flux on the same grid, so every magnification is a self-consistent ratio.
We use a coarser grid than above (same sky area, fewer pixels), which is ample for comparing ratios and keeps this
illustrative calculation fast — so the fourth number below will be close to, but not exactly equal to, the headline
value.
"""
grid_comparison = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.06)

source_plane_flux_comparison = float(
    np.sum(source_galaxy.bulge.image_2d_from(grid=grid_comparison))
)


def magnification_from(mass_profiles) -> float:
    """
    Compute the total magnification of the source for a given list of lens mass profiles.

    Each mass profile is placed in its own lens galaxy at the lens redshift, the comparison grid is
    ray-traced through them, and the lensed source flux is divided by the unlensed source-plane flux.
    """
    lens_galaxies = [al.Galaxy(redshift=0.5, mass=mass) for mass in mass_profiles]

    tracer_local = al.Tracer(galaxies=lens_galaxies + [source_galaxy])

    traced_grid_local = tracer_local.traced_grid_2d_list_from(grid=grid_comparison)[1]

    lensed_image_local = source_galaxy.bulge.image_2d_from(grid=traced_grid_local)

    return float(np.sum(lensed_image_local) / source_plane_flux_comparison)


mass_0 = lens_galaxy_0.mass
mass_1 = lens_galaxy_1.mass
shear = field.shear

magnification_lens_0 = magnification_from(mass_profiles=[mass_0])
magnification_lens_1 = magnification_from(mass_profiles=[mass_1])
magnification_both = magnification_from(mass_profiles=[mass_0, mass_1])
magnification_both_shear = magnification_from(mass_profiles=[mass_0, mass_1, shear])

print(f"Magnification (lens_0 alone)     : {magnification_lens_0:.3f}")
print(f"Magnification (lens_1 alone)     : {magnification_lens_1:.3f}")
print(f"Magnification (both deflectors)  : {magnification_both:.3f}")
print(f"Magnification (both + shear)     : {magnification_both_shear:.3f}")
print(
    f"Sum of individual magnifications : {magnification_lens_0 + magnification_lens_1:.3f} "
    "(note this does NOT equal the pair's magnification)"
)

"""
The pair magnifies the source several times more strongly than the sum of what each galaxy would achieve alone. Two
co-dominant deflectors are therefore not equivalent to one bigger galaxy, nor to two independent lenses — their
combined potential creates an extended caustic structure that neither produces by itself.

This is the quantitative reason multi-galaxy lenses are prized for studying faint, high-redshift sources.

__Tracer__

Lens modeling returns a `max_log_likelihood_tracer`, which is likely the object you have at hand to compute
source science calculations for real datasets.

The code below shows how using a tracer, composed of any combination of lens and source galaxies, we can
compute the source flux and magnification. It reproduces the calculations above, and works unchanged regardless of
how many deflectors the tracer contains.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

image_plane_grid = traced_grid_list[0]
source_plane_grid = traced_grid_list[1]

lensed_source_image = tracer.planes[1].image_2d_from(grid=source_plane_grid)
source_plane_image = tracer.planes[1].image_2d_from(grid=image_plane_grid)

total_image_plane_flux = np.sum(lensed_source_image)
total_source_plane_flux = np.sum(source_plane_image)

source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Plane Total Flux via Tracer: {total_source_plane_flux} e- s^-1")
print(f"Source Magnification via Tracer: {source_magnification}")

"""
Note that `tracer.planes` groups galaxies by **redshift**, not one plane per galaxy. Both deflectors here are at
z=0.5, so `tracer.planes[0]` contains them **both** (plus the shear `MassField`) and `tracer.planes[1]` is the
source.
This is why the code above is identical to the galaxy-scale version — the source is always the last plane.
"""
print(f"number of planes = {len(tracer.planes)}")
print(f"entries in the lens plane = {len(tracer.planes[0])}")

"""
__Magnification At A Point__

The magnifications above are the overall magnification of the source: a ratio of fluxes integrated over the
source's full light distribution. Lens modeling codes also commonly quote the magnification at a single (y,x)
coordinate in the image-plane, for example at a lensed image, a compact clump in an arc or a transient.

This is computed from the deflection angles of the lens model via the Hessian, which gives the lensing Jacobian `A`
at each coordinate, with the magnification `mu = 1 / det(A)`.

The point magnification is signed, and its sign is the parity of the lensed image at that coordinate: positive for
images at minima (or maxima) of the arrival-time surface and negative for saddle-point images, which are
mirror-reversed relative to the source. Its absolute value is how much a point-like source at that location is
magnified. It is therefore distinct from the source magnification computed above, which averages the magnification
over the extended source and is always positive. Close to a critical curve `det(A)` goes to zero, so the point
magnification becomes very large and changes quickly with the coordinate.

For a multi-galaxy lens, every mass profile in the tracer (both deflectors and the external shear) enters the
Hessian, so neither deflector has a point magnification of its own. Below, we compute it at two (y,x) coordinates on
the lensed arcs.

For a real lens, pass the `max_log_likelihood_tracer` of your lens model and coordinates measured from your data.
For a multi-plane system (for example a cluster with sources at several redshifts), the magnification must be
computed up to the plane of the source of interest with `plane_j`; see the `Trap (b)` section of
`autolens_workspace/*/guides/advanced/multi_plane.py`.
"""
point_grid = al.Grid2DIrregular(values=[(0.1, 1.9), (-1.6, 0.3)])

lens_calc = al.LensCalc.from_tracer(tracer=tracer)

magnification_at_points = lens_calc.magnification_2d_via_hessian_from(grid=point_grid)

print(f"Magnification At Points (signed): {magnification_at_points}")
print(f"Magnification At Points (absolute): {abs(magnification_at_points)}")

"""
__Area Magnification__

The area magnification of each lensed image is the image's area divided by the area of the source region it comes
from, where each small piece of the image, of area `dA`, comes from a piece of the source of area `dA / |mu|`. It needs
only the lens model and an outline of the images, and the source-plane area measures the size of the part of the
source that forms each image. The `imaging/source_science.py` example explains the calculation and its cautions.

Below we outline the arcs as the pixels where the lensed source is brighter than 10% of its peak, split them into
images (separated by critical curves, across which the magnification changes sign) and print each image's area
magnification and source-plane area.
"""
arc_grid = al.Grid2D.from_mask(mask=mask)

lensed_source_image = tracer.image_2d_list_from(grid=arc_grid)[-1]

arc_mask = al.Mask2D(
    mask=np.asarray(lensed_source_image.native) < 0.1 * np.max(lensed_source_image),
    pixel_scales=mask.pixel_scales,
)

image_masks = lens_calc.image_masks_from(mask=arc_mask)
area_magnifications = lens_calc.area_magnification_list_from(mask=arc_mask)

for image_mask, area_magnification in zip(image_masks, area_magnifications):
    source_plane_area = lens_calc.source_plane_area_from(mask=image_mask)

    print(
        f"Image of {np.sum(~np.asarray(image_mask))} pixels: "
        f"area magnification {area_magnification:.2f}, "
        f"source-plane area {source_plane_area:.4f} arcsec^2"
    )

"""
__Parametric Source Models__

If your lens modeling uses a parametric source model (e.g. Sersic, Multi Gaussian Expansion), the only object
you need to perform source science calculations is the `max_log_likelihood_tracer` returned by lens modeling.

Alternatively, as done above, you can manually set up a tracer using the lens and source galaxies inferred
by lens modeling.

Therefore, you may now wish to go to your results, extract the `max_log_likelihood_tracer`, and use it to compute
the source flux and magnification as shown above.

One caution specific to this regime: the magnification depends on the *total* deflection field, which a
multi-galaxy fit constrains well, but it also depends on how mass is split between the deflectors, which is
constrained less well (see the corner plot discussion in `multi_galaxy/modeling.py`). When quoting a magnification
for a multi-galaxy lens, propagate the posterior rather than using only the maximum likelihood tracer.

Where to go next:

- `autolens_workspace/*/guides/units/flux`: converting fluxes to magnitudes and physical units.
- `autolens_workspace/*/multi_galaxy/modeling`: inferring the lens model these calculations depend on.
- `autolens_workspace/*/imaging/source_science`: the galaxy-scale version of this guide, including pixelized
  source reconstructions.
"""

"""
__Magnification Errors__

Every magnification above is a single number, computed from one model of the lens. For a real lens that model is
inferred from the data and is uncertain, so every magnification computed from it has an error too.

We estimate these errors the standard way: fit a lens model, draw many models at random from its posterior, recompute
each magnification for every drawn model and quote the median and the range containing the central 68% of the values
(the 16th to 84th percentiles). This is how the LEGGOS II paper (arXiv:2606.20804) reports its magnification errors,
using about 300 drawn models.

This is the only part of this script that fits a lens model, which is why it comes last: everything above has already
run. On a laptop CPU the fit is slower than in the single-lens examples, because it models several galaxies, and can
take several hours; each draw below takes a few seconds. The `autolens_workspace/*/guides/results` examples describe how
to load results and draw from a posterior in more detail.

__Model Fit__

We fit a model of the two lens galaxies, each with Sersic light and an isothermal mass, an external shear held at
the system centre and a Sersic source. The light profiles are linear, so their intensities are solved for rather than
sampled, which keeps the number of free parameters down. The search is a quick Nautilus run with 200 live points, the
number `multi_galaxy/modeling` uses for its two-deflector model. A positions likelihood, which requires the multiple
images of the source to trace to the same location in the source plane, stops the fit from settling on a solution
where the lens has the wrong Einstein radius.

The simulated dataset contains a faint extra galaxy that is not part of the lens model, so its pixels are given large
noise values, as in the modeling examples.
"""
import autofit as af

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
dataset = dataset.apply_mask(mask=mask)

lens_0 = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp_linear.Sersic, mass=al.mp.Isothermal
)
lens_1 = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp_linear.Sersic, mass=al.mp.Isothermal
)

field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model = af.Collection(
    galaxies=af.Collection(lens_0=lens_0, lens_1=lens_1, source=source),
    fields=field,
)

positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
    use_jax=True,
)

search = af.Nautilus(
    path_prefix=Path("source_science"),
    name="magnification_errors",
    unique_tag=dataset_name,
    n_live=200,
)

result = search.fit(model=model, analysis=analysis)

"""
__Images__

The area magnification is computed per image, so we need the images. We split the arc mask used above with the
maximum likelihood model's critical curves and keep it fixed for every drawn model, as you would for a mask drawn on
the data.

Splitting the mask also returns pieces only one or two pixels in size, from pixels lying on a critical curve. They are
not images in their own right, so we keep only pieces of at least 10 pixels, which are the images of the source.
"""
image_masks = al.LensCalc.from_tracer(
    tracer=result.max_log_likelihood_tracer
).image_masks_from(mask=arc_mask)

image_masks = [
    image_mask for image_mask in image_masks if np.sum(~np.asarray(image_mask)) >= 10
]

print(f"Number of images: {len(image_masks)}")

"""
__Posterior Draws__

We now draw models from the posterior. For each one we rebuild the tracer, with the solved intensity of the linear
source included, and recompute the source magnification, the point magnification at the multiple image positions and
the area magnification of every image.

The calculations are done for the plane at the source's redshift, which `plane_index_via_redshift_from` converts to
the plane's index. For a system with sources at several redshifts (for example a cluster), change `source_redshift`
to compute every magnification for a different source.

The source magnification uses a 300 x 300 grid rather than the 1000 x 1000 grid above, which keeps each draw fast
and changes the source magnification by less than its error.
"""
import time

source_redshift = 1.0

total_draws = 100

grid = al.Grid2D.uniform(shape_native=(300, 300), pixel_scales=0.03)

source_magnification_list = []
point_magnification_list = []
magnification_map_list = []

map_grid = dataset.grid
area_magnification_list = []

start = time.time()

for _ in range(total_draws):
    instance = result.samples.draw_randomly_via_pdf()

    tracer_draw = analysis.fit_from(
        instance=instance
    ).tracer_linear_light_profiles_to_light_profiles

    plane_index = tracer_draw.plane_index_via_redshift_from(redshift=source_redshift)
    source_plane = tracer_draw.planes[plane_index]

    source_plane_grid = tracer_draw.traced_grid_2d_list_from(grid=grid)[plane_index]

    source_magnification_list.append(
        np.sum(source_plane.image_2d_from(grid=source_plane_grid))
        / np.sum(source_plane.image_2d_from(grid=grid))
    )

    lens_calc = al.LensCalc.from_tracer(tracer=tracer_draw, plane_j=plane_index)

    magnification_map_list.append(lens_calc.magnification_2d_from(grid=map_grid))

    point_magnification_list.append(
        np.asarray(lens_calc.magnification_2d_via_hessian_from(grid=positions))
    )

    area_magnification_list.append(
        [
            lens_calc.area_magnification_from(mask=image_mask)
            for image_mask in image_masks
        ]
    )

print(
    f"{total_draws} draws in {time.time() - start:.1f} s "
    f"({(time.time() - start) / total_draws:.2f} s per draw)"
)

"""
__Errors__

The median and the 16th and 84th percentiles of each magnification over the draws give its value and error. The
point magnification keeps its sign, which is the parity of each image.
"""
for name, values in [
    ("Source Magnification", source_magnification_list),
    ("Point Magnification At Positions", point_magnification_list),
    ("Area Magnification Per Image", area_magnification_list),
]:
    lower, median, upper = np.percentile(np.asarray(values), [16.0, 50.0, 84.0], axis=0)

    print(f"{name}: median {np.round(median, 2)}")
    print(f"    68% interval {np.round(lower, 2)} to {np.round(upper, 2)}")
    print(f"    fractional error {np.round((upper - lower) / 2.0 / np.abs(median), 3)}")

"""
Compare the medians with the true magnifications printed earlier in this script. The 68% intervals should contain the
true values about two-thirds of the time if the fitted model can describe the lens; offsets larger than the errors are a
sign that it cannot. LEGGOS II find fractional errors of 3% to 9% for the lensed galaxies they study.

These errors are typically smaller than real ones, because the dataset is simulated with a high signal-to-noise source,
and the model fitted is the one used to simulate it. Real lenses, and especially cluster lenses with many mass
components, have more complex mass distributions and fainter data, so their magnification errors are larger.

More draws give a smoother estimate of the percentiles: LEGGOS II use about 300, and 100 is enough here because the
errors are only used to illustrate the calculation. The errors also only include the uncertainty of the model that was
fitted: a different mass model (for example a power law instead of an isothermal) can change the magnifications by
more than these errors.
"""

"""
__Magnification Uncertainty Map__

Each draw's magnification map was stored in the loop above, so the error of the magnification at every image-plane
pixel can be mapped: half the 16th-84th percentile range of |mu| over the draws divided by the best-fit |mu|, with
the best-fit contours of constant magnification on top. The `imaging/source_science.py` example explains the map.
"""
tracer_best = result.max_log_likelihood_tracer

lens_calc_best = al.LensCalc.from_tracer(
    tracer=tracer_best,
    plane_j=tracer_best.plane_index_via_redshift_from(redshift=source_redshift),
)

aplt.plot_magnification_uncertainty_2d(
    magnification_list=magnification_map_list,
    magnification_best=lens_calc_best.magnification_2d_from(grid=map_grid),
    grid=map_grid,
    mass_obj=tracer_best,
)
