"""
Source Science
==============

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or galaxies).

Using a source galaxy model, we can compute key quantities such as the magnification, total flux, and intrinsic
size of the source.

This example shows how to perform these calculations using Multi Gaussian Expansion (MGE) sources on imaging data,
which is conceptually the simplest case for source science calculations and a good introduction to the topic.

__Contents__

- **Simulated Dataset:** We load and plot the `simple__no_lens_light` example dataset, which is simulated imaging of a.
- **Mask:** Define the 2D mask applied to the dataset for the model-fit.
- **Lens:** The simulated dataset was created using a lens galaxy with an Isothermal mass profile and External.
- **Multi Gaussian Expansion Source:** The default workspace source model is a Multi Gaussian Expansion (MGE) profile, which is a.
- **Source Flux:** A key quantity for a source galaxy is its total flux, which can be used to compute magnitudes (see.
- **Source Magnification:** The overall magnification of the source is estimated as the ratio of total surface brightness in.
- **Tracer:** Lens modeling returns a `max_log_likelihood_tracer`, which is likely the object you have at hand to.
- **Magnification At A Point:** The signed magnification `mu = 1 / det(A)` at chosen (y,x) image-plane coordinates, computed via the Hessian.
- **Parametric Source Models:** If your lens modeling uses a parametric source model (e.g.
- **Magnification Errors:** Errors on the magnifications above, from a quick lens model fit and draws from its posterior (the only slow part of the script, so it runs last).
- **Wrap Up:** Summary of the script and next steps.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt
from autogalaxy.profiles.plot.basis_plots import subplot_image as subplot_basis_image


"""
__Simulated Dataset__

We load and plot the `simple__no_lens_light` example dataset, which is simulated imaging of a strong lens
that we will use to demonstrate source science caluculations.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/imaging/features/no_lens_light/simulator.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We apply a 3.0 arcsecond circular mask and apply it to the `Imaging` object.

Source science calculations are typically performed on masked datasets to ensure only the lensed source is used
in the calculations.
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
__Lens__

The simulated dataset was created using a lens galaxy with an Isothermal mass profile and External Shear,
which we now define.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

field = al.MassField(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__Multi Gaussian Expansion Source__

The default workspace source model is a Multi Gaussian Expansion (MGE) profile, which is a superposition of 20
Gaussians whose `intensity` values are solved for via linear alegbra. 

These features are described in the `autolens_workspace/*/imaging/features/multi_gaussian_expansion`
and `autolens_workspace/*/imaging/features/linear_light_profiles` examples, but you do not need a full
understanding of these to follow this example on source science calculations.

We now set up a source galaxy using an MGE, use it to compute the source flux and magnification and compare
them to the true values computed above. We will then consider whether making different assumptions about the source
model (e.g. MGE versus Sersic) changes the inferred source science calculations.

We first set up a source galaxy using an MGE made up of 20 Gaussians whose `sigma` values span 0.01" to the mask 
radius of 3.0".
"""
total_gaussians = 20

# The sigma values of the Gaussians will be fixed to values spanning 0.0001 to the mask radius, 3.0".

log10_sigma_list = np.linspace(-4, np.log10(mask_radius), total_gaussians)

# A list of linear light profile Gaussians will be input here, which will then be used to fit the data.

bulge_gaussian_list = []

# Iterate over every Gaussian and create it, with it centered at (0.0", 0.0") and assuming spherical symmetry.

for i in range(total_gaussians):
    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        sigma=10 ** log10_sigma_list[i],
    )

    bulge_gaussian_list.append(gaussian)

# The Basis object groups many light profiles together into a single model component and is used to fit the data.

bulge = al.lp_basis.Basis(profile_list=bulge_gaussian_list)

source_galaxy = al.Galaxy(redshift=1.0, bulge=bulge)

"""
We now create the tracer using the lens galaxy and MGE source galaxy.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], fields=[field])

"""
Each of the Gaussians above does not have a manually input intensity, instead their intensities are solved for via
linear algebra when we use the tracer to fit strong lens imaging data.

We therefore create a fit of the tracer to the simulated dataset, which solves for the combination of Gaussian intensities
that best reconstruct the lensed source.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
From the fit, we can extract a tracer where the Gaussian intensities have been set to their best-fit values.

We therefore now have the tracer which we need to perform source science calculations.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

"""
However, lets first make a few plots confirming that the MGE source gives a good fit, and showing the individual 
Gaussians that make up the MGE (which uses a lower sized grid to make the Gaussians more visible).
"""
aplt.subplot_fit_imaging(fit=fit)

grid_basis_plot = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

subplot_basis_image(basis=tracer.galaxies[1].bulge, grid=grid_basis_plot)


"""
__Source Flux__

A key quantity for a source galaxy is its total flux, which can be used to compute magnitudes (see 
`autolens_workspace/*/guides/units/flux`) example for more details on this).

The most simple way to compute the total flux of a light profile is to create a grid of (y,x) coordinates over which
we compute the image of the light profile, and then sum the image. 

The units of the light profile `intensity` are the units of the data the light profile was fitted to. In this example
we will assume everything is in electrons per second (`e- s^-1`), which is typical for Hubble Space Telescope imaging data.
"""
print(f"Source Galaxy's Intensity {source_galaxy.bulge.intensity} e- s^-1")

"""
The total flux, in units of `e- s^-1` , is computed by summing the image of the light profile over all pixels.

Note that we can use a `grid` of any shape and pixel scale here, the important thing is that it is so large
and high enough resolution that it captures all the light from the light profile.

Note that we are using the source galaxy's true light profile, which corresponds to its emission in the source-plane.
For real datasets, we have to infer this via lens modeling.
"""
grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

source_galaxy = tracer.galaxies[1]

image = source_galaxy.bulge.image_2d_from(grid=grid)

total_flux = np.sum(image)  # in units e- s^-1 as summed over pixels

print(f"Total Source Flux: {total_flux} e- s^-1")

"""
Below, we will compare how this true source flux compares to the inferred source fluxes we compute using different
source modeling techniques (e.g. parametric and pixelized source models). Converting the flux to magnitudes or
other quantities used for tasks like SED fitting is described in the `autolens_workspace/*/guides/units/flux` example.

__Source Magnification__

The overall magnification of the source is estimated as the ratio of total surface brightness in the image-plane and 
total surface brightness in the source-plane.

Note that the surface brightness is different to the total flux above, as surface brightness is flux per unit area. 
We therefore explicitly mention how area folds into the calculation below.

To ensure the magnification is stable and that we resolve all source emission in both the image-plane and source-plane 
we use a very high resolution grid, higher than we used to compute the total flux above.
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

To calculation this, we first ray-trace the grid above from the image-plane to the source-plane using the tracer
and then pass it to the source galaxy's light profile to compute the lensed image.
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
__Tracer__

Lens modeling returns a `max_log_likelihood_tracer`, which is likely the object you have at hand to compute
source science calculations for real datasets.

The code below shows how using a tracer, composed of any combination of lens and source galaxies, we can
compute the source flux and magnification. It reproduces the calculations above.
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

Below, we compute it at three (y,x) coordinates on the lensed arcs of the simulated dataset.

For a real lens, pass the `max_log_likelihood_tracer` of your lens model and coordinates measured from your data.
For a multi-plane system (for example a cluster with sources at several redshifts), the magnification must be
computed up to the plane of the source of interest with `plane_j`; see the `Trap (b)` section of
`autolens_workspace/*/guides/advanced/multi_plane.py`.
"""
point_grid = al.Grid2DIrregular(values=[(0.3, 1.7), (-1.5, 0.2), (0.0, -1.7)])

lens_calc = al.LensCalc.from_tracer(tracer=tracer)

magnification_at_points = lens_calc.magnification_2d_via_hessian_from(grid=point_grid)

print(f"Magnification At Points (signed): {magnification_at_points}")
print(f"Magnification At Points (absolute): {abs(magnification_at_points)}")


"""
__Parametric Source Models__

If your lens modeling uses a parametric source model (e.g. Sersic, Multi Gaussian Expansion), the only object
you need to perform source science calculations is the `max_log_likelihood_tracer` returned by lens modeling.

Alternatively, as done above, you can manually set up a tracer using the lens and source galaxies inferred
by lens modeling.

Therefore, you may now wish to go to your results, extract the `max_log_likelihood_tracer`, and use it to compute
the source flux and magnification as shown above.

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
run. On a laptop CPU the fit takes about 8 minutes and the 100 draws below about 8 minutes (roughly 5 seconds each).
The `autolens_workspace/*/guides/results` examples describe how to load results and draw from a posterior in more
detail.

__Model Fit__

We fit an isothermal mass with external shear and an MGE source, set up with the `mge_model_from` utility used by
`autolens_workspace/*/imaging/modeling`. The search is a quick Nautilus run with 100 live points. A positions
likelihood, which requires the multiple images of the source to trace to the same location in the source plane, stops
the fit from settling on a solution where the lens has the wrong Einstein radius.

The simulated dataset contains a nearby galaxy that is not part of the lens model, so its pixels are given large
noise values, as in the modeling examples.
"""
import autofit as af

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
dataset = dataset.apply_mask(mask=mask)

lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)
field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
    use_jax=True,
)

search = af.Nautilus(
    path_prefix=Path("source_science"),
    name="magnification_errors_mge",
    unique_tag=dataset_name,
    n_live=100,
)

result = search.fit(model=model, analysis=analysis)

"""
__Posterior Draws__

We now draw models from the posterior. For each one we rebuild the tracer, with the solved intensities of the MGE
Gaussians included, and recompute the source magnification and the point magnification at the multiple image
positions.

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

    point_magnification_list.append(
        np.asarray(lens_calc.magnification_2d_via_hessian_from(grid=positions))
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
]:
    lower, median, upper = np.percentile(np.asarray(values), [16.0, 50.0, 84.0], axis=0)

    print(f"{name}: median {np.round(median, 2)}")
    print(f"    68% interval {np.round(lower, 2)} to {np.round(upper, 2)}")
    print(f"    fractional error {np.round((upper - lower) / 2.0 / np.abs(median), 3)}")

"""
On this dataset the fractional errors are about 1% for the source magnification and between 2% and 3% for the point
magnifications.

In our run of this script the point magnifications came out about 5% below the true values of the simulated lens
(10.07 and -9.21), about twice their errors, even though the Einstein radius and shear were recovered accurately. The
errors describe the uncertainty of the fitted model only: changing the source from the Sersic used to simulate the data
to an MGE shifts the point magnifications by more than that uncertainty, a reminder that comparing source models is
part of estimating magnification errors.

The errors here are smaller than LEGGOS II find (3% to 9%) because the dataset is simulated with a high
signal-to-noise source and a simple lens. Real lenses, and especially cluster lenses with many mass components, have
more complex mass distributions and fainter data, so their magnification errors are larger.

More draws give a smoother estimate of the percentiles: LEGGOS II use about 300, and 100 is enough here because the
errors are only used to illustrate the calculation. The errors also only include the uncertainty of the model that was
fitted: a different mass model (for example a power law instead of an isothermal) can change the magnifications by
more than these errors.

__Wrap Up__

In this example, the MGE Source gave very similar results for the source flux and magnification as did the Sersic
example illustrated in the `autolens_workspace/*/imaging/source_science.py` example.

In real lenses, this is not always the case, and if you are really interested in precise source science calculations
which estimate systematic uncertainties, you should explore how different source models impact these calculations.
"""
