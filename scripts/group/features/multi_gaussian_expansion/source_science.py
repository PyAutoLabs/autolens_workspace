"""
Source Science: Multi Gaussian Expansion (Group)
================================================

Source science focuses on studying the highly magnified properties of the background lensed source galaxy.

Using a source galaxy model, we can compute key quantities such as the magnification, total flux, and intrinsic
size of the source.

This example shows how to perform these calculations for a group-scale lens using an MGE source model. The key
differences from the standard group source science example are:

 - The source galaxy is modeled as a Multi Gaussian Expansion (MGE), where the intensities of ~20 Gaussians are
   solved via linear algebra when fitting the data.
 - MGE sources can provide more accurate flux and magnification estimates than single parametric profiles (e.g.
   Sersic) because they capture more complex source morphologies.
 - ALL mass profiles in the group must be included when computing ray-tracing and magnification.

__Contents__

- **Simulated Dataset:** Load the group dataset.
- **Mask:** Define the 2D mask.
- **Source Values:** Set up the lens and source galaxies.
- **MGE Source:** Create an MGE source and fit it to the data to obtain intensities.
- **Source Flux:** Compute the total source flux.
- **Source Magnification:** Compute the source magnification using all group galaxies.
- **Impact of Extra Galaxies:** Show that omitting extra galaxies gives incorrect magnification.
- **Tracer:** Using the tracer from lens modeling results.
- **Magnification At A Point:** The signed magnification `mu = 1 / det(A)` at chosen (y,x) image-plane coordinates, computed via the Hessian.
- **Magnification Errors:** Errors on the magnifications above, from a quick lens model fit and draws from its posterior (the only slow part of the script, so it runs last).

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Simulated Dataset__

We load and plot the `simple` group example dataset.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
al.util.dataset.simulate_if_needed(dataset_path, simulator="scripts/group/simulator.py")

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We apply a 7.5 arcsecond circular mask.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxy Centres__

Load the centres of the main lens galaxies and extra galaxies.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__

Apply adaptive over-sampling at all galaxy centres.
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Source Values__

For a group-scale lens, we must include ALL galaxies that contribute to the lensing potential when
performing source science calculations. This includes the main lens galaxy and all extra galaxies.

We first set up the lens and extra galaxies with their mass profiles (no light, since we only need
the mass for ray-tracing in this context). We also create an MGE source whose intensities will be
determined by fitting the data.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.dPIEMassSph(
        centre=(3.5, 2.5),
        sigma=212.0,
        r_core=0.0,
        r_cut=10.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.dPIEMassSph(
        centre=(-4.4, -5.0),
        sigma=239.0,
        r_core=0.0,
        r_cut=10.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

"""
__MGE Source__

We set up the source galaxy using an MGE made of 20 Gaussians whose `sigma` values span 0.01" to the
mask radius. The intensities are linear light profiles whose values are determined by fitting the data.

The MGE source model offers advantages for source science calculations:

 - It captures more complex source morphologies than a single Sersic profile.
 - The intensities are optimized to best reconstruct the lensed source, providing a more accurate
   estimate of the source's total flux.
 - It naturally handles sources with multiple components or asymmetric features.
"""
total_gaussians = 20

log10_sigma_list = np.linspace(-4, np.log10(mask_radius), total_gaussians)

bulge_gaussian_list = []

for i in range(total_gaussians):
    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        sigma=10 ** log10_sigma_list[i],
    )

    bulge_gaussian_list.append(gaussian)

bulge = al.lp_basis.Basis(profile_list=bulge_gaussian_list)

source_galaxy = al.Galaxy(redshift=1.0, bulge=bulge)

"""
We create the tracer using ALL group galaxies and the MGE source, then fit it to the data to
solve for the Gaussian intensities.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
From the fit, extract the tracer with solved-for Gaussian intensities.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

aplt.subplot_fit_imaging(fit=fit)

"""
__Source Flux__

We compute the total source flux by summing the source galaxy's image over a high-resolution grid.
"""
source_galaxy = tracer.galaxies[-1]

grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

image = source_galaxy.bulge.image_2d_from(grid=grid)

total_flux = np.sum(image)

print(f"Total Source Flux (MGE): {total_flux} e- s^-1")

"""
__Source Magnification__

The overall magnification is the ratio of total flux in the image-plane to the source-plane.

For group-scale lenses, the ray-tracing is performed through ALL mass profiles in the group, which
the tracer handles automatically.
"""
grid = al.Grid2D.uniform(shape_native=(1000, 1000), pixel_scales=0.03)

image = source_galaxy.bulge.image_2d_from(grid=grid)

total_source_plane_flux = np.sum(image)

traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

source_plane_grid = traced_grid_list[-1]

lensed_source_image = source_galaxy.bulge.image_2d_from(grid=source_plane_grid)

total_image_plane_flux = np.sum(lensed_source_image)

source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Magnification (all group galaxies, MGE): {source_magnification}")

"""
__Impact of Extra Galaxies__

For group-scale lenses, the magnification is determined by ALL mass profiles in the group. Omitting
the extra galaxy masses gives an incorrect magnification estimate.
"""
tracer_main_only = al.Tracer(galaxies=[tracer.galaxies[0], source_galaxy])

traced_grid_list_main_only = tracer_main_only.traced_grid_2d_list_from(grid=grid)

source_plane_grid_main_only = traced_grid_list_main_only[-1]

lensed_source_image_main_only = source_galaxy.bulge.image_2d_from(
    grid=source_plane_grid_main_only
)

total_image_plane_flux_main_only = np.sum(lensed_source_image_main_only)

source_magnification_main_only = (
    total_image_plane_flux_main_only / total_source_plane_flux
)

print(f"Source Magnification (main lens only, MGE): {source_magnification_main_only}")
print(
    f"Magnification difference when omitting extra galaxies: "
    f"{source_magnification - source_magnification_main_only:.4f}"
)

"""
__Tracer__

Lens modeling returns a `max_log_likelihood_tracer`, which is the object you would use to compute
source science calculations for real datasets. The code below reproduces the calculations above
using the tracer's built-in plane functionality.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

image_plane_grid = traced_grid_list[0]
source_plane_grid = traced_grid_list[-1]

lensed_source_image = tracer.planes[-1].image_2d_from(grid=source_plane_grid)
source_plane_image = tracer.planes[-1].image_2d_from(grid=image_plane_grid)

total_image_plane_flux = np.sum(lensed_source_image)
total_source_plane_flux = np.sum(source_plane_image)

source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Plane Total Flux via Tracer (MGE): {total_source_plane_flux} e- s^-1")
print(f"Source Magnification via Tracer (MGE): {source_magnification}")

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

For group-scale lenses, every mass profile in the tracer enters the Hessian, so the extra galaxies change the
magnification at a point just as they change the overall magnification above. Near an extra galaxy they can even
change the sign (parity) of the image. Below, we compute it at three (y,x) coordinates on the lensed arcs, using
the full group and the main lens only.

For a real lens, pass the `max_log_likelihood_tracer` of your lens model and coordinates measured from your data.
For a multi-plane system (for example a cluster with sources at several redshifts), the magnification must be
computed up to the plane of the source of interest with `plane_j`; see the `Trap (b)` section of
`autolens_workspace/*/guides/advanced/multi_plane.py`.
"""
point_grid = al.Grid2DIrregular(values=[(4.4, -2.0), (-2.1, 4.5), (4.5, 2.9)])

lens_calc = al.LensCalc.from_tracer(tracer=tracer)

magnification_at_points = lens_calc.magnification_2d_via_hessian_from(grid=point_grid)

print(f"Magnification At Points (signed): {magnification_at_points}")
print(f"Magnification At Points (absolute): {abs(magnification_at_points)}")

magnification_at_points_main_only = al.LensCalc.from_tracer(
    tracer=tracer_main_only
).magnification_2d_via_hessian_from(grid=point_grid)

print(
    f"Magnification At Points, main lens only (signed): {magnification_at_points_main_only}"
)

"""
__Parametric Source Models__

If your lens modeling uses a parametric source model (e.g. Sersic or MGE), the only object you need to
perform source science calculations is the `max_log_likelihood_tracer` returned by lens modeling.

For group-scale lenses, ensure that the tracer includes all group member galaxies with their mass profiles,
as each contributes to the ray-tracing and therefore to the magnification of the source.

An MGE source may give different flux and magnification estimates compared to a single Sersic source. If
precise source science calculations are important for your analysis, we recommend comparing results from
different source models (e.g. MGE, Sersic, pixelized reconstruction) to estimate systematic uncertainties.
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

We fit the group model used in `autolens_workspace/*/group/modeling`: the main lens galaxy with Sersic light and an
isothermal mass, external shear, and the two extra galaxies with spherical Sersic light and a dPIE mass whose centre is
fixed to the galaxy's observed centre and whose velocity dispersion `sigma` is its only free mass parameter. The
source is an MGE, set up with the `mge_model_from` utility used by the modeling examples. The light profiles are linear, so their intensities are solved for rather than sampled, which
keeps the number of free parameters down.

The search is a quick Nautilus run with 150 live points, the number `group/modeling` uses for this model. A positions
likelihood, which requires the multiple images of the source to trace to the same location in the source plane,
stops the fit from settling on a solution where the lens has the wrong Einstein radius.
"""
import autofit as af

main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

lens_dict = {}

for i, centre in enumerate(main_lens_centres):
    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy, redshift=0.5, bulge=al.lp_linear.Sersic, mass=al.mp.Isothermal
    )

field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)

extra_galaxies_list = []

for centre in extra_galaxies_centres:
    bulge = af.Model(al.lp_linear.SersicSph)
    bulge.centre = centre

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre
    mass.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=300.0)
    mass.r_core = 0.0
    mass.r_cut = 10.0
    mass.redshift_object = 0.5
    mass.redshift_source = 1.0
    mass.H0 = 67.66
    mass.Om0 = 0.30966

    extra_galaxies_list.append(
        af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    )

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    fields=field,
    extra_galaxies=af.Collection(extra_galaxies_list),
)

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
    n_live=150,
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
Compare the medians with the true magnifications printed earlier in this script. The 68% intervals should contain the
true values about two-thirds of the time if the fitted model can describe the lens; offsets larger than the errors are a
sign that it cannot.

These errors are typically smaller than LEGGOS II find (3% to 9%), because the dataset is simulated with a high
signal-to-noise source. Real lenses, and especially cluster lenses with many mass components, have more complex mass
distributions and fainter data, so their magnification errors are larger.

More draws give a smoother estimate of the percentiles: LEGGOS II use about 300, and 100 is enough here because the
errors are only used to illustrate the calculation. The errors also only include the uncertainty of the model that was
fitted: a different mass model or source model can change the magnifications by more than these errors.
"""
