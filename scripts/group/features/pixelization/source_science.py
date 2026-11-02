"""
Source Science: Pixelization (Group)
====================================

Source science focuses on studying the highly magnified properties of the background lensed source galaxy.

Using a pixelized source reconstruction from a group-scale lens, we can compute key quantities such as:

 - The total flux of the reconstructed source.
 - The magnification of the source due to the combined mass of all group galaxies.
 - The intrinsic size and morphology of the source.

For group-scale lenses, ALL mass profiles in the group (main lens + extra galaxies) contribute to the
magnification. Omitting any galaxy's mass would give an incorrect magnification estimate.

This script also compares the pixelized source flux to a parametric estimate, demonstrating that the
pixelized reconstruction can recover source properties that parametric models may miss.

__Contents__

- **Dataset & Mask:** Standard set up of the group dataset and 7.5" mask.
- **Galaxy Centres:** Load centres for main lens and extra galaxies.
- **Model Fit:** Create a FitImaging with a pixelized source.
- **Source Flux:** Compute the total flux from the pixelized source reconstruction.
- **Source Magnification:** Compute the magnification using all group mass profiles.
- **Impact of Extra Galaxies:** Demonstrate how omitting extra galaxies affects magnification.
- **Interpolated Source:** Interpolate the pixelized source to a uniform grid.
- **Parametric Comparison:** Compare pixelized source flux to a parametric estimate.
- **Magnification via Mesh:** The magnification computed directly on the mesh of the source reconstruction.
- **True Magnification:** The simulator's true magnification, to check the reconstruction against.
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
__Dataset__

Load the strong lens group dataset `simple`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__
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
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Galaxy Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=4,
)

"""
__Model Fit__

We create a fit with a pixelized source using concrete galaxy objects.

For the group lens, we include the main lens galaxy, extra galaxies, and a pixelized source.
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

mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mesh = al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
__Inversion__

The inversion contains all information about the pixelized source reconstruction.
"""
inversion = fit.inversion

mapper = inversion.cls_list_from(cls=al.Mapper)[0]

"""
__Source Flux__

The total flux of the pixelized source reconstruction is the sum of all source pixel fluxes.

The units are the same as the data (typically electrons per second, e- s^-1).
"""
reconstruction = inversion.reconstruction

total_source_flux = np.sum(reconstruction)

print(f"Total Source Flux via Pixelization: {total_source_flux} e- s^-1")

"""
The source pixel positions in the source plane are also available.
"""
source_plane_mesh_grid = mapper.source_plane_mesh_grid

print(f"Source Plane Mesh Grid: {source_plane_mesh_grid}")

"""
__Source Magnification__

The overall magnification of the source is the ratio of total flux in the image plane to total flux
in the source plane.

The image-plane reconstruction (mapped back from source pixels) is available from the inversion.
"""
mapped_reconstructed_operated_data = inversion.mapped_reconstructed_operated_data

"""
To compute the magnification, we need the areas of source and image pixels.

The image-plane pixels have a uniform area defined by the dataset pixel scale.

Interpolation grid points outside the mesh have no reconstructed source light, so `fill_value=0.0` sets them
to zero rather than `NaN`, which would otherwise make every sum below `NaN`.
"""
from scipy.interpolate import griddata

interpolation_grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

interpolated_reconstruction = griddata(
    points=source_plane_mesh_grid,
    values=reconstruction,
    xi=interpolation_grid,
    fill_value=0.0,
)

interpolated_reconstruction_ndarray = interpolated_reconstruction.reshape(
    interpolation_grid.shape_native
)

interpolated_reconstruction = al.Array2D.no_mask(
    values=interpolated_reconstruction_ndarray,
    pixel_scales=interpolation_grid.pixel_scales,
)

magnification = np.sum(
    mapped_reconstructed_operated_data * mapped_reconstructed_operated_data.pixel_area
) / np.sum(interpolated_reconstruction * interpolated_reconstruction.pixel_area)

print(f"Source Magnification (all group galaxies): {magnification}")

"""
__Impact of Extra Galaxies__

For group-scale lenses, ALL mass profiles contribute to the magnification. Omitting extra galaxies
gives an incorrect estimate.

Below, we compute the magnification using only the main lens galaxy for comparison.
"""
tracer_main_only = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit_main_only = al.FitImaging(dataset=dataset, tracer=tracer_main_only)

inversion_main_only = fit_main_only.inversion
reconstruction_main_only = inversion_main_only.reconstruction
mapped_main_only = inversion_main_only.mapped_reconstructed_operated_data

mapper_main_only = inversion_main_only.cls_list_from(cls=al.Mapper)[0]
source_plane_mesh_grid_main_only = mapper_main_only.source_plane_mesh_grid

interpolated_main_only = griddata(
    points=source_plane_mesh_grid_main_only,
    values=reconstruction_main_only,
    xi=interpolation_grid,
    fill_value=0.0,
)

interpolated_main_only_ndarray = interpolated_main_only.reshape(
    interpolation_grid.shape_native
)

interpolated_main_only = al.Array2D.no_mask(
    values=interpolated_main_only_ndarray,
    pixel_scales=interpolation_grid.pixel_scales,
)

magnification_main_only = np.sum(
    mapped_main_only * mapped_main_only.pixel_area
) / np.sum(interpolated_main_only * interpolated_main_only.pixel_area)

print(f"Source Magnification (main lens only): {magnification_main_only}")
print(
    f"Magnification difference when omitting extra galaxies: "
    f"{magnification - magnification_main_only:.4f}"
)

"""
__Interpolated Source__

For detailed source science, the pixelized source can be interpolated to a uniform 2D grid.
This allows standard image analysis tools to be used.
"""
aplt.plot_array(array=interpolated_reconstruction, title="Interpolated Source")

"""
__Zoom__

We can zoom in on the source region for higher resolution.
"""
extent = (-1.0, 1.0, -1.0, 1.0)
shape_native = (401, 401)

interpolation_grid_zoom = al.Grid2D.from_extent(
    extent=extent,
    shape_native=shape_native,
)

interpolated_reconstruction_zoom = griddata(
    points=source_plane_mesh_grid,
    values=reconstruction,
    xi=interpolation_grid_zoom,
    fill_value=0.0,
)

interpolated_reconstruction_zoom_ndarray = interpolated_reconstruction_zoom.reshape(
    interpolation_grid_zoom.shape_native
)

interpolated_reconstruction_zoom = al.Array2D.no_mask(
    values=interpolated_reconstruction_zoom_ndarray,
    pixel_scales=interpolation_grid_zoom.pixel_scales,
)

aplt.plot_array(
    array=interpolated_reconstruction_zoom, title="Zoomed Interpolated Source"
)

"""
__Errors__

The reconstruction noise map provides errors on each source pixel, enabling uncertainty propagation
for source science calculations.

Note that some source-plane pixels have a noise value of `NaN`. These are the pixels the inversion never
solved for: the mesh zeroes its poorly-constrained edge pixels before solving, to stop them absorbing flux
and destabilizing the linear algebra, and their reconstruction is therefore an exact `0.0` rather than a
fitted value. A pixel with no fitted value has no uncertainty either, so the noise map reports `NaN` --
"never estimated" -- rather than a number that would look like a real error bar.

This is not something to work around; it is the reconstruction and the noise map agreeing about which
pixels were actually fitted.

Use `np.isnan(reconstruction_noise_map)` to find these pixels, and not `reconstruction == 0.0`. The two are
not the same set, and the difference is large. The inversion's non-negative solver also pins pixels it *did*
solve for at exactly `0.0` -- wherever the fit wants no flux -- and those are real fitted values with real
error bars. On a typical compact-source fit with a 28x28 mesh, 603 of the 784 source pixels read `0.0`,
while only the 108 edge pixels the mesh zeroed are `NaN`. A `NaN` noise value always means the
reconstruction is `0.0` there; a reconstruction of `0.0` does not mean the pixel was never fitted.

`NaN` propagates through interpolation, so the interpolated noise map below is blank in the region covered
by those edge pixels -- which is honest, since there is nothing there to interpolate from. If you need an
interpolated error map that covers the full field, drop the `NaN` entries from `points` and `values` before
calling `griddata` rather than filling them with a number.
"""
reconstruction_noise_map = inversion.reconstruction_noise_map

interpolated_noise_map = griddata(
    points=source_plane_mesh_grid,
    values=reconstruction_noise_map,
    xi=interpolation_grid,
)

interpolated_noise_map_ndarray = interpolated_noise_map.reshape(
    interpolation_grid.shape_native
)

interpolated_noise_map = al.Array2D.no_mask(
    values=interpolated_noise_map_ndarray, pixel_scales=interpolation_grid.pixel_scales
)

aplt.plot_array(array=interpolated_noise_map, title="Source Reconstruction Noise Map")

"""
__Parametric Comparison__

For comparison, we compute the source flux using a parametric source model (Sersic) that
approximates the true source used to simulate the dataset.

The pixelized reconstruction may recover more flux than the parametric model if the source has
complex morphology that a Sersic profile cannot capture.
"""
source_parametric = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

parametric_image = source_parametric.bulge.image_2d_from(grid=grid)
total_parametric_flux = np.sum(parametric_image)

print(f"Total Source Flux (Parametric Sersic): {total_parametric_flux} e- s^-1")
print(f"Total Source Flux (Pixelized): {total_source_flux} e- s^-1")
print(
    f"Flux Difference (Pixelized - Parametric): "
    f"{total_source_flux - total_parametric_flux:.4f} e- s^-1"
)

"""
__Magnification via Mesh__

The magnification can also be computed directly on the mesh of the source reconstruction, without an interpolation
grid. Each mesh pixel's area, given by the `areas_for_magnification` attribute of the mapper's mesh geometry, is the
integral of the function that interpolates the source between that pixel and its neighbours (its basis function).
Multiplying every pixel's value by its area and summing therefore gives the total surface brightness of the
interpolated source.

For the `RectangularBilinearAdaptDensity` mesh used here, and for the other rectangular meshes and `Delaunay`, these
areas are computed exactly. The `KNearestNeighbor` meshes compute them numerically, and `DelaunayNN` uses an
approximation good to about 0.5%. When the mesh resolves the source well this calculation recovers the true
magnification to within about 1%. It integrates the reconstruction with exactly the interpolation the fit used,
so it does not depend on the resolution or extent of an interpolation grid. The same calculation is performed
during a model-fit by the `magnification` latent variable.

The `imaging/features/pixelization/source_science.py` example explains how the areas are computed for each mesh.
"""
mesh_areas = mapper.mesh_geometry.areas_for_magnification

magnification_mesh = np.sum(
    mapped_reconstructed_operated_data * mapped_reconstructed_operated_data.pixel_area
) / np.sum(reconstruction * mesh_areas)

print(f"Magnification via Mesh Areas: {magnification_mesh}")

"""
__True Magnification__

This dataset is simulated, so its true magnification is known: the simulator's tracer is stored in the dataset
folder as `tracer.json`. We compute it the same way as above, as the total lensed source light inside the mask
divided by the total light of the source itself, using finely sampled grids so the light profiles are integrated
accurately.

Comparing the two shows how accurate the reconstruction is. The mesh areas themselves are exact or close to it, so
any difference comes from the reconstruction: regularization smooths the source, spreading some of its light onto
neighbouring pixels, which makes the source look slightly larger and the magnification slightly smaller. For a real
lens the truth is unknown, and this comparison is not possible.
"""
true_tracer = al.from_json(file_path=dataset_path / "tracer.json")

image_grid = al.Grid2D.from_mask(mask=mask, over_sample_size=8)
lensed_source_image = true_tracer.image_2d_list_from(grid=image_grid)[-1]

source_grid = al.Grid2D.uniform(
    shape_native=(1000, 1000), pixel_scales=0.01, over_sample_size=4
)
source_image = sum(
    galaxy.image_2d_from(grid=source_grid) for galaxy in true_tracer.planes[-1]
)

true_magnification = np.sum(
    lensed_source_image * lensed_source_image.pixel_area
) / np.sum(source_image * source_image.pixel_area)

print(f"True Magnification: {true_magnification}")

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
source is reconstructed on the same 28 x 28 `RectangularBilinearAdaptDensity` mesh used above, with constant
regularization whose coefficient is a free parameter. The light profiles are linear, so their intensities are solved for rather than sampled, which
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

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

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
    name="magnification_errors",
    unique_tag=dataset_name,
    n_live=150,
)

result = search.fit(model=model, analysis=analysis)

"""
__Posterior Draws__

For each drawn model we fit the data again, which solves for the source reconstruction of that model, and recompute
the magnification via the mesh and the point magnification at the multiple image positions. The source reconstruction
is therefore redone for every draw, so its uncertainty is included in the errors.

The point magnification is computed for the plane at the source's redshift, which `plane_index_via_redshift_from`
converts to the plane's index. For a system with sources at several redshifts (for example a cluster), change
`source_redshift` to compute it for a different source.
"""
import time

source_redshift = 1.0

total_draws = 100

magnification_via_mesh_list = []
point_magnification_list = []

start = time.time()

for _ in range(total_draws):
    instance = result.samples.draw_randomly_via_pdf()

    fit_draw = analysis.fit_from(instance=instance)

    inversion_draw = fit_draw.inversion
    mapper_draw = inversion_draw.cls_list_from(cls=al.Mapper)[0]

    # The linear light profiles of the lens and extra galaxies are solved for alongside the source, so we take
    # the source's part of the reconstruction and of the image it maps to.
    mapped_reconstructed_operated_data_draw = (
        inversion_draw.mapped_reconstructed_operated_data_dict[mapper_draw]
    )

    magnification_via_mesh_list.append(
        np.sum(
            mapped_reconstructed_operated_data_draw
            * mapped_reconstructed_operated_data_draw.pixel_area
        )
        / np.sum(
            inversion_draw.reconstruction_dict[mapper_draw]
            * mapper_draw.mesh_geometry.areas_for_magnification
        )
    )

    plane_index = fit_draw.tracer.plane_index_via_redshift_from(
        redshift=source_redshift
    )

    lens_calc = al.LensCalc.from_tracer(tracer=fit_draw.tracer, plane_j=plane_index)

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
    ("Magnification via Mesh", magnification_via_mesh_list),
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

The errors also only include the uncertainty of the model that was fitted: a different mass model (for example a power
law instead of an isothermal), mesh or regularization can change the magnifications by more than these errors.

__Wrap Up__

This script demonstrated source science calculations using a pixelized source reconstruction for a
group-scale lens.

Key points:
 - The total source flux is the sum of all reconstructed source pixel values.
 - Magnification is computed as the ratio of image-plane to source-plane flux.
 - ALL group mass profiles must be included for accurate magnification.
 - The interpolated source can be analyzed using standard image tools.
 - Pixelized reconstructions may recover more flux than parametric models for complex sources.
"""
