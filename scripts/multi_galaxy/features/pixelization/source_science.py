"""
Source Science: Pixelization (Multi Galaxy)
===========================================

Read the [feature introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/pixelization/start_here.ipynb) for the shared concepts and a guide to the examples.

Source science studies the highly magnified background source rather than the deflectors. This script computes
the quantities a pixelized reconstruction gives access to for a multi-galaxy lens:

 - The total flux of the reconstructed source.
 - Its magnification, from the combined deflection of both deflectors.
 - Its intrinsic morphology, interpolated onto a uniform grid.
 - The errors on every source pixel.

__Contents__

- **Dataset:** Load the multi-galaxy dataset that is fitted.
- **Extra Galaxies Noise Scaling:** Scale the contaminating galaxy's light out of the fit.
- **Mask, Centres & Over Sampling:** Standard set up, over-sampled at every deflector centre.
- **Model Fit:** Fit the lens with a pixelized source.
- **Inversion:** The object holding the reconstruction.
- **Source Flux:** Sum the reconstructed source pixel fluxes.
- **Source Magnification:** The ratio of image-plane to source-plane flux.
- **Omitting A Deflector:** What the same calculation gives with one deflector's mass left out.
- **Interpolated Source:** Put the reconstruction on a uniform grid.
- **Zoom:** The same, at higher resolution over the source region.
- **Errors:** The reconstruction's noise map.
- **Parametric Comparison:** The flux a Sersic fit to the same source gives.
- **Magnification via Mesh:** The magnification computed directly on the mesh of the source reconstruction.
- **Source-Plane Magnification Map:** The magnification of every pixel of the source reconstruction's mesh.
- **True Magnification:** The simulator's true magnification, to check the reconstruction against.
- **Magnification At A Point:** The signed magnification `mu = 1 / det(A)` at chosen (y,x) image-plane coordinates, computed via the Hessian.
- **Magnification Errors:** Errors on the magnifications above, from a quick lens model fit and draws from its posterior (the only slow part of the script, so it runs last).
- **Magnification Uncertainty Map:** The error of the magnification at every image-plane pixel, from the same posterior draws.
- **Wrap Up:** Where to go next.

__What Changes For Multiple Deflectors__

Every quantity below is computed from the fit, and the fit's traced grid uses the summed deflection of both
deflectors. So the magnification, and everything derived from it, depends on both mass models — the
`Omitting A Deflector` section runs the same calculation with one left out so the size of that dependence is
visible in the printed output.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/source_science.py` for the parametric-source version, and
`imaging/features/pixelization/source_science.py` for the galaxy-scale walkthrough.

Interpolation grid points outside the mesh have no reconstructed source light, so `fill_value=0.0` sets them
to zero rather than `NaN`, which would otherwise make every sum below `NaN`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `simple` multi-galaxy dataset, the same co-dominant pair fitted by `multi_galaxy/modeling.py`.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/multi_galaxy/simulator.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Extra Galaxies Noise Scaling__

Scale the faint contaminant out of the fit, as `multi_galaxy/modeling.py` explains. Flux left in the data would
otherwise be reconstructed as source structure and counted in the fluxes below.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.
"""
mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    ),
    over_sample_size_pixelization=4,
)

"""
__Model Fit__

The two co-dominant deflectors and a pixelized source, at the values used by `multi_galaxy/simulator.py`. A real
analysis would take these from a model-fit result instead — `multi_galaxy/features/pixelization/modeling.py`
produces one.
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

field = al.MassField(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

mesh_shape = (28, 28)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_0, lens_1, source_galaxy], fields=[field])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
__Inversion__

The inversion holds everything about the reconstruction. The `Mapper` inside it holds the source pixels'
source-plane positions.
"""
inversion = fit.inversion

mapper = inversion.cls_list_from(cls=al.Mapper)[0]

"""
__Source Flux__

The total flux of the reconstructed source is the sum of its pixel fluxes, in the same units as the data
(typically electrons per second).
"""
reconstruction = inversion.reconstruction

total_source_flux = np.sum(reconstruction)

print(f"Total Source Flux via Pixelization: {total_source_flux} e- s^-1")

"""
The source-plane positions of those pixels. Where they sit is set by the traced grid, so both deflectors'
masses determine them.
"""
source_plane_mesh_grid = mapper.source_plane_mesh_grid

print(f"Source Plane Mesh Grid: {source_plane_mesh_grid}")

"""
__Source Magnification__

The magnification is the ratio of the total flux in the image plane to the total flux in the source plane.

The image-plane flux is the reconstruction mapped back through the mapping matrix. The source-plane flux needs
the reconstruction on a grid with known pixel areas, which is what the interpolation below provides.
"""
mapped_reconstructed_operated_data = inversion.mapped_reconstructed_operated_data

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

print(f"Source Magnification (both deflectors): {magnification}")

"""
__Omitting A Deflector__

The same calculation with `lens_1` left out of the tracer. Its mass no longer contributes to the deflection
field, so the traced grid, the source pixel positions and the magnification all change.

Run this whenever you are unsure how much a given deflector matters to a source-plane quantity you are quoting.
"""
tracer_lens_0_only = al.Tracer(galaxies=[lens_0, source_galaxy], fields=[field])

fit_lens_0_only = al.FitImaging(dataset=dataset, tracer=tracer_lens_0_only)

inversion_lens_0_only = fit_lens_0_only.inversion
reconstruction_lens_0_only = inversion_lens_0_only.reconstruction
mapped_lens_0_only = inversion_lens_0_only.mapped_reconstructed_operated_data

mapper_lens_0_only = inversion_lens_0_only.cls_list_from(cls=al.Mapper)[0]
source_plane_mesh_grid_lens_0_only = mapper_lens_0_only.source_plane_mesh_grid

interpolated_lens_0_only = griddata(
    points=source_plane_mesh_grid_lens_0_only,
    values=reconstruction_lens_0_only,
    xi=interpolation_grid,
    fill_value=0.0,
)

interpolated_lens_0_only_ndarray = interpolated_lens_0_only.reshape(
    interpolation_grid.shape_native
)

interpolated_lens_0_only = al.Array2D.no_mask(
    values=interpolated_lens_0_only_ndarray,
    pixel_scales=interpolation_grid.pixel_scales,
)

magnification_lens_0_only = np.sum(
    mapped_lens_0_only * mapped_lens_0_only.pixel_area
) / np.sum(interpolated_lens_0_only * interpolated_lens_0_only.pixel_area)

print(f"Source Magnification (lens_0 only): {magnification_lens_0_only}")
print(
    f"Magnification difference when omitting lens_1: "
    f"{magnification - magnification_lens_0_only:.4f}"
)

"""
__Interpolated Source__

The interpolated reconstruction is a regular 2D array, so standard image-analysis tools apply to it.
"""
aplt.plot_array(array=interpolated_reconstruction, title="Interpolated Source")

"""
__Zoom__

The same interpolation over a smaller extent, at higher resolution, for the source region itself.
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

Every source pixel has an error, held on the inversion's reconstruction noise map. Interpolating it the same way
gives an error map alongside the source image, which is what propagating an uncertainty onto any of the
quantities above requires.

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

The flux of the analytic source used to simulate this dataset, computed on a fine grid. Comparing it to the
pixelized total above is the check that the reconstruction is not missing or inventing flux.
"""
source_parametric = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
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
__Source-Plane Magnification Map__

The magnification of every mesh pixel, the image-plane area mapping to it divided by its source-plane area, is the
mapper's `mesh_magnification`. `plot_magnification_source_plane` colours the reconstruction's mesh by it, with the
caustics of both deflectors drawn on top. The `imaging/features/pixelization/source_science.py` example explains the map.
"""
aplt.plot_magnification_source_plane(mapper=mapper, mass_obj=tracer, grid=dataset.grid)

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

image_grid = al.Grid2D.from_mask(mask=dataset.mask, over_sample_size=8)
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

For a multi-galaxy lens, every mass profile in the tracer (both deflectors and the external shear) enters the
Hessian, so neither deflector has a point magnification of its own. Below, we compute it at two (y,x) coordinates on
the lensed arcs, with both deflectors and with `lens_1` omitted.

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

magnification_at_points_lens_0_only = al.LensCalc.from_tracer(
    tracer=tracer_lens_0_only
).magnification_2d_via_hessian_from(grid=point_grid)

print(
    f"Magnification At Points, lens_0 only (signed): {magnification_at_points_lens_0_only}"
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

We fit a model of the two lens galaxies, each with Sersic light and an isothermal mass, an external shear held at
the system centre and the source reconstructed on the same 28 x 28 `RectangularBilinearAdaptDensity` mesh used
above, with constant regularization whose coefficient is a free parameter. The light profiles are linear, so their intensities are solved for rather than
sampled, which keeps the number of free parameters down. The search is a quick Nautilus run with 200 live points, the
number `multi_galaxy/modeling` uses for its two-deflector model. A positions likelihood, which requires the multiple
images of the source to trace to the same location in the source plane, stops the fit from settling on a solution
where the lens has the wrong Einstein radius.
"""
import autofit as af

lens_0 = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp_linear.Sersic, mass=al.mp.Isothermal
)
lens_1 = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp_linear.Sersic, mass=al.mp.Isothermal
)

field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

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
magnification_map_list = []

map_grid = dataset.grid

start = time.time()

for _ in range(total_draws):
    instance = result.samples.draw_randomly_via_pdf()

    fit_draw = analysis.fit_from(instance=instance)

    inversion_draw = fit_draw.inversion
    mapper_draw = inversion_draw.cls_list_from(cls=al.Mapper)[0]

    # The linear light profiles of the two deflectors are solved for alongside the source, so we take
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

    magnification_map_list.append(lens_calc.magnification_2d_from(grid=map_grid))

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

Where to go next:

 - `multi_galaxy/source_science.py` — the same quantities from a parametric source fit.
 - `multi_galaxy/features/pixelization/modeling.py` — producing the result a real analysis would take the
   galaxies from.
 - `multi_galaxy/features/pixelization/plot.py` — plotting the inversion objects used above.
 - `imaging/features/pixelization/source_science.py` — the galaxy-scale walkthrough, including its
   reconstruction CSV output.

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
