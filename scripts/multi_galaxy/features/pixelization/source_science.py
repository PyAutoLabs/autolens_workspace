"""
Source Science: Pixelization (Multi Galaxy)
===========================================

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
- **Magnification via Mesh:** The magnification from the mesh pixel areas directly.
- **Wrap Up:** Where to go next.

__What Changes For Multiple Deflectors__

Every quantity below is computed from the fit, and the fit's traced grid uses the summed deflection of both
deflectors. So the magnification, and everything derived from it, depends on both mass models — the
`Omitting A Deflector` section runs the same calculation with one left out so the size of that dependence is
visible in the printed output.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/source_science.py` for the parametric-source version, and
`imaging/features/pixelization/source_science.py` for the galaxy-scale walkthrough.
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

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

mesh_shape = (28, 28)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source_galaxy])

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
    points=source_plane_mesh_grid, values=reconstruction, xi=interpolation_grid
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
tracer_lens_0_only = al.Tracer(galaxies=[lens_0, shear_galaxy, source_galaxy])

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

The interpolation above introduces its own error. Using the mesh pixel areas directly avoids it, at the cost of
having no regular grid to plot.
"""
mesh_areas = mapper.mesh_geometry.areas_for_magnification

magnification_mesh = np.sum(
    mapped_reconstructed_operated_data * mapped_reconstructed_operated_data.pixel_area
) / np.sum(reconstruction * mesh_areas)

print(f"Magnification via Mesh Areas: {magnification_mesh}")

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/source_science.py` — the same quantities from a parametric source fit.
 - `multi_galaxy/features/pixelization/modeling.py` — producing the result a real analysis would take the
   galaxies from.
 - `multi_galaxy/features/pixelization/plot.py` — plotting the inversion objects used above.
 - `imaging/features/pixelization/source_science.py` — the galaxy-scale walkthrough, including its
   reconstruction CSV output.
"""
