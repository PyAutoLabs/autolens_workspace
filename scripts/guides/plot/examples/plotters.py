"""
Plots: Plotters
===============

This example illustrates the new plotting API for all key PyAutoLens objects.

The old API used dedicated `*Plotter` classes (e.g. `Tracer`, `Imaging`,
`FitImaging`, `LightProfile`, `MassProfilePlotter`, etc.). These have all been removed.

The new API uses:

 - `aplt.plot_array(array, title, ...)` — plot any 2D array.
 - `aplt.plot_grid(grid, title, ...)` — plot a grid of coordinates.
 - `aplt.subplot_imaging_dataset(dataset)` — multi-panel dataset overview.
 - `aplt.subplot_tracer(tracer, grid)` — multi-panel tracer overview.
 - `aplt.subplot_fit_imaging(fit)` — multi-panel fit overview.
 - `aplt.subplot_interferometer_dirty_images(dataset)` — interferometer dataset overview.
 - `aplt.subplot_fit_interferometer(fit)` — interferometer fit overview.
 - `aplt.subplot_galaxies_images(tracer, grid)` — per-plane images.
 - `aplt.subplot_fit_point(fit)` — point source fit overview.

__Start Here Notebook__

Refer to `plots/start_here.ipynb` for an introduction to the new plotting API.

__Contents__

**Setup:** General setup for the analysis.
**Array2D:** Any `Array2D` — images, convergence, noise-maps, etc.
**Grid2D:** A `Grid2D` of (y,x) coordinates is plotted with `aplt.plot_grid()`.
**Tracer:** Tracer quantities (image, convergence, potential, deflections, magnification) are computed via.
**Imaging Dataset:** An `Imaging` dataset's data, noise-map and PSF are plotted individually with `aplt.plot_array()`.
**Fit Imaging:** A fit's residuals, chi-squared, model image, etc.
**Light Profile:** A light profile image is computed via `image_2d_from()` and plotted with `aplt.plot_array()`.
**Mass Profile:** Mass profile quantities are computed and plotted individually.
**Galaxy:** A galaxy's image and mass quantities are computed and plotted with `aplt.plot_array()`.
**Interferometer:** Interferometer datasets and fits are plotted using their dedicated subplot functions.

__Setup__

Set up standard objects used throughout this example.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Array2D__

Any `Array2D` — images, convergence, noise-maps, etc. — is plotted with `aplt.plot_array()`.
"""
aplt.plot_array(array=dataset.data, title="Data")
aplt.plot_array(array=dataset.noise_map, title="Noise Map")

"""
__Grid2D__

A `Grid2D` of (y,x) coordinates is plotted with `aplt.plot_grid()`.
"""
aplt.plot_grid(grid=grid, title="Uniform Grid")

"""
A ray-traced (lensed) grid can be computed and plotted.
"""
deflections = tracer.deflections_yx_2d_from(grid=grid)
lensed_grid = grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)
aplt.plot_grid(grid=lensed_grid, title="Lensed Grid")

"""
__Tracer__

Tracer quantities (image, convergence, potential, deflections, magnification) are computed
via method calls and plotted with `aplt.plot_array()`.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Tracer Image")
aplt.plot_array(array=tracer.convergence_2d_from(grid=grid), title="Convergence")
aplt.plot_array(array=tracer.potential_2d_from(grid=grid), title="Potential")

deflections_yx = tracer.deflections_yx_2d_from(grid=grid)

import autoarray as aa

aplt.plot_array(
    array=aa.Array2D(values=deflections_yx.slim[:, 0], mask=grid.mask),
    title="Deflections Y",
)
aplt.plot_array(
    array=aa.Array2D(values=deflections_yx.slim[:, 1], mask=grid.mask),
    title="Deflections X",
)
lens_calc = al.LensCalc.from_tracer(tracer=tracer)
aplt.plot_array(array=lens_calc.magnification_2d_from(grid=grid), title="Magnification")

"""
A multi-panel subplot of the tracer is produced with `aplt.subplot_tracer()`.
"""
aplt.subplot_tracer(tracer=tracer, grid=grid)

"""
A subplot of the per-plane images is produced with `aplt.subplot_galaxies_images()`.
"""
aplt.subplot_galaxies_images(tracer=tracer, grid=grid)

"""
The source-plane image (plane index 1) is accessed via the image list.
"""
aplt.plot_array(
    array=tracer.image_2d_list_from(grid=grid)[1],
    title="Source Plane Image",
)

"""
__Imaging Dataset__

An `Imaging` dataset's data, noise-map and PSF are plotted individually with `aplt.plot_array()`.
"""
aplt.plot_array(array=dataset.data, title="Data")
aplt.plot_array(array=dataset.noise_map, title="Noise Map")
aplt.plot_array(array=dataset.psf.kernel, title="PSF")

"""
A multi-panel subplot of the dataset is produced with `aplt.subplot_imaging_dataset()`.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Fit Imaging__

A fit's residuals, chi-squared, model image, etc. are accessed as attributes and plotted
with `aplt.plot_array()`.
"""
aplt.plot_array(array=fit.data, title="Data")
aplt.plot_array(array=fit.noise_map, title="Noise Map")
aplt.plot_array(array=fit.signal_to_noise_map, title="Signal-to-Noise Map")
aplt.plot_array(array=fit.model_data, title="Model Image")
aplt.plot_array(array=fit.residual_map, title="Residual Map")
aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")
aplt.plot_array(array=fit.chi_squared_map, title="Chi-Squared Map")

"""
Per-plane model images are accessed via `model_images_of_planes_list`.
"""
aplt.plot_array(array=fit.model_images_of_planes_list[0], title="Plane 0 Model Image")
aplt.plot_array(array=fit.model_images_of_planes_list[1], title="Plane 1 Model Image")

"""
A multi-panel fit subplot is produced with `aplt.subplot_fit_imaging()`.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
__Light Profile__

A light profile image is computed via `image_2d_from()` and plotted with `aplt.plot_array()`.
"""
bulge = tracer.galaxies[0].bulge
aplt.plot_array(array=bulge.image_2d_from(grid=grid), title="Bulge Image")

"""
__Mass Profile__

Mass profile quantities are computed and plotted individually.
"""
mass = tracer.galaxies[0].mass
aplt.plot_array(array=mass.convergence_2d_from(grid=grid), title="Mass Convergence")
aplt.plot_array(array=mass.potential_2d_from(grid=grid), title="Mass Potential")

mass_deflections = mass.deflections_yx_2d_from(grid=grid)
aplt.plot_array(
    array=aa.Array2D(values=mass_deflections.slim[:, 0], mask=grid.mask),
    title="Mass Deflections Y",
)
aplt.plot_array(
    array=aa.Array2D(values=mass_deflections.slim[:, 1], mask=grid.mask),
    title="Mass Deflections X",
)

"""
__Galaxy__

A galaxy's image and mass quantities are computed and plotted with `aplt.plot_array()`.
"""
galaxy = tracer.galaxies[0]
aplt.plot_array(array=galaxy.image_2d_from(grid=grid), title="Galaxy Image")
aplt.plot_array(array=galaxy.convergence_2d_from(grid=grid), title="Galaxy Convergence")

"""
__1D Profiles__

1D radial profiles are computed using a projected 2D grid and plotted with matplotlib directly.

There is no 1D plotting function in the new API — use matplotlib.
"""
grid_2d_projected = grid.grid_2d_radial_projected_from(
    centre=galaxy.bulge.centre, angle=bulge.angle()
)

image_1d = galaxy.bulge.image_2d_from(grid=grid_2d_projected)

plt.plot(grid_2d_projected[:, 1], image_1d)
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.show()
plt.close()

"""
Using a radial grid of (y,x) coordinates along the x-axis plots the 1D radial profile.
"""
radii = np.arange(10000) * 0.01
grid_radial = al.Grid2DIrregular(values=[(0.0, r) for r in radii])
image_1d = bulge.image_2d_from(grid=grid_radial)

plt.plot(radii, image_1d)
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.show()
plt.close()

"""
__Interferometer__

Interferometer datasets and fits are plotted using their dedicated subplot functions.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator.py"],
        check=True,
    )

real_space_mask = al.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

aplt.subplot_fit_interferometer(fit=fit)

"""
__Point Dataset / Fit__

A point source fit is plotted with `aplt.subplot_fit_point()`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "point_source" / dataset_name

dataset = al.from_json(
    file_path=Path(dataset_path, "point_dataset_positions_only.json"),
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.8,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0, point_0=al.ps.PointFlux(centre=(0.0, 0.0), flux=0.8)
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

point_grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,
)

solver = al.PointSolver.for_grid(
    grid=point_grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

fit = al.FitPointDataset(dataset=dataset, tracer=tracer, solver=solver)

aplt.subplot_fit_point(fit=fit)

"""
Finish.
"""
