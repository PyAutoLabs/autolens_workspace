"""
Plots: Start Here
=================

This example introduces the new plotting API in PyAutoLens.

The old API (removed) used `*Plotter` classes (e.g. `Imaging`, `Tracer`) together with
`MatPlot2D` and `Visuals2D` helper objects. These have all been removed.

The new API uses standalone functions:

 - `aplt.plot_array()` — plot any 2D array.
 - `aplt.plot_grid()` — plot a 2D grid of (y,x) coordinates.
 - `aplt.subplot_tracer()`, `aplt.subplot_fit_imaging()`, etc. — multi-panel subplots for standard objects.

__Contents__

- **Dataset**: Load objects used to illustrate plotting.
- **plot_array**: The fundamental function for 2D visualization.
- **plot_grid**: Plot a Grid2D of coordinates.
- **Customization**: Pass title, colormap, use_log10, output_path, output_format directly.
- **Config Defaults**: Adjust defaults via config files.
- **Overlays**: Use `lines=` and `positions=` to add overlays.
- **subplot_* Functions**: Multi-panel subplots for standard objects.
- **What Is Gone**: MatPlot2D, Visuals2D, and all *Plotter classes removed.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load an example imaging dataset and set up objects used throughout this example.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "scripts/imaging/features/no_lens_light/simulator.py"],
        check=True,
    )

data_path = dataset_path / "data.fits"
data = al.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.1)

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.2, 0.2)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCoreSph(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__plot_array__

The fundamental plotting function is `aplt.plot_array()`, which displays any 2D `Array2D`.

We can plot the raw data array loaded from a .fits file.
"""
aplt.plot_array(array=data, title="Data")

"""
We can also plot quantities computed from a tracer, such as its image, convergence and potential.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Tracer Image")
aplt.plot_array(array=tracer.convergence_2d_from(grid=grid), title="Convergence")
aplt.plot_array(array=tracer.potential_2d_from(grid=grid), title="Potential")

"""
__plot_grid__

The `aplt.plot_grid()` function displays a 2D grid of (y,x) coordinates.

This is useful for visualizing image-plane and source-plane grids.
"""
aplt.plot_grid(grid=grid, title="Uniform Grid")

traced_grid = tracer.traced_grid_2d_list_from(grid=grid)[1]
aplt.plot_grid(grid=traced_grid, title="Source-Plane Grid")

"""
__Customization__

Each plotting function accepts direct keyword arguments for customization:

 - `title`: The figure title string.
 - `colormap`: The matplotlib colormap name (e.g. "jet", "hot", "gray").
 - `use_log10`: If True, the colormap is plotted in log10 scale.
 - `output_path`: Directory path to save the figure on disk.
 - `output_format`: Format of the saved file, e.g. "png" or "pdf".

These replace the old `MatPlot2D` object entirely — there is no `MatPlot2D` anymore.
"""
aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Tracer Image (Log10)",
    use_log10=True,
)

aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Tracer Image (Jet Colormap)",
    colormap="jet",
)

"""
To save a figure to disk, pass `output_path` and `output_format`.
"""
aplt.plot_array(
    array=data,
    title="Data Saved to Disk",
    output_path=Path("output"),
    output_format="png",
)

"""
__Config Defaults__

All default plotting values are configured via config files in:

  autolens_workspace/config/visualize/

When no explicit keyword is passed to a plotting function the config value is used, allowing
the default appearance to be controlled project-wide without changing code.
"""

"""
__Overlays__

Overlays are added to plots using the `lines=` and `positions=` keyword arguments:

 - `lines=`: A list of `Grid2DIrregular` objects drawn as lines (e.g. critical curves, caustics).
 - `positions=`: An `Grid2DIrregular` object drawn as scatter points (e.g. image positions).

These replace the old `Visuals2D` object entirely — there is no `Visuals2D` anymore.
"""
lens_calc = al.LensCalc.from_tracer(tracer=tracer)
tangential_critical_curve_list = lens_calc.tangential_critical_curve_list_from(grid=grid)
tangential_caustic_list = lens_calc.tangential_caustic_list_from(grid=grid)

aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Image with Critical Curves",
    lines=tangential_critical_curve_list,
)

source_image = tracer.image_2d_list_from(grid=grid)[1]
aplt.plot_array(
    array=source_image,
    title="Source Plane with Caustics",
    lines=tangential_caustic_list,
)

positions = al.Grid2DIrregular(values=[(1.0, 1.0), (2.0, 2.0), (-1.0, 0.5)])
aplt.plot_array(
    array=data,
    title="Data with Positions",
    positions=positions,
)

"""
__subplot_* Functions__

For standard objects (datasets, tracers, fits), dedicated subplot functions produce
multi-panel overviews automatically.

These replace all the old `*Plotter` class `.subplot_*()` method calls.
"""
dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

aplt.subplot_tracer(tracer=tracer, grid=grid)

aplt.subplot_galaxies_images(tracer=tracer, grid=grid)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)
dataset = dataset.apply_mask(mask=mask)
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
The search plotting functions (`aplt.corner_anesthetic`, `aplt.corner_cornerpy`, etc.) provide
visualization of non-linear search results -- see `scripts/guides/plot/examples/searches.py`.
"""
