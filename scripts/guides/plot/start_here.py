"""
Plots: Start Here
=================

This example introduces the PyAutoLens plotting API.

Plotting is performed by standalone functions in the `autolens.plot` module, imported as `aplt`:

 - `aplt.plot_array()` — plot any 2D array (data, images, convergence maps, residuals, etc.).
 - `aplt.plot_grid()` — plot a 2D grid of (y,x) coordinates.
 - `aplt.subplot_tracer()`, `aplt.subplot_imaging_dataset()`, etc. — multi-panel subplots for standard objects.

The API follows a simple pattern: quantities are computed from PyAutoLens objects via their methods
(e.g. `tracer.image_2d_from(grid=grid)`) and the resulting array or grid is passed to a plotting
function. This means anything the library can compute, you can plot, without needing a dedicated
plotting class for every object.

Figure appearance (titles, colormaps, log10 scaling, output to disk) is customized by passing
keyword arguments directly to the plotting functions, with project-wide defaults set via config
files.

__Contents__

- **Dataset:** Load the strong lens dataset and set up the tracer used throughout this example.
- **plot_array:** The fundamental function for plotting any 2D array.
- **plot_grid:** Plot 2D (y,x) coordinate grids, including ray-traced source-plane grids.
- **Customization:** Titles, colormaps, log10 scaling and value limits via keyword arguments.
- **Output:** Save figures to disk in one or more formats, with control over path and filename.
- **Config Defaults:** Project-wide default appearance via `config/visualize/`.
- **Overlays:** Overlay critical curves, caustics and positions using `lines=` and `positions=`.
- **subplot_* Functions:** Multi-panel overviews of datasets, tracers and galaxies.
- **Where To Next:** Object-by-object figures, fit plotting per dataset type and search results.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

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
if al.util.dataset.should_simulate(str(dataset_path)):
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

This is the pattern used throughout PyAutoLens: compute the quantity you want via the object's
method, then pass it to `plot_array`.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Tracer Image")
aplt.plot_array(array=tracer.convergence_2d_from(grid=grid), title="Convergence")
aplt.plot_array(array=tracer.potential_2d_from(grid=grid), title="Potential")

"""
__plot_grid__

The `aplt.plot_grid()` function displays a 2D grid of (y,x) coordinates.

This is useful for visualizing image-plane and source-plane grids, for example a uniform image-plane
grid and the same grid after it has been ray-traced to the source plane.
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
 - `vmin` / `vmax`: The minimum and maximum values of the colormap scale.

The colormap accepts any valid matplotlib colormap name.
"""
aplt.plot_array(array=data, title="Jet Colormap", colormap="jet")
aplt.plot_array(array=data, title="Hot Colormap", colormap="hot")
aplt.plot_array(array=data, title="Gray Colormap", colormap="gray")

"""
Many lensing quantities (images, convergence, potential) span several orders of magnitude and are
easier to interpret in log10 space, which `use_log10=True` provides.
"""
aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Tracer Image (Log10)",
    use_log10=True,
)

aplt.plot_array(
    array=tracer.convergence_2d_from(grid=grid),
    title="Convergence (Log10)",
    use_log10=True,
)

"""
The `vmin` / `vmax` keywords fix the colormap limits, which is useful for comparing figures on the
same scale.
"""
aplt.plot_array(array=data, title="Data (Fixed Scale)", vmin=0.0, vmax=1.0)

"""
__Output__

By default (with no `output_path` input), figures are displayed on screen.

To save a figure to disk instead, pass `output_path` (a directory) and `output_format`. The file
is saved as `{output_path}/{title}.{output_format}`, so the title doubles as the filename unless
an explicit `output_filename` is given.
"""
aplt.plot_array(
    array=data,
    title="example",
    output_path=Path("output") / "plot",
    output_format="png",
)

"""
Multiple formats can be specified as a list to save the same figure in each format at once, for
example a .png for quick inspection alongside a .pdf for publication.
"""
aplt.plot_array(
    array=data,
    title="example",
    output_path=Path("output") / "plot",
    output_format=["png", "pdf"],
)

"""
__Config Defaults__

When no explicit keyword is passed to a plotting function the default value is read from the
config files in:

  autolens_workspace/config/visualize/

Key entries in `config/visualize/general.yaml` include:

 - `colormap`: The default colormap of all 2D plots.
 - `general` -> `output_format`: The default output behaviour ("show", "png", "pdf", ...).
 - `subplot_shape_to_figsize_factor`: The scaling of subplot figure sizes.
 - `ticks` -> `number_of_ticks_2d`: The number of ticks on each spatial axis.
 - `colorbar` -> `labelsize`: The font size of colorbar tick labels.
 - `units` -> `cb_unit`: The unit label of the colorbar.

This allows the default appearance to be controlled project-wide without changing code.

The separate `config/visualize/plots.yaml` file controls which figures are output automatically
during a model-fit — see the `__Visualizer__` documentation at the end of `scripts/imaging/plot.py`.

__Overlays__

Overlays are added to plots using the `lines=` and `positions=` keyword arguments:

 - `lines=`: A list of `Grid2DIrregular` objects drawn as lines (e.g. critical curves, caustics).
 - `positions=`: A `Grid2DIrregular` object drawn as scatter points (e.g. image positions).
"""
lens_calc = al.LensCalc.from_tracer(tracer=tracer)
tangential_critical_curve_list = lens_calc.tangential_critical_curve_list_from(
    grid=grid
)
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
The full range of overlays (critical curves, caustics, multiple-image positions, profile centres
and combinations of them) is documented in `scripts/guides/plot/visuals.py`.

__subplot_* Functions__

For standard objects (datasets, tracers, galaxies), dedicated subplot functions produce
multi-panel overviews automatically.
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

"""
__Where To Next__

- **Object-by-object figures** (light profiles, mass profiles, galaxies, deflection angles,
  magnification, 1D radial profiles): `scripts/guides/plot/plotters.py`.

- **Fit plotting** (e.g. `FitImaging` residuals, chi-squared maps and fit subplots) is documented
  per dataset type in each dataset package's `plot.py` example, e.g. `scripts/imaging/plot.py`,
  `scripts/interferometer/plot.py` and `scripts/point_source/plot.py`. Each of these also documents
  the `Visualizer`, which outputs these figures automatically during a model-fit.

- **Non-linear search results** (corner plots via `aplt.corner_anesthetic`, `aplt.corner_cornerpy`
  and search-specific visualization): `scripts/guides/plot/searches.py`.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
