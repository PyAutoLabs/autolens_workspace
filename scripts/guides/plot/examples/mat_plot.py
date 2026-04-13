"""
Plots: Customization
====================

This example illustrates how to customize the appearance of figures in the new plotting API.

In the old API, customization was done via a `MatPlot2D` object passed to a `*Plotter` class.
Both `MatPlot2D` and all `*Plotter` classes have been removed.

In the new API, customization is done by passing keyword arguments directly to `aplt.plot_array()`,
`aplt.plot_grid()`, and `aplt.subplot_*()` functions.

The main customization kwargs are:

 - `title`: Figure title string.
 - `colormap`: Matplotlib colormap name (e.g. "jet", "gray", "hot").
 - `use_log10`: Plot colormap in log10 scale.
 - `output_path`: Directory path for saving the figure.
 - `output_format`: File format, e.g. "png" or "pdf".

__Start Here Notebook__

Refer to `plots/start_here.ipynb` for a general introduction to the new plotting API.

__Contents__

**Setup:** General setup for the analysis.
**Output:** To save a figure to disk, pass `output_path` (a directory) and `output_format`.
**Title:** The figure title is set with the `title=` kwarg.
**Colormap:** The colormap is set with the `colormap=` kwarg.
**Log10:** Many lensing quantities (images, convergence, potential) span many orders of magnitude and are.
**Config Defaults:** All default values (colormaps, tick sizes, label fonts, etc.) are configured via the config files.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Setup__

Load a dataset to illustrate customization.
"""
dataset_path = Path("dataset") / "imaging" / "slacs1430+4105"
data_path = dataset_path / "data.fits"
data = al.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.03)

"""
__Output__

To save a figure to disk, pass `output_path` (a directory) and `output_format`.

The file is saved as `{output_path}/{title}.{output_format}` by default.
"""
aplt.plot_array(
    array=data,
    title="example",
    output_path=Path("notebooks") / "plot" / "plots",
    output_format="png",
)

"""
Multiple formats can be specified as a list to save the same figure in multiple formats.
"""
aplt.plot_array(
    array=data,
    title="example",
    output_path=Path("notebooks") / "plot" / "plots",
    output_format=["png", "pdf"],
)

"""
To display the figure on screen (instead of saving it), omit `output_path`.

This is also the default behaviour when no `output_path` is provided.
"""
aplt.plot_array(array=data, title="Data")

"""
__Title__

The figure title is set with the `title=` kwarg.
"""
aplt.plot_array(array=data, title="This is the title")

"""
__Colormap__

The colormap is set with the `colormap=` kwarg.

Any valid matplotlib colormap name can be used.
"""
aplt.plot_array(array=data, title="Jet Colormap", colormap="jet")
aplt.plot_array(array=data, title="Hot Colormap", colormap="hot")
aplt.plot_array(array=data, title="Gray Colormap", colormap="gray")

"""
__Log10__

Many lensing quantities (images, convergence, potential) span many orders of magnitude and are
easier to interpret in log10 space.

Pass `use_log10=True` to plot in log10 scale.
"""
aplt.plot_array(array=data, title="Data (Log10)", use_log10=True)

"""
Log10 is particularly useful for convergence and potential maps.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.0, 0.0)),
)
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCoreSph(
        centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5, sersic_index=2.0
    ),
)
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

aplt.plot_array(
    array=tracer.convergence_2d_from(grid=grid),
    title="Convergence (Log10)",
    use_log10=True,
)

"""
__Config Defaults__

All default values (colormaps, tick sizes, label fonts, etc.) are configured via the config files:

  autolens_workspace/config/visualize/

Key config files and entries:

 - `mat_wrap.yaml` -> Figure -> figure: -> figsize
 - `mat_wrap.yaml` -> YLabel -> figure: -> fontsize
 - `mat_wrap.yaml` -> XLabel -> figure: -> fontsize
 - `mat_wrap.yaml` -> TickParams -> figure: -> labelsize
 - `mat_wrap.yaml` -> YTicks -> figure: -> labelsize
 - `mat_wrap.yaml` -> XTicks -> figure: -> labelsize

When no explicit keyword is passed to a plotting function, the config value is used.
This allows project-wide defaults to be set without changing code.
"""

"""
Finish.
"""
