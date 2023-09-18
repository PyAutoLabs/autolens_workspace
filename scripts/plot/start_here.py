"""
Plots: Start Here
=================

This example illustrates the basic API for plotting.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load an example image of of a strong lens as an `Array2D`.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)
data_path = path.join(dataset_path, "data.fits")
data = al.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.1)

"""
__Figures__

We now pass the array to an `Array2DPlotter` and call the `figure` method.

The `autolens.workspace.*.plot.plotters` illustrates every `Plotter` object, for 
example `ImagingPlotter`, `LightProfilePlotter`, etc.
"""
array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()

"""
__Plot Customization via MatPlot__

You can customize a number of matplotlib setup options using a `MatPlot` object, which 
wraps the `matplotlib` methods used to display the image.

(For example, the `Figure` class wraps the `matplotlib` method `plt.figure()`, whereas the `YTicks` class wraps
`plt.yticks`).

The `autolens.workspace.*.plot.mat_wrap` illustrates every `MatPlot` object, for 
example `Figure`, `YTicks`, etc.
"""
mat_plot = aplt.MatPlot2D(
    figure=aplt.Figure(figsize=(7, 7)),
    yticks=aplt.YTicks(fontsize=8),
    xticks=aplt.XTicks(fontsize=8),
    title=aplt.Title(fontsize=12),
    ylabel=aplt.YLabel(fontsize=6),
    xlabel=aplt.XLabel(fontsize=6),
)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Configs__

All matplotlib options can be customized via the config files, such that those values are used every time.

Checkout the `mat_wrap.yaml`, `mat_wrap_1d.yaml` and `mat_wrap_2d.yaml` files 
in `autolens_workspace/config/visualize/mat_wrap`.

All default matplotlib values are here. There are a lot of entries, so lets focus on whats important for displaying 
figures:

 - mat_wrap.yaml -> Figure -> figure: -> figsize
 - mat_wrap.yaml -> YLabel -> figure: -> fontsize
 - mat_wrap.yaml -> XLabel -> figure: -> fontsize
 - mat_wrap.yaml -> TickParams -> figure: -> labelsize
 - mat_wrap.yaml -> YTicks -> figure: -> labelsize
 - mat_wrap.yaml -> XTicks -> figure: -> labelsize

__Subplots__

In addition to plotting individual `figures`, **PyAutoLens** can also plot `subplots` which are again customized via
the `mat_plot` objects.

__Visuals__

Visuals can be added to any figure, using standard quantities.

For example, we can plot a mask on the image above using a `Visuals2D` object.

The `autolens.workspace.*.plot.visuals_2d` illustrates every `Visuals` object, for 
example `MaskScatter`, `LightProfileCentreScatter`, etc.
"""
mask = al.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)

visuals = aplt.Visuals2D(mask=mask)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
__Searches__

Model-fits using a non-linear search (e.g. Nautilus, Emcee) produce search-specific visualization.

The `autolens.workspace.*.plot.search` illustrates how to perform this visualization for every search (e.g.
`NautilusPlotter`, `EmceePlotter`.

__Adding Plotter Objects Together__

The `MatPlot` objects can be added together. 

This is useful when we want to perform multiple visualizations which share the same base settings, but have
individually tailored settings:
"""
mat_plot_base = aplt.MatPlot2D(
    yticks=aplt.YTicks(fontsize=18),
    xticks=aplt.XTicks(fontsize=18),
    ylabel=aplt.YLabel(ylabel=""),
    xlabel=aplt.XLabel(xlabel=""),
)

mat_plot = aplt.MatPlot2D(
    title=aplt.Title(label="Example Figure 1"),
)

mat_plot = mat_plot + mat_plot_base

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

mat_plot = aplt.MatPlot2D(
    title=aplt.Title(label="Example Figure 2"),
)

mat_plot = mat_plot + mat_plot_base

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

mat_plot = mat_plot + mat_plot_base


"""
The `Visuals` objects can also be added together.
"""
light_profile_centres = al.Grid2DIrregular(values=[(1.0, 0.0), (0.0, 1.0)])

visuals_2d_0 = aplt.Visuals2D(mask=mask)
visuals_2d_1 = aplt.Visuals2D(light_profile_centres=light_profile_centres)

visuals = visuals_2d_0 + visuals_2d_1

array_plotter = aplt.Array2DPlotter(
    array=data, visuals_2d=visuals, mat_plot_2d=aplt.MatPlot2D()
)
array_plotter.figure_2d()

"""
Finish.
"""
