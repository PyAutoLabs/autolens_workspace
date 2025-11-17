"""
Plots: Mat Plot
===============

This example illustrates the API for customizing the appearance of figures and subplots using the `MatPlot` object,
which wraps the `matplotlib` methods used to display figures.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how visuals work and the default
behaviour of plotting visuals.

__Contents__

**Setup:** Set up all objects (e.g. grid, tracer, data) used to illustrate plotting.
**

__Setup__

To illustrate plotting, we require standard objects like a grid, tracer and dataset.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt

dataset_path = Path("dataset") / "slacs" / "slacs1430+4105"
data_path = dataset_path / "data.fits"
data = al.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.03)

"""
__Units__

The source code has internal units which are used for calculations and model-fitting (e.g. arc-seconds,
electrons per second, dimensionless mass units)

Visualization is performed in these internal units, however, the `Units` object allows the user to customize the
units of the figure (e.g. convert arc-seconds to kiloparsecs).

However, how to use the `Units` object to customize the units of a figure is not described here, but instead
in`autolens_workspace/*/guides/results/examples/units_and_cosmology.ipynb`. Unit conversions in general is
described in this script.

__Output__

We can specify the output of the figure using the `Output` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.show.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.savefig.html

Below, we specify that the figure should be output as a `.png` file, with the name `example.png` in the `plot/plots` 
folder of the workspace.
"""
output = aplt.Output(
    path=Path("notebooks") / "plot" / "plots", filename="example", format="png"
)

mat_plot = aplt.MatPlot2D(output=output)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
We can specify a list of output formats, such that the figure is output to all of them.
"""
output = aplt.Output(
    path=Path("notebooks") / "plot" / "plots",
    filename="example",
    format=["png", "pdf"],
)

mat_plot = aplt.MatPlot2D(output=output)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
This `Output` object does not display the figure on your computer screen, bypassing this to output the `.png`. This is
the default behaviour of PyAutoLens plots, but can be manually specified using the `format-"show"`
"""
output = aplt.Output(format="show")

mat_plot = aplt.MatPlot2D(output=output)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()


"""
__Figure__

We can customize the figure using the `Figure` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.figure.html
"""
figure = aplt.Figure(
    figsize=(7, 7),
    dpi=100.0,
    facecolor="white",
    edgecolor="black",
    frameon=True,
    clear=False,
    tight_layout=False,
    constrained_layout=False,
)

mat_plot = aplt.MatPlot2D(figure=figure)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
We can also customize the aspect ratio of the image displayed in a figure by passing the `Figure` an aspect ratio. 

This customizes the aspect ratio when the method `plt.imshow` is called.

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
"""
figure = aplt.Figure(aspect="square")

mat_plot = aplt.MatPlot2D(figure=figure)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Labels__

We can customize the title, y and x labels using the `Title`, `YLabel` and `XLabel` matplotlib wrapper object 
which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.title.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.ylabel.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.xlabel.html

We can manually specify the label of the title, ylabel and xlabel.
"""
title = aplt.Title(label="This is the title", color="r", fontsize=20)

ylabel = aplt.YLabel(ylabel="Label of Y", color="b", fontsize=5, position=(0.2, 0.5))

xlabel = aplt.XLabel(xlabel="Label of X", color="g", fontsize=10)

mat_plot = aplt.MatPlot2D(title=title, ylabel=ylabel, xlabel=xlabel)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
If we do not manually specify a label, the name of the function used to plot the image will be used as the title 
and the units of the image will be used for the ylabel and xlabel.
"""
title = aplt.Title()
ylabel = aplt.YLabel()
xlabel = aplt.XLabel()

mat_plot = aplt.MatPlot2D(title=title, ylabel=ylabel, xlabel=xlabel)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The units can be manually specified.
"""
mat_plot = aplt.MatPlot2D(units=aplt.Units(in_kpc=True, ticks_convert_factor=5.0))

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()


"""
__Ticks__

We can customize the ticks using the `YTicks` and `XTicks matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.tick_params.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.yticks.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.xticks.html
"""
tickparams = aplt.TickParams(
    axis="y",
    which="major",
    direction="out",
    color="b",
    labelsize=20,
    labelcolor="r",
    length=2,
    pad=5,
    width=3,
    grid_alpha=0.8,
)

yticks = aplt.YTicks(alpha=0.8, fontsize=10, rotation="vertical")
xticks = aplt.XTicks(alpha=0.5, fontsize=5, rotation="horizontal")

mat_plot = aplt.MatPlot2D(tickparams=tickparams, yticks=yticks, xticks=xticks)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
A suffix can be added to every tick, for example the arc-seconds units can be included.

This means one does not need to include the "x (arcsec)" and "y (arcsec)" labels on the plot, saving space for
publication figures (see how to remove labels in `Labels.py`.
"""
yticks = aplt.YTicks(manual_suffix='"')
xticks = aplt.XTicks(manual_suffix='"')

mat_plot = aplt.MatPlot2D(yticks=yticks, xticks=xticks)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()


"""
Ticks and their labels can be removed altogether by the following code:
"""
tickparams = aplt.TickParams(
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
    labelright=False,
    labeltop=False,
)

mat_plot = aplt.MatPlot2D(tickparams=tickparams, yticks=yticks, xticks=xticks)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Axis__

We can customize the figure using the `Axis` matplotlib wrapper object which wraps the following method(s):

 plt.axis: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axis.html
"""
array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()

axis = aplt.Axis(extent=[-1.0, 1.0, -1.0, 1.0])

mat_plot = aplt.MatPlot2D(axis=axis)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__CMap__

We can customize the colormap using the `Cmap` matplotlib wrapper object which wraps the following method(s):

 colors.Linear: https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html
 colors.LogNorm: https://matplotlib.org/3.3.2/tutorials/colors/colormapnorms.html
 colors.SymLogNorm: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.colors.SymLogNorm.html

The colormap is used in various functions that plot images with a `cmap`, most notably `plt.imshow`:

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html

First, lets plot the image using a linear colormap which uses a `colors.Normalize` object.
"""
cmap = aplt.Cmap(cmap="jet", norm="linear", vmin=0.0, vmax=1.0)

mat_plot = aplt.MatPlot2D(cmap=cmap)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
We can instead use logarithmic colormap (this wraps the `colors.LogNorm` matplotlib object).
"""
cmap = aplt.Cmap(cmap="hot", norm="log", vmin=0.0, vmax=2.0)

mat_plot = aplt.MatPlot2D(cmap=cmap)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
We can use a symmetric log norm (this wraps the `colors.SymLogNorm` matplotlib object).
"""
cmap = aplt.Cmap(
    cmap="twilight",
    norm="symmetric_log",
    vmin=0.0,
    vmax=1.0,
    linthresh=0.05,
    linscale=0.1,
)

mat_plot = aplt.MatPlot2D(cmap=cmap)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The diverge normalization ensures the colorbar is centred around 0.0, irrespective of whether vmin and vmax are input.
"""
cmap = aplt.Cmap(cmap="twilight", norm="diverge")

mat_plot = aplt.MatPlot2D(cmap=cmap)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Colorbar__

We can customize the colorbar using the `Colorbar` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.colorbar.html
"""
cb = aplt.Colorbar(
    fraction=0.047,
    shrink=5.0,
    aspect=1.0,
    pad=0.01,
    anchor=(0.0, 0.5),
    panchor=(1.0, 0.0),
)

mat_plot = aplt.MatPlot2D(colorbar=cb)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The labels of the `Colorbar` can also be customized. 

This uses the `cb.ax.set_yticklabels` to manually override the tick locations and labels:
 
 https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html
 
The input parameters of both the above methods can be passed into the `Colorbar` object.
"""
cb = aplt.Colorbar(manual_tick_labels=[1.0, 2.0], manual_tick_values=[0.0, 0.25])


mat_plot = aplt.MatPlot2D(colorbar=cb)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__ColorbarTickParams__

We can customize the colorbar ticks using the `ColorbarTickParams` matplotlib wrapper object which wraps the 
following method of the matplotlib colorbar:

 https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
"""
colorbar_tickparams = aplt.ColorbarTickParams(
    axis="both",
    reset=False,
    which="major",
    direction="in",
    length=2,
    width=2,
    color="r",
    pad=0.1,
    labelsize=10,
    labelcolor="r",
)

mat_plot = aplt.MatPlot2D(colorbar_tickparams=colorbar_tickparams)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The colorbar tick parameters can be removed altogether with the following:
"""
colorbar_tickparams = aplt.ColorbarTickParams(
    bottom=False, top=False, left=False, right=False
)

mat_plot = aplt.MatPlot2D(colorbar_tickparams=colorbar_tickparams)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Scatter__

The appearance of a (y,x) `Grid2D` of coordinates is customized using `Scatter` objects. To illustrate this, we will 
customize the appearance of the (y,x) origin on a figure using an `OriginScatter` object.

To plot a (y,x) grids of coordinates (like an origin) these objects wrap the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
origin_scatter = aplt.OriginScatter(marker="o", s=50)

mat_plot = aplt.MatPlot2D(origin_scatter=origin_scatter)

visuals = aplt.Visuals2D(origin=al.Grid2DIrregular(values=[(0.0, 0.0)]))

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
There are numerous (y,x) grids of coordinates that plots. For example, in addition to the origin,
there are grids like the multiple images of a strong lens, a source-plane grid of traced coordinates, etc.

All of these grids are plotted using a `Scatter` object and they are described in more detail in the 
`plot/visuals_2d` example scripts. 

__Annotate__

We can customize the annotations using the `Annotate` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.annotate.html

We manually specify the annotation using the input `s` and locations `x` and `y`, which means a solid white line
will appear.
"""
annotate = aplt.Annotate(
    text="",
    xy=(2.0, 2.0),
    xytext=(3.0, 2.0),
    arrowprops=dict(arrowstyle="-", color="w"),
)

mat_plot = aplt.MatPlot2D(annotate=annotate)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
By passing a list of `Annotate` objects we can plot multiple text extracts.
"""
annotate_0 = aplt.Annotate(
    text="",
    xy=(2.0, 2.0),
    xytext=(3.0, 2.0),
    arrowprops=dict(arrowstyle="-", color="w"),
)
annotate_1 = aplt.Annotate(
    text="",
    xy=(1.0, 1.0),
    xytext=(2.0, 1.0),
    arrowprops=dict(arrowstyle="-", color="r"),
)

mat_plot = aplt.MatPlot2D(annotate=[annotate_0, annotate_1])

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Text__

We can customize the text using the `Text` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html

We manually specify the text using the input `s` and locations `x` and `y`.
"""
text = aplt.Text(s="Example text", x=0, y=0, color="r", fontsize=20)

mat_plot = aplt.MatPlot2D(text=text)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
By passing a list of `Text` objects we can plot multiple text extracts.
"""
text_0 = aplt.Text(s="Example text 0", x=0, y=0, color="r", fontsize=20)
text_1 = aplt.Text(s="Example text 1", x=10, y=10, color="b", fontsize=20)

mat_plot = aplt.MatPlot2D(text=[text_0, text_1])

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Contour__

We can customize the contour using the `Contour` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.contour.html

We first create a light profile which we will use to plot contours over a image map.
"""
light = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=0.1,
    effective_radius=1.0,
    sersic_index=4.0,
)

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

image = light.image_2d_from(grid=grid)

contour = aplt.Contour(colors="k")

mat_plot = aplt.MatPlot2D(contour=contour)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
By default, the contour levels are computed automatically from the minimum and maximum values of the array. 

They are then plotted in 10 intervals spaced evenly in log10 values between these limits.

The number of contour levels and use of linear spacing can be manually input.
"""
contour = aplt.Contour(colors="k", total_contours=5, use_log10=False)

mat_plot = aplt.MatPlot2D(contour=contour)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The levels can also be manually specified using the `manual_levels` input.
"""
contour = aplt.Contour(manual_levels=[0.1, 0.5, 10.0])

mat_plot = aplt.MatPlot2D(contour=contour)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
By default, the value of each contour is shown on the figure.

This can be disabled using the `include_values` input.
"""
contour = aplt.Contour(include_values=False)

mat_plot = aplt.MatPlot2D(contour=contour)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Legend__

We can customize the legend using the `Legend` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.legend.html
"""
legend = aplt.Legend(include=True, loc="upper left", fontsize=10, ncol=2)

mat_plot = aplt.MatPlot2D(legend=legend)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__FillBetween__

We next illustrate how to fill between two lines on a 1D Matplotlib figure and subplots.

First, lets create 1D data for the plot.
"""
y = al.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0, 5.0], pixel_scales=1.0)
x = al.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0, 5.0], pixel_scales=1.0)

"""
We now pass this y and x data to a `YX1DPlotter` and call the `figure` method.
"""
yx_plotter = aplt.YX1DPlotter(y=y, x=x)
yx_plotter.figure_1d()

"""
We can fill between two lines line on the y vs x plot using the `FillBetween` matplotlib wrapper object which wraps 
the following method(s):

 plt.fill_between: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.fill_between.html

The input`match_color_to_yx=True` ensures the filled area is the same as the line that is plotted, as opposed
to a different color. This ensures that shaded regions can easily be paired in color to the plot when it is 
appropriate (e.g. plotted the errors of the line).
"""
y1 = al.Array1D.no_mask(values=[0.5, 1.5, 2.5, 3.5, 4.5], pixel_scales=1.0)
y2 = al.Array1D.no_mask(values=[1.5, 2.5, 3.5, 4.5, 5.5], pixel_scales=1.0)

mat_plot = aplt.MatPlot1D(
    fill_between=aplt.FillBetween(match_color_to_yx=True, alpha=0.3)
)

visuals = aplt.Visuals1D(shaded_region=[y1, y2])

yx_plotter = aplt.YX1DPlotter(y=y, x=x, mat_plot_1d=mat_plot, visuals_1d=visuals)
yx_plotter.figure_1d()

"""
__AXVLine__

We now illustrate how to plot vertical lines on a 1D Matplotlib figure and subplots.
"""
y = al.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0, 5.0], pixel_scales=1.0)
x = al.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0, 5.0], pixel_scales=1.0)

"""
We now pass this y and x data to a `YX1DPlotter` and call the `figure` method.
"""
yx_plotter = aplt.YX1DPlotter(y=y, x=x)
yx_plotter.figure_1d()

"""
We can plot a vertical line on the y vs x plot using the `AXVLine` matplotlib wrapper object which wraps the 
following method(s):

 plt.axvline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html
"""
# mat_plot = aplt.MatPlot1D(axv)

yx_plotter = aplt.YX1DPlotter(y=y, x=x)
yx_plotter.figure_1d()

"""
Finish.
"""
