"""
Plots: Publication Image
========================

Scientific papers have specific requirements on producing plots and figures so that they look good in the paper.
This includes large labels, clear axis ticks and minimizing white space.

This example illustrates how to plot an image-plane image (e.g. the observed data of a strong lens, or the
image-plane model-image of a fit) with `Plotter` settings that are suitable for a paper.

Note that all of these settings are defaults in PyAutoLens, so you do not need to use them specifically in order
to make paper-quality plots. However, they are described here such that you can customize them for your own plots
if you so wish!

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
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

First, lets load example imaging of of a strong lens as an `Imaging` object.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    radius=3.0, shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales
)

dataset = dataset.apply_mask(mask=mask)

"""
We now pass the imaging to an `ImagingPlotter` and plot its `image`, to show what a default image looks like
before we make changes that make it more suitable for publication.

(The default settings of PyAutoLens visualization have been updated to be more suitable for publication, so
the example below actually fixes many of the issues described below. I have simply not updated the example
to reflect this).
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(
    data=True,
)

"""
__Y,X Label / Ticks and Title__

The y / x labels and ticks, and the title, are not ideal for a paper figure:
 
 - The fontsize of the title and ticks are small and may be difficult to read in a paper.
 - The title "Image" is not descriptive.
 - The labels take up a lot of whitespace, requiring the labels "y (arcsec)" and "x (arcsec") next to the ticks. 
 - The ticks go from -3.1 to 3.1, which is an unround number.
 - The y ticks require extra whitespace because they are written horizontally.
 - The numerical tick values at the bottom-left of the figure (the -3.1 ticks for y and x) may overlap with one
   another if the fontsize is increased.
   
We address all of the issues below, lets take a look at the figure before discussing the changes.
"""
mat_plot_ticks = aplt.MatPlot2D(
    title=aplt.Title(label=f"Image Illustrating Publication Plot", fontsize=24),
    yticks=aplt.YTicks(
        fontsize=22,
        manual_suffix='"',
        manual_values=[-2.5, 0.0, 2.5],
        rotation="vertical",
        va="center",
    ),
    xticks=aplt.XTicks(fontsize=22, manual_suffix='"', manual_values=[-2.5, 0.0, 2.5]),
    ylabel=aplt.YLabel(ylabel=""),
    xlabel=aplt.XLabel(xlabel=""),
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot_ticks)
dataset_plotter.figures_2d(
    data=True,
)

"""
The new figure improves all of the issues above:

 - The title is larger and more descriptive.
 - The ticks fontsizes are bigger, ensuring they will be large and readable for the paper.  
 - The arcseconds units are now a " symbol next to each tick value, therefore the "y (arcsec)" and "x (arcsec)" labels 
   are removed, avoiding unused whitespace.
 - By specifying the ticks `manual_values=[-2.5, 0.0, 2.5]`, we remove the unround 3.1 and -3.1 values and do not
   have two -3.1 values overlapping one another in the bottom left corner.
 - The y ticks are now vertical, removing unused whitespace.

__Colorbar__

The colorbar is also not ideal for a paper:

 - The fontsize of the color tick values are small and may be difficult to read in a paper.
 - The ticks are horizontal, taking up extra whitespace.

We address these issues below, lets take a look at the figure before discussing the changes.

NOTE: In order to retain the changes to the title, labels and ticks above, we add the `MatPlot2D` objects together.
If you are unfamiliar with this API, checkout the example ? for a discussion, but in brief adding `MatPlot2D` objects
together is equivalent to specifying all the inputs in a single `MatPlot2D` object.
"""
mat_plot_2d_cb = aplt.MatPlot2D(
    colorbar=aplt.Colorbar(
        manual_tick_values=[0.0, 0.3, 0.6], manual_tick_labels=[0.0, 0.3, 0.6]
    ),
    colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
)

mat_plot = mat_plot_ticks + mat_plot_2d_cb

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.figures_2d(
    data=True,
)

"""
The new colorbar improves on the previous one:

 - The colorbar tick label fontsize is bigger, which will be readable in a paper.
 - The veritcal tick labels saves on unused whitespace.

__Y,X Ticks Alternative__

Below, we show a different visualization where the information contained in the Y and X ticks is expressed in a more 
compressed way, removing white space:
"""
xpos = 2.0
ypos = -2.9
yposoff = 0.4
xposoff = -0.1
length = 1.0

mat_plot_ticks = aplt.MatPlot2D(
    title=aplt.Title(label=f"Image Illustrating Publication Plot", fontsize=24),
    xticks=aplt.XTicks(manual_values=[]),
    yticks=aplt.YTicks(manual_values=[]),
    ylabel=aplt.YLabel(ylabel=""),
    xlabel=aplt.XLabel(xlabel=""),
    text=aplt.Text(s='1.0"', x=xpos, y=ypos, c="w", fontsize=30),
    annotate=aplt.Annotate(
        text="",
        xy=(xpos + xposoff, ypos + yposoff),
        xytext=(xpos + xposoff + length, ypos + yposoff),
        arrowprops=dict(arrowstyle="-", color="w"),
    ),
)

mat_plot = mat_plot_ticks + mat_plot_2d_cb

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.figures_2d(
    data=True,
)

"""
The ticks are completely removed and replaced with a white line in the image, which makes there even less unused 
white space.

This is only possible because the y and x axis had the same scale (e.g. -3.0" -> 3.0"). If each axis spanned a 
different range this information would be lost in this visual.

__Output__

We output the visual to both .pdf and .png. For a publication, we recommend you use .pdf, as it is a higher quality
image format. However, .pngs may be easier to quickly inspect on your computer as they are supported by more visual 
software packages.

We also specify the following inputs:

 - `format_folder`: Images are output in separate folders based on format called `png` and `pdf`, which can be useful
   for file management.
 - `bbox_inches`: Uses the matplotlib input `plt.savefig(bbox_inches="tight")` to remove edge whitespace before
   outputting the file.
"""

mat_plot_2d_output = aplt.MatPlot2D(
    output=aplt.Output(
        filename=f"image_publication",
        path=path.join("scripts", "plot", "publication"),
        format=["png", "pdf"],
        format_folder=True,
        bbox_inches="tight",
    )
)

mat_plot = mat_plot + mat_plot_2d_output

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.figures_2d(
    data=True,
)
