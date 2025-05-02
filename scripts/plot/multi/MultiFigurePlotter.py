"""
Plots: MultiFigurePlotter
=========================

This example illustrates how to plot figures from different plotters on the same subplot, assuming that the same
type of `Plotter` and figure is being plotted.

An example of when to use this plotter would be when two different datasets (e.g. at different wavelengths) are loaded
and visualized, and the images of each dataset are plotted on the same subplot side-by-side. This is the example we
will use in this example script.

This uses a `MultiFigurePlotter` object, which requires only a list of imaging datasets and `ImagingPlotter` objects
to be passed to it. The `MultiFigurePlotter` object then plots the same figure from each `ImagingPlotter` on the same
subplot.

The script `MultiSubplot.py` illustrates a similar example, but a more general use-case where different figures
from different plotters are plotted on the same subplot. This script offers a more concise way of plotting the same
figures on the same subplot, but is less general.

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

Load the multi-wavelength `lens_sersic` datasets, which we visualize in this example script.
"""
color_list = ["g", "r"]

pixel_scales_list = [0.08, 0.12]

dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "lens_sersic"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

dataset_list = [
    al.Imaging.from_fits(
        data_path=path.join(dataset_path, f"{color}_data.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

"""
__Plot__

Plot the subhplot of each `Imaging` dataset individually using an `ImagingPlotter` object.
"""
for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Multi Plot__

We now pass the list of `ImagingPlotter` objects to a `MultiFigurePlotter` object, which we use to plot the 
image of each dataset on the same subplot.

The `MultiFigurePlotter` object uses the `subplot_of_figure` method to plot the same figure from each `ImagingPlotter`,
with the inputs:

 - `func_name`: The name of the function used to plot the figure in the `ImagingPlotter` (e.g. `figures_2d`).
 - `figure_name`: The name of the figure plotted by the function (e.g. `image`).

"""
imaging_plotter_list = [
    aplt.ImagingPlotter(dataset=dataset) for dataset in dataset_list
]

multi_figure_plotter = aplt.MultiFigurePlotter(plotter_list=imaging_plotter_list)

multi_figure_plotter.subplot_of_figure(func_name="figures_2d", figure_name="data")

"""
__Multi Fits__

The `MultiFigurePlotter` object can also output a list of figures to a single `.fits` file, where each image 
goes in each hdu extension as it is called.

The interface is similiar to above using a list of plotters, however the inputs to `output_to_fits` are different
and are:

 - `func_name_list`: The list of the function names used to plot the figure in the `ImagingPlotter` (e.g. `figures_2d`).
 - `figure_name_list`: The list of the figure names plotted by the function (e.g. `data`).
 - `filename`: The name of the `.fits` file written to the output path.
 - `tag_list`: The list of tags used to name the `.fits` file extensions.
 - `remove_fits_first`: If the `.fits` file already exists, should it be overwritten?

A `func_name` must be given for every `figure_name` and vice versa, so that the `MultiFigurePlotter` knows which
pair of inputs to use.

Therefore in the example below we input `figures_2d` twice in `func_name_list`, which correspond to the calls
`figures_2d(data=True`) and `figures_2d(noise_map=True)` in the `ImagingPlotter` objects.   
"""
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path="."))

imaging_plotter_list = [
    aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot_2d)
    for dataset in dataset_list
]

multi_plotter = aplt.MultiFigurePlotter(
    plotter_list=imaging_plotter_list,
)

multi_plotter.output_to_fits(
    func_name_list=["figures_2d", "figures_2d"],
    figure_name_list=["data", "noise_map"],
    filename="data_and_noise_map",
    tag_list=["DATA", "NOISE_MAP"],
    remove_fits_first=True,
)

"""
__Wrap Up__

In the simple example above, we used a `MultiFigurePlotter` to plot the same figure from each `ImagingPlotter` on
the same `matplotlib` subplot. 

This can be used for any figure plotted by any `Plotter` object, as long as the figure is plotted using the same
function name and figure name.
"""
