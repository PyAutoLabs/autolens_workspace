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
__Wrap Up__

In the simple example above, we used a `MultiFigurePlotter` to plot the same figure from each `ImagingPlotter` on
the same `matplotlib` subplot. 

This can be used for any figure plotted by any `Plotter` object, as long as the figure is plotted using the same
function name and figure name.
"""
