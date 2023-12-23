"""
Plots: ImagingPlotter
=====================

This example illustrates how to plot an `Imaging` dataset using an `ImagingPlotter`.

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

"""
__Figures__

We now pass the imaging to an `ImagingPlotter` and call various `figure_*` methods to plot different attributes.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(
    data=True,
    noise_map=True,
    psf=True,
)

"""
__Subplots__

The `ImagingPlotter` may also plot a subplot of all of these attributes.
"""
dataset_plotter.subplot_dataset()

"""`
__Include__

Imaging` contains the following attributes which can be plotted automatically via the `Include2D` object.

(By default, an `Array2D` does not contain a `Mask2D`, we therefore manually created an `Array2D` with a mask to illustrate
the plotted of a mask and its border below).
"""
include = aplt.Include2D(origin=True, mask=True, border=True)

mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
dataset = dataset.apply_mask(mask=mask)
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, include_2d=include)
dataset_plotter.subplot_dataset()

"""
Finish.
"""
