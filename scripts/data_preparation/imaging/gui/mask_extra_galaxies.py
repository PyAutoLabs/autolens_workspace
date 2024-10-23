"""
GUI Preprocessing: Extra Galaxies Mask (Optional)
=================================================

There may be regions of an image that have signal near the lens and source that is from other galaxies not associated 
with the strong lenswe are studying. The emission from these images will impact our model fitting and needs to be 
removed from the analysis.

The example `data_preparation/imaging/example/optional/extra_galaxies_mask.py` provides a full description of
what the extra galaxies are and how they are used in the model-fit. You should read this script first before
using this script.

This script uses a GUI to mark the regions of the image where these extra galaxies are located, in contrast to the
example above which requires you to input these values manually.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
__Dataset__

The path where the extra galaxy mask is output, which is `dataset/imaging/extra_galaxies`.
"""
dataset_name = "extra_galaxies"
dataset_path = path.join("dataset", "imaging", dataset_name)

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` data, where the extra galaxies are visible in the data.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

cmap = aplt.Cmap(
    norm="log", vmin=1.0e-4, vmax=0.4 * np.max(data), linthresh=0.05, linscale=0.1
)

"""
__Scribbler__

Load the Scribbler GUI for spray painting the scaled regions of the dataset. 

Push Esc when you are finished spray painting.
"""
scribbler = al.Scribbler(image=data.native, cmap=cmap)
mask = scribbler.show_mask()
mask = al.Mask2D(mask=mask, pixel_scales=pixel_scales)

"""
The GUI has now closed and the extra galaxies mask has been created.

Apply the extra galaxies mask to the image, which will remove them from visualization.
"""
data = data.apply_mask(mask=mask)

"""
__Output__

The new image is plotted for inspection.
"""
array_2d_plotter = aplt.Array2DPlotter(array=data)
array_2d_plotter.figure_2d()

"""
Plot the data with the new mask, in order to check that the mask removes the regions of the image corresponding to the
extra galaxies.
"""
array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()

"""
__Output__

Output to a .png file for easy inspection.
"""
mat_plot = aplt.MatPlot2D(
    output=aplt.Output(
        path=dataset_path, filename=f"data_mask_extra_galaxies", format="png"
    )
)
array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Output the extra galaxies mask, which will be load and used before a model fit.
"""
mask.output_to_fits(
    file_path=path.join(dataset_path, "mask_extra_galaxies.fits"), overwrite=True
)
