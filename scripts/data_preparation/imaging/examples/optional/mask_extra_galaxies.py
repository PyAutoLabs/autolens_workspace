"""
Data Preparation: Extra Galaxies Mask (Optional)
================================================

There may be regions of an image that have signal near the lens and source that is from other galaxies not associated 
with the strong lens we are studying. The emission from these images will impact our model fitting and needs to be 
removed from the analysis.

This script creates a mask of these regions of the image, called the `mask_extra_galaxies`, which can be used to
prevent them from impacting a fit. This mask may also include emission from objects which are not technically galaxies,
but blend with the galaxy we are studying in a similar way. Common examples of such objects are foreground stars
or emission due to the data reduction process.

The mask can be applied in different ways. For example, it could be applied such that the image pixels are discarded
from the fit entirely, Alternatively the mask could be used to set the image values to (near) zero and increase their
corresponding noise-map to large values.

The exact method used depends on the nature of the model being fitted. For simple fits like a light profile a mask
is appropriate, as removing image pixels does not change how the model is fitted. However, for more complex models
fits, like those using a pixelization, masking regions of the image in a way that removes their image pixels entirely
from the fit can produce discontinuities in the pixelixation. In this case, scaling the data and noise-map values
may be a better approach.

This script outputs a `mask_extra_galaxies.fits` file, which can be loaded and used before a model fit, in whatever
way is appropriate for the model being fitted.

__Links / Resources__

The script `data_preparation/gui/extra_galaxies_mask.ipynb` shows how to use a Graphical User Interface (GUI) to create
the extra galaxies mask.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

# %matplotlib inline
from os import path
import autolens as al
import autolens.plot as aplt

import numpy as np

"""
The path where the extra galaxy centres are output, which is `dataset/imaging/extra_galaxies`.
"""
dataset_type = "imaging"
dataset_name = "extra_galaxies"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the dataset image, so that the location of galaxies is clear when scaling the noise-map.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()

"""
Manually define the extra galaxies mask corresponding to the regions of the image where extra galaxies are located
whose emission needs to be omitted from the model-fit.
"""
mask = al.Mask2D.all_false(
    shape_native=data.shape_native, pixel_scales=data.pixel_scales
)
mask[100:140, 45:82] = True
mask[70:100, 125:150] = True

"""
Apply the extra galaxies mask to the image, which will remove them from visualization.
"""
data = data.apply_mask(mask=mask)

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

"""
The workspace also includes a GUI for image and noise-map scaling, which can be found at 
`autolens_workspace/*/data_preparation/imaging/gui/mask_extra_galaxies.py`. 

This tools allows you `spray paint` on the image where an you want to scale, allow irregular patterns (i.e. not 
rectangles) to be scaled.
"""
