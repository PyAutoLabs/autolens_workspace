"""
GUI Preprocessing: Mask
=======================

This tool allows one to mask a bespoke mask for a given image of a strong lens using an interactive GUI. This mask
can then be loaded before a pipeline is run and passed to that pipeline so as to become the default masked used by a
search (if a mask function is not passed to that search).

This GUI is adapted from the following code: https://gist.github.com/brikeats/4f63f867fd8ea0f196c78e9b835150ab
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

Setup the path the datasets we'll use to illustrate preprocessing, which is the 
folder `dataset/imaging/simple__no_lens_light`.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` dataset, so that the mask can be plotted over the strong lens image.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

"""
__Scribbler__

Load the Scribbler GUI for drawing the mask. 

Push Esc when you are finished drawing the mask.
"""
scribbler = al.Scribbler(image=data.native)
mask = scribbler.show_mask()
mask = al.Mask2D(mask=np.invert(mask), pixel_scales=pixel_scales)

"""
__Output__

Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
"""
visuals = aplt.Visuals2D(mask=mask)
array_2d_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_2d_plotter.figure_2d()

"""
Output this image of the mask to a .png file in the dataset folder for future reference.
"""
array_2d_plotter = aplt.Array2DPlotter(
    array=data,
    visuals_2d=visuals,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, filename="mask_gui", format="png")
    ),
)
array_2d_plotter.figure_2d()

"""
Output it to the dataset folder of the lens, so that we can load it from a .fits in our modeling scripts.
"""
mask.output_to_fits(file_path=path.join(dataset_path, "mask_gui.fits"), overwrite=True)
