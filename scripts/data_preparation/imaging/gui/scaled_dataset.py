"""
GUI Preprocessing: Scaled Dataset
=================================

This tool allows one to mask a bespoke noise-map for a given image of a strong lens, using a GUI.

This noise-map is primarily used for increasing the variances of pixels that have non-modeled components in an image,
for example intervening line-of-sight galaxies that are near the lens, but not directly interfering with the
analysis of the lens and source galaxies.

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
folder `dataset/imaging/clumps`.
"""
dataset_name = "clumps"
dataset_path = path.join("dataset", "imaging", dataset_name)

"""
If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
"""
pixel_scales = 0.1

"""
First, load the `Imaging` dataset, so that the location of galaxies is clear when scaling the noise-map.
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
Change the image flux values to zeros. 

If included, we add some random Gaussian noise to most close resemble noise in the image.
"""
background_level = al.preprocess.background_noise_map_via_edges_from(
    image=data, no_edges=2
)[0]

# gaussian_sigma = None
gaussian_sigma = 0.1

data = np.where(mask, 0.0, data.native)
data = al.Array2D.no_mask(values=data, pixel_scales=pixel_scales)

if gaussian_sigma is not None:
    random_noise = np.random.normal(
        loc=background_level, scale=gaussian_sigma, size=data.shape_native
    )
    data = np.where(mask, random_noise, data.native)
    data = al.Array2D.no_mask(values=data, pixel_scales=pixel_scales)

"""
__Output__

The new image is plotted for inspection.
"""
array_2d_plotter = aplt.Array2DPlotter(array=data)
array_2d_plotter.figure_2d()

"""
Output this image of the mask to a .png file in the dataset folder for future reference.
"""
array_2d_plotter = aplt.Array2DPlotter(
    array=data,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, filename="data_scaled", format="png")
    ),
)
array_2d_plotter.figure_2d()

"""
Output image to the dataset folder of the lens, so that we can load it from a .fits file for modeling.
"""
data.output_to_fits(
    file_path=path.join(dataset_path, "data_scaled.fits"), overwrite=True
)

"""
__Noise Map__

Next, load the `Imaging` noise-map, which we will use the scale the noise-map.
"""
noise_map = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "noise_map.fits"), pixel_scales=pixel_scales
)

"""
Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
"""
array_2d_plotter = aplt.Array2DPlotter(array=data / noise_map)
array_2d_plotter.figure_2d()

"""
Manually increase the noise values to extremely large values, such that the analysis essentially omits them.
"""
noise_map = np.where(mask, 1.0e8, noise_map.native)
noise_map = al.Array2D.no_mask(values=noise_map, pixel_scales=pixel_scales)

"""
The signal to noise-map is the best way to determine if these regions are appropriately masked out.
"""
array_2d_plotter = aplt.Array2DPlotter(array=data / noise_map)
array_2d_plotter.figure_2d()

"""
__Output__

Output it to the dataset folder of the lens, so that we can load it from a .fits in our modeling scripts.
"""
noise_map.output_to_fits(
    file_path=path.join(dataset_path, "noise_map_scaled.fits"), overwrite=True
)

"""
Output this image of the mask to a .png file in the dataset folder for future reference.
"""
array_2d_plotter = aplt.Array2DPlotter(
    array=noise_map,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, filename="noise_map_scaled", format="png")
    ),
)
array_2d_plotter.figure_2d()

array_2d_plotter = aplt.Array2DPlotter(
    array=data / noise_map,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=dataset_path, filename="signal_to_noise_map_scaled", format="png"
        )
    ),
)
array_2d_plotter.figure_2d()

"""
Lets also output the mask.
"""
mask.output_to_fits(
    file_path=path.join(dataset_path, "mask_scaled.fits"), overwrite=True
)
