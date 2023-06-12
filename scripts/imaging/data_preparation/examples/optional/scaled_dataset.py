"""
Data Preparation: Scaled Dataset (Optional)
===========================================

There may be regions of an image that have signal near the lens and source that is from other sources (e.g. foreground
stars, background galaxies not associated with the strong lens). The emission from these images will impact our
model fitting and needs to be removed from the analysis.

This script marks these regions of the image and scales their image values to zero and increases their corresponding
noise-map to large values. This means that the model-fit will ignore these regions.

Why not just mask these regions instead? For fits using light profiles for the source (e.g. `Sersic`'s, shapelets
or a multi gaussian expansion) masking does not make a significant difference.

However, for fits using a `Pixelization` for the source, masking these regions can have a significant impact on the
reconstruction. Masking regions of the image removes them entirely from the fitting procedure. This means
their deflection angles are not computed, they are not traced to the source-plane and their corresponding
Delaunay / Voronoi cells do not form.

This means there are discontinuities in the source `Pixelization`'s mesh which can degrade the quality of the
reconstruction and negatively impact the `Regularization` scheme.

Therefore, by retaining them in the mask but scaling their values these source-mesh discontinuities are not
created and regularization still occurs over these regions of the source reconstruction.

Links / Resources:

The script `data_prepration/gui/scaled_data.ipynb` shows how to use a Graphical User Interface (GUI) to scale
the data in this way.
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
The path where the dataset we scale is loaded from, which 
is `dataset/imaging/clumps`
"""
dataset_type = "imaging"
dataset_name = "clumps"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
"""
pixel_scales = 0.1

"""
First, load the dataset image, so that the location of galaxies is clear when scaling the noise-map.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()

"""
Next, load the noise-map, which we will use the scale the noise-map.
"""
noise_map = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "noise_map.fits"), pixel_scales=pixel_scales
)

array_plotter = aplt.Array2DPlotter(array=noise_map)
array_plotter.figure_2d()

"""
Now lets plot the signal to noise-map, which will be reduced to nearly zero one we scale the noise.
"""
array_plotter = aplt.Array2DPlotter(array=data / noise_map)
array_plotter.figure_2d()

"""
First, we manually define a mask corresponding to the regions of the image we will scale.
"""
mask = al.Mask2D.all_false(
    shape_native=data.shape_native, pixel_scales=data.pixel_scales
)
mask[25:55, 77:96] = True
mask[55:85, 3:27] = True

"""
We are going to change the image flux values to low values. Not zeros, but values consistent with the background
signa in the image, which we can estimate from the image itself.
"""
background_level = al.preprocess.background_noise_map_via_edges_from(
    image=data, no_edges=2
)[0]

"""
This function uses the mask to scale the appropriate regions of the image to the background level.
"""
data = np.where(mask, background_level, data.native)
data = al.Array2D.no_mask(values=data, pixel_scales=pixel_scales)

"""
To make our scaled image look as realistic as possible, we can optionally included some noise drawn from a Gaussian
distribution to replicate the noise-pattern in the image. This requires us to choose a `gaussian_sigma` value 
representative of the data, which you should choose via `trial and error` until you get a noise pattern that is
visually hard to discern from the rest of the image.
"""
# gaussian_sigma = None
gaussian_sigma = 0.03

if gaussian_sigma is not None:
    random_noise = np.random.normal(
        loc=background_level, scale=gaussian_sigma, size=data.shape_native
    )
    data = np.where(mask, random_noise, data.native)
    data = al.Array2D.no_mask(values=data, pixel_scales=pixel_scales)

"""
The new image is plotted for inspection.
"""
array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()

"""
Now we`re happy with the image, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""
data.output_to_fits(
    file_path=path.join(dataset_path, "data_scaled.fits"), overwrite=True
)

"""
Here, we manually increase the noise values at these points in the mask to extremely large values, such that the 
analysis essentially omits them.
"""
noise_map = noise_map.native
noise_map[mask == True] = 1.0e8

"""
The noise-map and signal to noise-map show the noise-map being scaled in the correct regions of the image.
"""
array_plotter = aplt.Array2DPlotter(array=noise_map)
array_plotter.figure_2d()

array_plotter = aplt.Array2DPlotter(array=data / noise_map.slim)
array_plotter.figure_2d()

"""
Now we`re happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""
noise_map.output_to_fits(
    file_path=path.join(dataset_path, "noise_map_scaled.fits"), overwrite=True
)

"""
Finally, we can output the scaled mask encase we need it in the future.
"""
mask.output_to_fits(
    file_path=path.join(dataset_path, "mask_scaled.fits"), overwrite=True
)

"""
The workspace also includes a GUI for image and noise-map scaling, which can be found at 
`autolens_workspace/*/data_preparation/imaging/gui/scaled_dataset.py`. This tools allows you `spray paint` on the image where 
an you want to scale, allow irregular patterns (i.e. not rectangles) to be scaled.
"""
