"""
Data Preparation: Clumps (Optional)
===================================

There may be galaxies nearby the lens and source galaxies, whose emission blends with that of the lens and source
and whose mass may contribute to the ray-tracing and lens model.

We can include these galaxies in the lens model, either as light profiles, mass profiles, or both, using the
**PyAutoLens** clump API, where these nearby objects are given the term `clumps`.

This script marks the (y,x) arcsecond locations of these clumps, so that when they are included in the lens model the
centre of these clumps light and / or mass profiles are fixed to these values (or their priors are initialized
surrounding these centres).

The example `scaled_dataset.py` marks the regions of an image where clumps are present, but  but instead remove their
signal and increase their noise to make them not impact the fit. Which approach you use to account for clumps depends
on how significant the blending of their emission is and whether they are expected to impact the ray-tracing.

This tutorial closely mirrors tutorial 7, `lens_light_centre`, where the main purpose of this script is to mark the
centres of every object we'll model as a clump. A GUI is also available to do this.

Links / Resources:

The script `data_prepration/gui/clump_centres.ipynb` shows how to use a Graphical User Interface (GUI) to mark the
clump centres in this way.

The script `modeling/features/clumps.py` shows how to use clumps in a model-fit, including loading the clump centres
created by this script.

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

"""
The path where the clump centre is output, which is `dataset/imaging/clumps`.
"""
dataset_type = "imaging"
dataset_name = "clumps"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
"""
pixel_scales = 0.1

"""
First, load the `Imaging` dataset, so that the lens light centres can be plotted over the strong lens image.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

"""
Now, create the clump centres, which is a Grid2DIrregular object of (y,x) values.
"""
clump_centres = al.Grid2DIrregular(values=[(1.0, 3.5), (-2.0, -3.5)])

"""
Now lets plot the image and clump centres, so we can check that the centre overlaps the lens light.
"""
mat_plot = aplt.MatPlot2D()
visuals = aplt.Visuals2D(light_profile_centres=clump_centres)

array_plotter = aplt.Array2DPlotter(
    array=data, visuals_2d=visuals, mat_plot_2d=mat_plot
)
array_plotter.figure_2d()

"""
Now we`re happy with the clump centre(s), lets output them to the dataset folder of the lens, so that we can load them 
from a .json file in our pipelines!
"""
clump_centres.output_to_json(
    file_path=path.join(dataset_path, "clump_centres.json"), overwrite=True
)

"""
The workspace also includes a GUI for drawing clump centres, which can be found at 
`autolens_workspace/*/data_preparation/imaging/gui/clump_centres.py`. 

This tools allows you `click` on the image where an image of the lensed source is, and it will use the brightest pixel 
within a 5x5 box of pixels to select the coordinate.
"""
