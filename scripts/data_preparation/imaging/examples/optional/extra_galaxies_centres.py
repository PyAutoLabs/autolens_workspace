"""
Data Preparation: Extra Galaxies (Optional)
===========================================

There may be extra galaxies nearby the lens and source galaxies, whose emission blends with the lens and source
and whose mass may contribute to the ray-tracing and lens model.

We can include these extra galaxies in the lens model, either as light profiles, mass profiles, or both, using the
modeling API, where these nearby objects are denoted `extra_galaxies`.

This script marks the (y,x) arcsecond locations of these extra galaxies, so that when they are included in the lens model 
the centre of these extra galaxies light and / or mass profiles are fixed to these values (or their priors are initialized
surrounding these centres).

This tutorial closely mirrors tutorial 7, `lens_light_centre`, where the main purpose of this script is to mark the
centres of every object we'll model as an extra galaxy. A GUI is also available to do this.

__Masking Extra Galaxies__

The example `mask_extra_galaxies.py` masks the regions of an image where extra galaxies are present. This mask is used 
to remove their signal from the data and increase their noise to make them not impact the fit. This means their 
luminous emission does not need to be included in the model, reducing the number of free parameters and speeding up the 
analysis. It is still a choice whether their mass is included in the model.

Which approach you use to account for the emission of extra galaxies, modeling or masking, depends on how significant 
the blending of their emission with the lens and source galaxies is and how much it impacts the model-fit.

__Links / Resources__

The script `data_preparation/gui/extra_galaxies_centres.ipynb` shows how to use a Graphical User Interface (GUI) to mark 
the extra galaxy centres in this way.

The script `modeling/features/extra_galaxies.py` shows how to use extra galaxies in a model-fit, including loading the 
extra galaxy centres created by this script.

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
Load the `Imaging` dataset, so that the lens light centres can be plotted over the strong lens image.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

"""
Create the extra galaxy centres, which is a `Grid2DIrregular` object of (y,x) values.
"""
extra_galaxies_centres = al.Grid2DIrregular(values=[(1.0, 3.5), (-2.0, -3.5)])

"""
Plot the image and extra galaxy centres, so we can check that the centre overlaps the lens light.
"""
mat_plot = aplt.MatPlot2D()
visuals = aplt.Visuals2D(light_profile_centres=extra_galaxies_centres)

array_plotter = aplt.Array2DPlotter(
    array=data, visuals_2d=visuals, mat_plot_2d=mat_plot
)
array_plotter.figure_2d()

"""
__Output__

Save this as a .png image in the dataset folder for easy inspection later.
"""
mat_plot = aplt.MatPlot2D(
    output=aplt.Output(
        path=dataset_path, filename="data_with_extra_galaxies", format="png"
    )
)
visuals = aplt.Visuals2D(light_profile_centres=extra_galaxies_centres)

array_plotter = aplt.Array2DPlotter(
    array=data, visuals_2d=visuals, mat_plot_2d=mat_plot
)
array_plotter.figure_2d()

"""
Output the extra galaxy centres to the dataset folder of the lens, so that we can load them from a .json file 
when we model them.
"""
al.output_to_json(
    obj=extra_galaxies_centres,
    file_path=path.join(dataset_path, "extra_galaxies_centres.json"),
)

"""
The workspace also includes a GUI for drawing extra galaxy centres, which can be found at 
`autolens_workspace/*/data_preparation/imaging/gui/extra_galaxies_centres.py`. 

This tools allows you `click` on the image where an image of the lensed source is, and it will use the brightest pixel 
within a 5x5 box of pixels to select the coordinate.
"""
