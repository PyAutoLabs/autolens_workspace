"""
GUI Preprocessing: Extra Galaxies Centres
=========================================

There may be extra galaxies nearby the lens and source galaxies, whose emission blends with the lens and source
and whose mass may contribute to the ray-tracing and lens model.

The example `data_preparation/imaging/example/optional/extra_galaxies_centres.py` provides a full description of
what the extra galaxies are and how they are used in the model-fit. You should read this script first before
using this script.

This script uses a GUI to mark the (y,x) arcsecond locations of these extra galaxies, in contrast to the example
above which requires you to input these values manually.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt
from matplotlib import pyplot as plt

"""
__Dataset__

The path where the extra galaxy centres are output, which is `dataset/imaging/extra_galaxies`.
"""
dataset_name = "extra_galaxies"
dataset_path = path.join("dataset", "imaging", dataset_name)

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the image which we will use to mark the lens light centre.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

"""
__Search Box__

When you click on a pixel to mark a position, the search box looks around this click and finds the pixel with
the highest flux to mark the position.

The `search_box_size` is the number of pixels around your click this search takes place.
"""
search_box_size = 5

"""
__Clicker__

Set up the `Clicker` object from the `clicker.py` module, which monitors your mouse clicks in order to determine
the extra galaxy centres.
"""
clicker = al.Clicker(
    image=data, pixel_scales=pixel_scales, search_box_size=search_box_size
)

"""
Set up the clicker canvas and load the GUI which you can now click on to mark the extra galaxy centres.
"""
n_y, n_x = data.shape_native
hw = int(n_x / 2) * pixel_scales
ext = [-hw, hw, -hw, hw]
fig = plt.figure(figsize=(14, 14))
plt.imshow(data.native, cmap="jet", extent=ext)
plt.colorbar()
cid = fig.canvas.mpl_connect("button_press_event", clicker.onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)
plt.close(fig)

"""
Use the results of the Clicker GUI to create the list of extra galaxy centres.
"""
extra_galaxies_centres = al.Grid2DIrregular(values=clicker.click_list)

"""
__Output__

Now lets plot the image and extra galaxy centres, so we can check that the centre overlaps the brightest pixels in the
extra galaxies.
"""
visuals = aplt.Visuals2D(mass_profile_centres=extra_galaxies_centres)

array_2d_plotter = aplt.Array2DPlotter(
    array=data, visuals_2d=visuals, mat_plot_2d=aplt.MatPlot2D()
)
array_2d_plotter.figure_2d()

"""
Output this image of the extra galaxy centres to a .png file in the dataset folder for future reference.
"""
array_2d_plotter = aplt.Array2DPlotter(
    array=data,
    visuals_2d=visuals,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=dataset_path, filename="extra_galaxies_centres", format="png"
        )
    ),
)
array_2d_plotter.figure_2d()

"""
Output the extra galaxy centres to the dataset folder of the lens, so that we can load them from a .json file 
when we model them.
"""
al.output_to_json(
    obj=extra_galaxies_centres,
    file_path=path.join(dataset_path, "extra_galaxies_centres.json"),
)
