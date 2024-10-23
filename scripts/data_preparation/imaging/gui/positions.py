"""
GUI Preprocessing: Positions
============================

This tool allows one to input the positions of strong lenses via a GUI, which can be used to resample inaccurate
mass models during lensing modeling.

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
from matplotlib import pyplot as plt
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
Load the image which we will use to mark the positions.
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
the positions.
"""
clicker = al.Clicker(
    image=data, pixel_scales=pixel_scales, search_box_size=search_box_size
)


"""
For lenses with bright lens light emission, it can be difficult to get the source light to show. The normalization
below uses a log-scale with a capped maximum, which better contrasts the lens and source emission.
"""
cmap = aplt.Cmap(norm="linear", vmin=1.0e-4, vmax=np.max(data))

norm = cmap.norm_from(array=None)

"""
Set up the clicker canvas and load the GUI which you can now click on to mark the positionss.
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
Use the results of the Clicker GUI to create the list of the positions.
"""
positions = al.Grid2DIrregular(values=clicker.click_list)

"""
__Output__

Now lets plot the image and positions,, so we can check that the positions overlap the brightest pixels in the
lensed source.
"""
visuals = aplt.Visuals2D(mass_profile_centres=positions)

array_2d_plotter = aplt.Array2DPlotter(
    array=data, visuals_2d=visuals, mat_plot_2d=aplt.MatPlot2D()
)
array_2d_plotter.figure_2d()

"""
Output this image of the positions to a .png file in the dataset folder for future reference.
"""
array_2d_plotter = aplt.Array2DPlotter(
    array=data,
    visuals_2d=visuals,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, filename="positions", format="png")
    ),
)
array_2d_plotter.figure_2d()

"""
Output the positions to a .json file in the dataset folder, so we can load them in modeling scripts.
"""
al.output_to_json(
    obj=positions,
    file_path=path.join(dataset_path, "positions.json"),
)
