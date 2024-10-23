"""
Data Preparation: Mask (Optional)
=================================

The mask removes the regions of the image where the lens and source galaxy are not present, typically the edges of the
image.

Example modeling scripts internally create a 3.0" circular mask and therefore do not require that a mask has been
created externally via a data preparation script.

This script shows how to create customize masked (e.g. annular, ellipses) which are tailored to match the lens or
lensed source emission.

If you have not analysed your dataset yet and do not know of a specific reason why you need the bespoke masks
created by this script, it is recommended that you simply use the default ~3.0" circular mask internally made in each
script and omit this data preparation tutorial.

Links / Resources:

The `examples/mask.ipynb` scripts shows how to create customize masked (e.g. annular, ellipses)
which are tailored to match the lens or lensed source emission of your data.

The script `data_preparation/gui/mask.ipynb` shows how to use a Graphical User Interface (GUI) to create an even
more custom mask.

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
This tool allows one to mask a bespoke mask for a given image of a strong lens, which is loaded before a
pipeline is run and passed to that pipeline.

Whereas in the previous 3 tutorials we used the data_raw folder of `autolens/propocess`, the mask is generated from
the reduced dataset, so we'll example `Imaging` in the `autolens_workspace/dataset` folder where your dataset reduced
following `data_preparation` tutorials 1-3 should be located.

Setup the path to the autolens_workspace, using the correct path name below.

The `dataset label` is the name of the folder in the `autolens_workspace/dataset` folder and `dataset_name` the 
folder the dataset is stored in, e.g, `/autolens_workspace/dataset/dataset_type/dataset_name`. The mask will be 
output here as `mask.fits`.
"""
dataset_type = "imaging"
dataset_name = "simple__no_lens_light"

"""
Returns the path where the mask will be output, which in this case is
`/autolens_workspace/dataset/imaging/simple__no_lens_light`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
First, load the image of the dataset, so that the mask can be plotted over the strong lens.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

"""
Create a mask for this dataset, using the Mask2D object I`ll use a circular-annular mask here, but I`ve commented 
other options you might want to use (feel free to experiment!)
"""
mask = al.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.5,
    outer_radius=2.5,
    centre=(0.0, 0.0),
)

# mask = al.Mask2D.circular(
#     shape_native=data.shape_native,
#     pixel_scales=data.pixel_scales,
#     radius=2.5,
#     centre=(0.0, 0.0),
# )

# mask = al.Mask2D.elliptical(
#     shape_native=data.shape_native,
#     pixel_scales=data.pixel_scales,
#     major_axis_radius=2.5,
#     axis_ratio=0.7,
#     angle=45.0,
#     centre=(0.0, 0.0),
# )

# mask = al.Mask2D.elliptical_annular(
#     shape_native=data.shape_native,
#     pixel_scales=data.pixel_scales,
#     inner_major_axis_radius=0.5,
#     inner_axis_ratio=0.7,
#     inner_phi=45.0,
#     outer_major_axis_radius=0.5,
#     outer_axis_ratio=0.7,
#     outer_phi=45.0,
#     centre=(0.0, 0.0),
# )

"""
Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
"""
visuals = aplt.Visuals2D(mask=mask)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
Output the masked image to clearly show what parts of the source are included.
"""
data = data.apply_mask(mask=mask)

mat_plot = aplt.MatPlot2D(
    output=aplt.Output(path=dataset_path, filename=f"data_masked", format="png")
)
array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Now we`re happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""
mask.output_to_fits(file_path=path.join(dataset_path, "mask.fits"), overwrite=True)

"""
The workspace also includes a GUI for drawing a mask, which can be found at 
`autolens_workspace/*/data_preparation/imaging/gui/mask.py`. This tools allows you to draw the mask via a `spray paint` mouse
icon, such that you can draw irregular masks more tailored to the source's light.
"""
