"""
Data Preparation: Lens Light Centre (Optional)
==============================================

This script marks the (y,x) arcsecond locations of the lens galaxy light centre(s) of the strong lens you are
analysing. These can be used as fixed values for the lens light and mass models in a model-fit.

This reduces the number of free parameters fitted for in a lens model and removes inaccurate solutions where
the lens mass model centre is unrealistically far from its true centre.

Advanced `chaining` scripts often use these input centres in the early fits to infer an accurate initial lens model,
amd then make the centres free parameters in later searches to ensure a general and accurate lens model is inferred.

If you create a `light_centre` for your dataset, you must also update your modeling script to use them.

If your **PyAutoLens** analysis is struggling to converge to a good lens model, you should consider using a fixed
lens light and / or mass centre to help the non-linear search find a good lens model.

Links / Resources:

The script `data_preparation/gui/lens_light_centre.ipynb` shows how to use a Graphical User Interface (GUI) to mask the
lens galaxy light centres.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
The path where the lens light centre is output, which is `dataset/imaging/simple`.
"""
dataset_type = "imaging"
dataset_name = "simple"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` dataset, so that the lens light centres can be plotted over the strong lens image.
"""
data = al.Array2D.from_fits(
    file_path=dataset_path / "data.fits", pixel_scales=pixel_scales
)

"""
Now, create a lens light centre, which is a `Grid2DIrregular` object of (y,x) values.
"""
light_centre = al.Grid2DIrregular(values=[(0.0, 0.0)])

"""
Now lets plot the image and lens light centre, so we can check that the centre overlaps the lens light.
"""

aplt.plot_array(array=data, title="")

"""
Now we`re happy with the lens light centre(s), lets output them to the dataset folder of the lens, so that we can
load them from a .json file in our pipelines!
"""
al.output_to_json(
    obj=light_centre,
    file_path=Path(dataset_path, "lens_light_centre.json"),
)

"""
The workspace also includes a GUI for drawing lens light centres, which can be found at
`autolens_workspace/*/imaging/data_preparation/gui/light_centres.py`.

This tools allows you `click` on the image where the lens light centres are, and it uses the brightest
pixel within a 5x5 box of pixels to select the coordinate.
"""
