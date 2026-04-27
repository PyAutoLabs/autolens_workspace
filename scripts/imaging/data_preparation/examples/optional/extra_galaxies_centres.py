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

__Contents__

**Masking Extra Galaxies:** The example `mask_extra_galaxies.py` masks the regions of an image where extra galaxies are present.
**Output:** Save this as a .png image in the dataset folder for easy inspection later.

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

The script `features/extra_galaxies/modeling` shows how to use extra galaxies in a model-fit, including loading the
extra galaxy centres created by this script.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
The path where the extra galaxy centres are output, which is `dataset/imaging/extra_galaxies`.
"""
dataset_type = "imaging"
dataset_name = "extra_galaxies"
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
        [sys.executable, "scripts/imaging/features/extra_galaxies/simulator.py"],
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
Create the extra galaxy centres, which is a `Grid2DIrregular` object of (y,x) values.
"""
extra_galaxies_centres = al.Grid2DIrregular(values=[(1.0, 3.5), (-2.0, -3.5)])

"""
Plot the image and extra galaxy centres, so we can check that the centre overlaps the lens light.
"""

aplt.plot_array(array=data, title="")

"""
__Output__

Save this as a .png image in the dataset folder for easy inspection later.
"""

aplt.plot_array(array=data, title="")

"""
Output the extra galaxy centres to the dataset folder of the lens, so that we can load them from a .json file
when we model them.
"""
al.output_to_json(
    obj=extra_galaxies_centres,
    file_path=Path(dataset_path, "extra_galaxies_centres.json"),
)

"""
The workspace also includes a GUI for drawing extra galaxy centres, which can be found at
`autolens_workspace/*/imaging/data_preparation/gui/extra_galaxies_centres.py`.

This tools allows you `click` on the image where an image of the lensed source is, and it will use the brightest pixel
within a 5x5 box of pixels to select the coordinate.
"""
