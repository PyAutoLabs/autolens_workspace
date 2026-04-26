"""
Data Preparation: Positions (Optional)
======================================

The script manually marks the (y,x) arc-second positions of the multiply imaged lensed source galaxy in the image-plane,
under the assumption that they originate from the same location in the source-plane.

A non-linear search (e.g. Nautilus) can then use these positions to preferentially choose mass models where these
positions trace close to one another in the source-plane. This speeding up the initial fitting of lens models and
removes unwanted solutions from parameter space which have too much or too little mass in the lens galaxy.

If you create positions for your dataset, you must also update your modeling script to use them by loading them
and passing them to the `Analysis` object via a `PositionsLH` object.

If your **PyAutoLens** analysis is struggling to converge to a good lens model, you should consider using positions
to help the non-linear search find a good lens model.

Links / Resources:

Position-based lens model resampling is particularly important for fitting pixelized source models, for the
reasons disucssed in the following readthedocs
webapge  https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html

The script `data_preparation/gui/positions.ipynb` shows how to use a Graphical User Interface (GUI) to mask the
positions on the lensed source.

See `autolens_workspace/*/guides/modeling/customize` for an example.of how to use positions in a
`modeling` script.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
The path where positions are output, which is `dataset/imaging/simple`
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
Load the `Imaging` dataset, so that the positions can be plotted over the strong lens image.
"""
data = al.Array2D.from_fits(
    file_path=dataset_path / "data.fits", pixel_scales=pixel_scales
)

"""
Now, create a set of positions, which is a Coordinate of (y,x) values.
"""
positions = al.Grid2DIrregular(
    values=[(0.4, 1.6), (1.58, -0.35), (-0.43, -1.59), (-1.45, 0.2)]
)

"""
Now lets plot the image and positions, so we can check that the positions overlap different regions of the source.
"""

aplt.plot_array(array=data, title="")

"""
Now we`re happy with the positions, lets output them to the dataset folder of the lens, so that we can load them from a
.json file in our pipelines!
"""
al.output_to_json(
    obj=positions,
    file_path=Path(dataset_path, "positions.json"),
)

"""
Finished.
"""
