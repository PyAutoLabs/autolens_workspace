"""
Plots: PointDatasetPlotter
==========================

This example illustrates how to plot a `PointDataset` dataset using an `PointDatasetPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load an example strong lens `PointDataset` object.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "point_source", dataset_name)

dataset = al.from_json(
    file_path=path.join(dataset_path, "point_dataset.json"),
)

"""
__Figures__

We now pass the point dataset to a `PointDatasetPlotter` and call various `figure_*` methods to plot different 
attributes.
"""
point_dataset_plotter = aplt.PointDatasetPlotter(dataset=dataset)
# point_dataset_plotter.figures_2d(positions=True, fluxes=True)

"""
__Subplots__

The `PointDatasetPlotter` can also plot a subplot of all of these attributes.
"""
point_dataset_plotter.subplot_dataset()

"""
Finish.
"""
