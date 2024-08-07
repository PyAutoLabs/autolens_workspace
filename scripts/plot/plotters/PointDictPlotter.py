"""
Plots: PointDatasetPlotter
==========================

This example illustrates how to plot a `PointDict` dataset using an `PointDictPlotter`.

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

First, lets load an example strong lens `PointDict` object.

This dataset contains two `PointDatasets`, corresponding to a double Einstein cross system.
"""
dataset_name = "double_einstein_cross"
dataset_path = path.join("dataset", "point_source", dataset_name)

dataset = al.PointDict.from_json(file_path=path.join(dataset_path, "dataset.json"))

"""
__Subplots__

We now pass the point dict to a `PointDictPlotter`.
"""
dataset_plotter = aplt.PointDictPlotter(dataset=dataset)

"""
We first call the `subplot_positions` method to plot the multiple image positions of every point dataset in the
point dict. 
"""
dataset_plotter.subplot_positions()

"""
The `subplot_fluxes` method plots the flux of every multiple image of every point dataset in the point dict. 
"""
dataset_plotter.subplot_fluxes()

"""
Finish.
"""
