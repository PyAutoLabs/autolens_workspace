"""
Plots: MultiYX1DPlotter
=========================

This example illustrates how to plot multi 1D figure lines on the same plot.

It uses the specific example of plotting a `MassProfile`'s 1D convergence using multiple `MassProfilePlotter`'s.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
First, lets create two simple `MassProfile`'s which we'll plot the 1D convergences of on the same figure.
"""
mass_0 = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.0,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
)

mass_1 = al.mp.PowerLaw(
    centre=(0.0, 0.0),
    einstein_radius=1.0,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    slope=2.1,
)

"""
We also need the 2D grid the `MassProfile`'s are evaluated on.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)
