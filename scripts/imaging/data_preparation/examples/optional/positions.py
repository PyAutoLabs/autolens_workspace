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

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from autoconf import jax_wrapper  # Sets JAX environment before other imports

