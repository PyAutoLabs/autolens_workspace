"""
Data Preparation: Info (Optional)
=================================

Auxiliary information about a strong lens dataset may used during an analysis or afterwards when interpreting the
 modeling results. For example, the redshifts of the source and lens galaxy.

By storing these as an `info.json` file in the lens's dataset folder, it is straight forward to load the redshifts
in a modeling script and pass them to a fit, such that **PyAutoLens** can then output results in physical
units (e.g. kpc instead of arc-seconds).

For analysing large quantities of  modeling results, **PyAutoLens** has an sqlite database feature. The info file
may can also be loaded by the database after a model-fit has completed, such that when one is interpreting
the results of a model fit additional data on a lens can be used to.

For example, to plot the model-results against other measurements of a lens not made by PyAutoLens. Examples of such
data might be:

- The velocity dispersion of the lens galaxy.
- The stellar mass of the lens galaxy.
- The results of previous strong lens models to the lens performed in previous papers.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path

from autoconf import jax_wrapper  # Sets JAX environment before other imports

