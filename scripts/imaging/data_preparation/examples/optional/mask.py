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

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from autoconf import jax_wrapper  # Sets JAX environment before other imports

