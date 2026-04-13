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

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

