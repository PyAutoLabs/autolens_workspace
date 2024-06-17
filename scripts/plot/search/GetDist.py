"""
Plots: GetDist
==============

This example illustrates how to plot visualization summarizing the results of model-fit using any non-linear search
using GetDist:

 - https://getdist.readthedocs.io/en/latest/

GetDist is an optional library which creates 1D and 2D plots of probability distribution functions (PDF)s. Its
visualization tools has more than the in-built visualization tools of many non-linear searches (e.g. Nautilus /
emcee) and can often produce better looking plots.

GetDist was developed for the analysis of Cosmological datasets.

Installation
------------

Because GetDist is an optional library, you will likely have to install it manually via the command:

`pip install getdist`
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")


import numpy as np
import matplotlib.pyplot as plt
from os import path

from getdist import MCSamples
from getdist import plots

import autofit as af
import autolens as al

"""
__Model Fit__

First, lets create a result via Nautilus by repeating the simple model-fit that is performed in 
the `modeling/start_here.py` example.

We'll use Nautilus in this example, but any MCMC / nested sampling non-linear search which produces samples of
the posterior could be used.
"""
dataset_name = "simple__no_lens_light"

search = af.Nautilus(
    path_prefix=path.join("plot"), name="GetDist", unique_tag=dataset_name, n_live=100
)

dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

samples = result.samples

"""
__Param Names__

GetDist uses a `model.paramnames` file to load the name of every parameter in the model-fit and pair it with the
latex symbol used to represent it in plots.

This file is not created by **PyAutoLens** by default, but can be output by the `search.paths` object as shown below.
"""
search.paths._save_parameter_names_file(model=model)
search.paths.zip_remove()
search.paths._zip()

"""
__GetDist MCSamples__

GetDist uses an `MCSamples` object to store the samples of a non-linear search.

Below, we create this object via a conversion from **PyAutoFit** `Samples`, as well as using the `names`
and `labels` of parameters in the `Samples` object.

The input `sampler="nested"` is input because we used a nested sampling, `Nautilus`. For MCMC this should be
replaced with "mcmc".
"""
gd_samples = MCSamples(
    samples=np.asarray(samples.parameter_lists),
    loglikes=np.asarray(samples.log_likelihood_list),
    weights=np.asarray(samples.weight_list),
    names=samples.model.model_component_and_parameter_names,
    labels=samples.model.parameter_labels_with_superscripts,
    sampler="nested",
)

"""
__Parameter Names__

Note that in order to customize the figure, we will use the `samples.model.parameter_names` list.
"""
print(samples.model.model_component_and_parameter_names)

"""
__GetDist Plotter__

To make plots we use a GetDist plotter object, which can be customized to change the appearance of the plots.
"""
gd_plotter = plots.get_subplot_plotter(width_inch=12)

"""
__GetDist Subplots__

Using the plotter we can make different plots, for example a triangle plot showing the 1D and 2D PDFs of every 
parameter.
"""
gd_plotter.triangle_plot(roots=gd_samples, filled=True)

plt.show()
plt.close()

"""
A triangle plot with specific parameters can be plotted by using the `params` input, whereby we specify the specific
parameter names to plot.
"""
gd_plotter.triangle_plot(
    roots=gd_samples,
    filled=True,
    params=[
        "galaxies_lens_mass_einstein_radius",
        "galaxies_lens_mass_ell_comps_0",
    ],
)

plt.show()
plt.close()

"""
Rectangle plots can be used to show specific 2D combinations of parameters.
"""
gd_plotter.rectangle_plot(
    roots=gd_samples,
    yparams=["galaxies_lens_mass_einstein_radius"],
    xparams=[
        "galaxies_lens_mass_ell_comps_0",
        "galaxies_lens_mass_ell_comps_1",
    ],
)

plt.show()
plt.close()

"""
__GetDist Single Plots__

We can make plots of specific 1D or 2D PDFs, using the single plotter object.
"""
gd_plotter = plots.get_single_plotter()

gd_plotter.plot_1d(roots=gd_samples, param="galaxies_lens_mass_einstein_radius")

plt.show()
plt.close()

gd_plotter = plots.get_single_plotter()

gd_plotter.plot_2d(
    roots=gd_samples,
    param1="galaxies_lens_mass_einstein_radius",
    param2="galaxies_lens_mass_ell_comps_0",
)

plt.show()
plt.close()

"""
We can also make a 3D plot, where the 2D PDF is plotted colored by the value of a third parameter.
"""
gd_plotter = plots.get_single_plotter()

gd_plotter.plot_3d(
    roots=gd_samples,
    params=[
        "galaxies_lens_mass_einstein_radius",
        "galaxies_lens_mass_ell_comps_0",
        "galaxies_lens_mass_ell_comps_1",
    ],
)

plt.show()
plt.close()

"""
__Output__

A figure can be output using standard matplotlib functionality.
"""

gd_plotter = plots.get_single_plotter()

gd_plotter.plot_3d(roots=gd_samples, params=["centre", "sigma", "normalization"])

output_path = path.join("output")

plt.savefig(path.join(output_path, "getdist.png"))
plt.close()

"""
__GetDist Other Plots__

There are many more ways to visualize PDFs possible with GetDist, checkout the official documentation for them all!

 - https://getdist.readthedocs.io/en/latest/
 - https://getdist.readthedocs.io/en/latest/plots.html

__Plotting Multiple Samples__

Finally, we can plot the results of multiple different non-linear searches on the same plot, using all
of the functions above.

Lets quickly make a second set of `Nautilus` results and plot them on the same figure above with the results
of the first search.
"""

Nautilus = af.Nautilus(path_prefix="plot", name="GetDist_2")

result_extra = Nautilus.fit(model=model, analysis=analysis)

samples_extra = result_extra.samples

gd_samples_extra = MCSamples(
    samples=np.asarray(samples_extra.parameter_lists),
    loglikes=np.asarray(samples_extra.log_likelihood_list),
    weights=np.asarray(samples_extra.weight_list),
    names=samples_extra.model.model_component_and_parameter_names,
    labels=samples.model.parameter_labels_with_superscripts,
    sampler="nested",
)

gd_plotter = plots.get_subplot_plotter(width_inch=12)

gd_plotter.triangle_plot(roots=[gd_samples, gd_samples_extra], filled=True)

plt.show()
plt.close()

"""
Note that the models do not need to be the same to make the plots above.

GetDist will clever use the `names` of the parameters to combine the parameters into customizeable PDF plots.
"""
