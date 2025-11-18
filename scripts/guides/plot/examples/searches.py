"""
Plots: Searches
===============

This example illustrates the API for plotting the results of different non-linear searches.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how visuals work and the default
behaviour of plotting visuals.

__Contents__

- **Setup:** Sets up a dataset and model which we will perform quick model-fits to for illustration.
- **DynestyPlotter:**: Plots results of the nested sampling method Dynesty.
- **MCMCPlotter:**: Plots results of an Emcee fit (e.g. cornerplot).
- **PySwarmsPlotter:**: Plots results of a PySwarms fit (e.g. contour).
- **ZeusPlotter:**: Plots results of a Zeus fit (e.g. cornerplot).
- **GetDist:**: Plot results of any MCMC / nested sampler non-linear search using the library GetDist.

__Setup__

To illustrate plotting, we require standard objects like a dataset and model which we will perform quick model-fits to
for illustration.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_name = "simple__no_lens_light"

dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Notation__

Plot are labeled with short hand parameter names (e.g. the `centre` parameters are plotted using an `x`). 

The mappings of every parameter to its shorthand symbol for plots is specified in the `config/notation.yaml` file 
and can be customized.

Each label also has a superscript corresponding to the model component the parameter originates from. For example,
Gaussians are given the superscript `g`. This can also be customized in the `config/notation.yaml` file.

__DynestyPlotter__

We set up the Dynesty non-linear search and perform the fit to get the samples we will plot below.
"""
search = af.DynestyStatic(
    path_prefix=Path("plot"),
    name="DynestyPlotter",
    unique_tag=dataset_name,
    n_live=100,
)

result = search.fit(model=model, analysis=analysis)

"""
We now pass the samples to a `DynestyPlotter` which will allow us to use the corner plotting function of the
public library anesthetic

The Dynesty readthedocs describes fully all of the methods used below 

 - https://dynesty-sampler.readthedocs.io/en/latest/quickstart.html
 - https://dynesty-sampler.readthedocs.io/en/latest/api.html#module-Dynesty.plotting
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.

Dynesty plotters use `_kwargs` dictionaries to pass visualization settings to matplotlib lib. For example, below,
we:

 - Set the fontsize of the x and y labels by passing `label_kwargs={"fontsize": 16}`.
 - Set the fontsize of the title by passing `title_kwargs={"fontsize": "10"}`.
 
There are other `_kwargs` inputs we pass as None, you should check out the Dynesty docs if you need to customize your
figure.
"""
plotter = aplt.NestPlotter(samples=result.samples)

"""
The `corner_anesthetic` method produces a triangle of 1D and 2D PDF's of every parameter using the library `anesthetic`.
"""
plotter.corner_anesthetic()

"""
The `corner_cornerpy` method produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
plotter.corner_cornerpy(
    dims=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    color="black",
    smooth=0.02,
    quantiles_2d=None,
    hist_kwargs=None,
    hist2d_kwargs=None,
    label_kwargs={"fontsize": 16},
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": "10"},
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    verbose=False,
)


"""
The internal sampler can be used to plot the results of the non-linear search. 

We do this using the `search_internal` attribute which contains the sampler in its native form.

The first time you run a search, the `search_internal` attribute will be available because it is passed ot the
result via memory. 

If you rerun the fit on a completed result, it will not be available in memory, and therefore
will be loaded from the `files/search_internal` folder. The `search_internal` entry of the `output.yaml` must be true 
for this to be possible.
"""
search_internal = result.search_internal

"""
__Plots__

All plots use dynesty's inbuilt plotting library and the model.
"""
from dynesty import plotting as dyplot

model = result.model


"""
The boundplot plots the bounding distribution used to propose either (1) live points at a given iteration or (2) a 
specific dead point during the course of a run, projected onto the two dimensions specified by `dims`.
"""
dyplot.boundplot(
    results=search_internal.results,
    labels=model.parameter_labels_with_superscripts_latex,
    dims=(2, 2),
    it=-1,  # -1 is the final iteration of the dynesty samples, change this to plot a different iteration
    idx=None,
    prior_transform=None,
    periodic=None,
    reflective=None,
    ndraws=5000,
    color="gray",
    plot_kwargs=None,
    label_kwargs={"fontsize": 16},
    max_n_ticks=5,
    use_math_text=False,
    show_live=False,
    live_color="darkviolet",
    live_kwargs=None,
    span=None,
    fig=None,
)

plt.show()
plt.close()

"""
The cornerbound plots the bounding distribution used to propose either (1) live points at a given iteration or (2) a 
specific dead point during the course of a run, projected onto all pairs of dimensions.
"""
try:
    dyplot.cornerbound(
        results=search_internal.results,
        labels=model.parameter_labels_with_superscripts_latex,
        it=-1,  # -1 is the final iteration of the dynesty samples, change this to plot a different iteration
        idx=None,
        dims=None,
        prior_transform=None,
        periodic=None,
        reflective=None,
        ndraws=5000,
        color="gray",
        plot_kwargs=None,
        label_kwargs={"fontsize": 16},
        max_n_ticks=5,
        use_math_text=False,
        show_live=False,
        live_color="darkviolet",
        live_kwargs=None,
        span=None,
        fig=None,
    )

    plt.show()
    plt.close()

except ValueError:
    pass

"""
The cornerplot plots a corner plot of the 1-D and 2-D marginalized posteriors.
"""

try:
    dyplot.cornerplot(
        results=search_internal.results,
        labels=model.parameter_labels_with_superscripts_latex,
        dims=None,
        span=None,
        quantiles=[0.025, 0.5, 0.975],
        color="black",
        smooth=0.02,
        quantiles_2d=None,
        hist_kwargs=None,
        hist2d_kwargs=None,
        label_kwargs={"fontsize": 16},
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": "10"},
        truths=None,
        truth_color="red",
        truth_kwargs=None,
        max_n_ticks=5,
        top_ticks=False,
        use_math_text=False,
        verbose=False,
    )

    plt.show()
    plt.close()

except ValueError:
    pass


"""
The cornerpoints plots a (sub-)corner plot of (weighted) samples.
"""
dyplot.cornerpoints(
    results=search_internal.results,
    labels=model.parameter_labels_with_superscripts_latex,
    dims=None,
    thin=1,
    span=None,
    cmap="plasma",
    color=None,
    kde=True,
    nkde=1000,
    plot_kwargs=None,
    label_kwargs={"fontsize": 16},
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    max_n_ticks=5,
    use_math_text=False,
    fig=None,
)

plt.show()
plt.close()


"""
The runplot plots live points, ln(likelihood), ln(weight), and ln(evidence) as a function of ln(prior volume).
"""
dyplot.runplot(
    results=search_internal.results,
    span=None,
    logplot=False,
    kde=True,
    nkde=1000,
    color="blue",
    plot_kwargs=None,
    label_kwargs={"fontsize": 16},
    lnz_error=True,
    lnz_truth=None,
    truth_color="red",
    truth_kwargs=None,
    max_x_ticks=8,
    max_y_ticks=3,
    use_math_text=True,
    mark_final_live=True,
    fig=None,
)

plt.show()
plt.close()


"""
The traceplot plots traces and marginalized posteriors for each parameter.
"""
dyplot.traceplot(
    results=search_internal.results,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    smooth=0.02,
    thin=1,
    dims=None,
    post_color="blue",
    post_kwargs=None,
    kde=True,
    nkde=1000,
    trace_cmap="plasma",
    trace_color=None,
    trace_kwargs=None,
    connect=False,
    connect_highlight=10,
    connect_color="red",
    connect_kwargs=None,
    max_n_ticks=5,
    use_math_text=False,
    label_kwargs={"fontsize": 16},
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": "10"},
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    verbose=False,
    fig=None,
)

plt.show()
plt.close()

"""
__EmceePlotter__

We now pass the samples to a `MCMCPlotter` which will allow us to use emcee's in-built plotting libraries to 
make figures.

The emcee readthedocs describes fully all of the methods used below 

 - https://emcee.readthedocs.io/en/stable/user/sampler/
 
 The plotter wraps the `corner` method of the library `corner.py` to make corner plots of the PDF:

- https://corner.readthedocs.io/en/latest/index.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
plotter = aplt.MCMCPlotter(samples=result.samples)


"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter in the model fit.
"""
plotter.corner_cornerpy(
    bins=20,
    range=None,
    color="k",
    hist_bin_factor=1,
    smooth=None,
    smooth1d=None,
    label_kwargs=None,
    titles=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="#4682b4",
    scale_hist=False,
    quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    divergences=False,
    divergences_kwargs=None,
    labeller=None,
)


"""
__Search Specific Visualization__

The internal sampler can be used to plot the results of the non-linear search. 

We do this using the `search_internal` attribute which contains the sampler in its native form.

The first time you run a search, the `search_internal` attribute will be available because it is passed ot the
result via memory. 

If you rerun the fit on a completed result, it will not be available in memory, and therefore
will be loaded from the `files/search_internal` folder. The `search_internal` entry of the `output.yaml` must be true 
for this to be possible.
"""
search_internal = result.search_internal

"""
The method below shows a 2D projection of the walker trajectories.
"""
fig, axes = plt.subplots(result.model.prior_count, figsize=(10, 7))

for i in range(result.model.prior_count):
    for walker_index in range(search_internal.get_log_prob().shape[1]):
        ax = axes[i]
        ax.plot(
            search_internal.get_chain()[:, walker_index, i],
            search_internal.get_log_prob()[:, walker_index],
            alpha=0.3,
        )

    ax.set_ylabel("Log Likelihood")
    ax.set_xlabel(result.model.parameter_labels_with_superscripts_latex[i])

plt.show()

"""
This method shows the likelihood as a series of steps.
"""

fig, axes = plt.subplots(1, figsize=(10, 7))

for walker_index in range(search_internal.get_log_prob().shape[1]):
    axes.plot(search_internal.get_log_prob()[:, walker_index], alpha=0.3)

axes.set_ylabel("Log Likelihood")
axes.set_xlabel("step number")

plt.show()

"""
This method shows the parameter values of every walker at every step.
"""
fig, axes = plt.subplots(result.samples.model.prior_count, figsize=(10, 7), sharex=True)

for i in range(result.samples.model.prior_count):
    ax = axes[i]
    ax.plot(search_internal.get_chain()[:, :, i], alpha=0.3)
    ax.set_ylabel(result.model.parameter_labels_with_superscripts_latex[i])

axes[-1].set_xlabel("step number")

plt.show()

"""
__ZeusPlotter__

We now pass the samples to a `ZeusPlotter` which will allow us to use Nautilus's in-built plotting libraries to 
make figures.

The zeus readthedocs describes fully all of the methods used below 

 - https://zeus-mcmc.readthedocs.io/en/latest/api/plotting.html#cornerplot
 - https://zeus-mcmc.readthedocs.io/en/latest/notebooks/normal_distribution.html
 
 The plotter wraps the `corner` method of the library `corner.py` to make corner plots of the PDF:

- https://corner.readthedocs.io/en/latest/index.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
plotter = aplt.MCMCPlotter(samples=result.samples)

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter in the model fit.
"""
plotter.corner_cornerpy(
    weight_list=None,
    levels=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    truth=None,
    color=None,
    alpha=0.5,
    linewidth=1.5,
    fill=True,
    fontsize=10,
    show_titles=True,
    title_fmt=".2f",
    title_fontsize=12,
    cut=3,
    fig=None,
    size=(10, 10),
)


"""
__Search Specific Visualization__

The internal sampler can be used to plot the results of the non-linear search. 

We do this using the `search_internal` attribute which contains the sampler in its native form.

For zeus, the `search_internal` attribute is only available if the zeus sampler results are output to hard-disk
via hdf5. The `search_internal` entry of the `output.yaml` must be true for this to be the case.
"""
search_internal = result.search_internal

"""
__Plots__

The method below shows a 2D projection of the walker trajectories.
"""
fig, axes = plt.subplots(result.model.prior_count, figsize=(10, 7))

for i in range(result.model.prior_count):
    for walker_index in range(search_internal.get_log_prob().shape[1]):
        ax = axes[i]
        ax.plot(
            search_internal.get_chain()[:, walker_index, i],
            search_internal.get_log_prob()[:, walker_index],
            alpha=0.3,
        )

    ax.set_ylabel("Log Likelihood")
    ax.set_xlabel(result.model.parameter_labels_with_superscripts_latex[i])

plt.show()

"""
This method shows the likelihood as a series of steps.
"""

fig, axes = plt.subplots(1, figsize=(10, 7))

for walker_index in range(search_internal.get_log_prob().shape[1]):
    axes.plot(search_internal.get_log_prob()[:, walker_index], alpha=0.3)

axes.set_ylabel("Log Likelihood")
axes.set_xlabel("step number")

plt.show()

"""
This method shows the parameter values of every walker at every step.
"""
fig, axes = plt.subplots(result.samples.model.prior_count, figsize=(10, 7), sharex=True)

for i in range(result.samples.model.prior_count):
    ax = axes[i]
    ax.plot(search_internal.get_chain()[:, :, i], alpha=0.3)
    ax.set_ylabel(result.model.parameter_labels_with_superscripts_latex[i])

axes[-1].set_xlabel("step number")

plt.show()

"""
__PySwarmsPlotter__

We now pass the samples to a `MLEPlotter` which will allow us to use pyswarms's in-built plotting libraries to 
make figures.

The pyswarms readthedocs describes fully all of the methods used below 

 - https://pyswarms.readthedocs.io/en/latest/api/pyswarms.utils.plotters.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
plotter = aplt.MLEPlotter(samples=result.samples)

"""
__Search Specific Visualization__

The internal sampler can be used to plot the results of the non-linear search. 

We do this using the `search_internal` attribute which contains the sampler in its native form.

The first time you run a search, the `search_internal` attribute will be available because it is passed ot the
result via memory. 

If you rerun the fit on a completed result, it will not be available in memory, and therefore
will be loaded from the `files/search_internal` folder. The `search_internal` entry of the `output.yaml` must be true 
for this to be possible.
"""
search_internal = result.search_internal

"""
__Plots__

The `contour` method shows a 2D projection of the particle trajectories.
"""
from pyswarms.utils import plotters

plotters.plot_contour(
    pos_history=search_internal.pos_history,
    canvas=None,
    title="Trajectories",
    mark=None,
    designer=None,
    mesher=None,
    animator=None,
)
plt.show()

plotters.plot_cost_history(
    cost_history=search_internal.cost_history,
    ax=None,
    title="Cost History",
    designer=None,
)
plt.show()

"""
__GetDist__

This example illustrates how to plot visualization summarizing the results of model-fit using any non-linear search
using GetDist:

 - https://getdist.readthedocs.io/en/latest/

GetDist is an optional library which creates 1D and 2D plots of probability distribution functions (PDF)s. Its
visualization tools has more than the in-built visualization tools of many non-linear searches (e.g. Nautilus /
emcee) and can often produce better looking plots.

GetDist was developed for the analysis of Cosmological datasets.

Because GetDist is an optional library, you will likely have to install it manually via the command:

`pip install getdist`
"""
from getdist import MCSamples
from getdist import plots
import numpy as np

"""
GetDist uses a `model.paramnames` file to load the name of every parameter in the model-fit and pair it with the
latex symbol used to represent it in plots.

This file is not created by default, but can be output by the `search.paths` object as shown below.
"""
search.paths._save_parameter_names_file(model=model)
search.paths.zip_remove()
search.paths._zip()

"""
GetDist uses an `MCSamples` object to store the samples of a non-linear search.

We create this object via a conversion from **PyAutoFit** `Samples`, as well as using the `names`
and `labels` of parameters in the `Samples` object.

The input `sampler="nested"` is input because we used a nested sampling, `Nautilus`. For MCMC this should be
replaced with "mcmc".
"""
samples = result.samples

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

output_path = Path("output")

plt.savefig(Path(output_path, "getdist.png"))
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
