"""
Plots: DynestyPlotter
=====================

This example illustrates how to plot visualization summarizing the results of a Nautilus non-linear search using
a `ZeusPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
from os import path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
First, lets create a result via Nautilus by repeating the simple model-fit that is performed in 
the `modeling/start_here.py` example.
"""
dataset_name = "simple__no_lens_light"

search = af.DynestyStatic(
    path_prefix=path.join("plot"),
    name="DynestyPlotter",
    unique_tag=dataset_name,
    n_live=100,
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

"""
__Notation__

Plot are labeled with short hand parameter names (e.g. the `centre` parameters are plotted using an `x`). 

The mappings of every parameter to its shorthand symbol for plots is specified in the `config/notation.yaml` file 
and can be customized.

Each label also has a superscript corresponding to the model component the parameter originates from. For example,
Gaussians are given the superscript `g`. This can also be customized in the `config/notation.yaml` file.

__Plotting__

We now pass the samples to a `DynestyPlotter` which will allow us to use Nautilus's in-built plotting libraries to 
make figures.

The Nautilus readthedocs describes fully all of the methods used below 

 - https://nautilus-sampler.readthedocs.io/en/latest/quickstart.html
 - https://nautilus-sampler.readthedocs.io/en/latest/api.html#module-Nautilus.plotting
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.

Nautilus plotters use `_kwargs` dictionaries to pass visualization settings to matplotlib lib. For example, below,
we:

 - Set the fontsize of the x and y labels by passing `label_kwargs={"fontsize": 16}`.
 - Set the fontsize of the title by passing `title_kwargs={"fontsize": "10"}`.
 
There are other `_kwargs` inputs we pass as None, you should check out the Nautilus docs if you need to customize your
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
Finish.
"""
