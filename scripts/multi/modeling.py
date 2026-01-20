"""
Modeling: Multi Modeling
========================

This script fits a multi-wavelength `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a MGE bulge where the `ell_comps` varies across wavelength.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a an MGE where the `ell_comps` varies across wavelength.

Two images are fitted, corresponding to a greener ('g' band) redder image (`r` band).

This is an advanced script and assumes previous knowledge of the core **PyAutoLens** API for lens modeling. Thus,
certain parts of code are not documented to ensure the script is concise.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).

The strings are used for load each dataset.
"""
waveband_list = ["g", "r"]

"""
__Pixel Scales__

Every multi-wavelength dataset can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.12]

"""
__Dataset__

Load and plot each multi-wavelength strong lens dataset, using a list of their waveband colors.

Note how the lens and source appear different brightnesses in each wavelength. Multi-wavelength image can therefore 
better separate the lens and source galaxies.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "lens_sersic"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

dataset_list = [
    al.Imaging.from_fits(
        data_path=Path(dataset_path) / f"{waveband}_data.fits",
        psf_path=Path(dataset_path) / f"{waveband}_psf.fits",
        noise_map_path=Path(dataset_path) / f"{waveband}_noise_map.fits",
        pixel_scales=pixel_scales,
    )
    for waveband, pixel_scales in zip(waveband_list, pixel_scales_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.

For multi-wavelength lens modeling, we use the same mask for every dataset whenever possible. This is not
absolutely necessary, but provides a more reliable analysis.
"""
mask_radius = 3.0

mask_list = [
    al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    for dataset in dataset_list
]


dataset_list = [
    dataset.apply_mask(mask=mask) for imaging, mask in zip(dataset_list, mask_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Model__

We compose a lens model where:

 - The lens galaxy's light is an MGE with 2 x 30 Gaussians, where the `intensity` parameter of the lens galaxy
 for each individual waveband of imaging is a different free parameter [6 parameters].

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's light is a an MGE, where the `intensity` parameter of the source galaxy
 for each individual waveband of imaging is a different free parameter [8 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=23.

__Model Extension__

Galaxies change appearance across wavelength, for example their ellipticities.

Models applied to combined analyses can be extended to include free parameters specific to each dataset. In this example,
we will make the galaxy's ellipticity vary across the g and r-band datasets, which will be illustrated below.

__Linear Light Profiles__

As an advanced user you should be familiar wiht linear light profiles, see elsewhere in the workspace for informaiton
if not.

For multi wavelength dataset modeling, the `lp_linear` API is extremely powerful as the `ell_comps` varies across
the datasets, meaning that making it linear reduces the dimensionality of parameter space significantly.
"""
total_gaussians = 20

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=total_gaussians,
    gaussian_per_basis=1,
    centre_prior_is_uniform=True,
)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Analysis List__

Set up two instances of the `Analysis` class object, one for each dataset.

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.
"""
analysis_list = [
    al.AnalysisImaging(
        dataset=dataset,
        use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
    )
    for dataset in dataset_list
]

"""
__Analysis Factor__

Each analysis object is wrapped in an `AnalysisFactor`, which pairs it with the model and prepares it for use in a 
factor graph. This step allows us to flexibly define how each dataset relates to the model.

The term "Factor" comes from factor graphs, a type of probabilistic graphical model. In this context, each factor 
represents the connection between one dataset and the shared model.

The API for extending the model across datasets is shown below, by overwriting the `ell_comps`
variables of the model passed to each `AnalysisFactor` object with new priors, making each dataset have its own
`ell_comps` free parameter.

NOTE: Other aspects of galaxies may vary across wavelength, none of which are included in this example. The API below 
can easily be extended to include these additional parameters, and the `features` package explains other tools for 
extending the model across datasets.
"""
analysis_factor_list = []

for analysis in analysis_list:

    model_analysis = model.copy()

    ell_comps_0_prior = af.GaussianPrior(mean=0.0, sigma=0.3)
    ell_comps_1_prior = af.GaussianPrior(mean=0.0, sigma=0.3)

    for i in range(total_gaussians):

        model_analysis.galaxies.lens.bulge.profile_list[i].ell_comps.ell_comps_0 = (
            ell_comps_0_prior
        )
        model_analysis.galaxies.lens.bulge.profile_list[i].ell_comps.ell_comps_1 = (
            ell_comps_1_prior
        )

    analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

All `AnalysisFactor` objects are combined into a `FactorGraphModel`, which represents a global model fit to 
multiple datasets using a graphical model structure.

The key outcomes of this setup are:

 - The individual log likelihoods from each `Analysis` object are summed to form the total log likelihood 
   evaluated during the model-fitting process.

 - Results from all datasets are output to a unified directory, with subdirectories for visualizations 
   from each analysis object, as defined by their `visualize` methods.

This is a basic use of **PyAutoFit**'s graphical modeling capabilities, which support advanced hierarchical 
and probabilistic modeling for large, multi-dataset analyses.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
To inspect this new model, with extra parameters for each dataset created, we 
print `factor_graph.global_prior_model.info`.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=Path(
        "multi_wavelength"
    ),  # The path where results and output are stored.
    name="modeling",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=150,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=10000,  # Every N iterations the max likelihood model is visualized in the Jupter Notebook and output to hard-disk.
)

"""
__VRAM Use__

The `modeling` examples of individual dataset types explain how VRAM is used during GPU-based fitting and how to 
print the estimated VRAM required by a model.

When multiple datasets are fitted simultaneously, as in this example, VRAM usage increases with each
dataset, as their data structures must all be stored in VRAM.

Given VRAM use is an important consideration, we print out the estimated VRAM required for this 
model-fit and advise you do this for your own pixelization model-fits.
"""
factor_graph.print_vram_use(
    model=factor_graph.global_prior_model, batch_size=search.batch_size
)

"""
__Model-Fit__

To fit multiple datasets, we pass the `FactorGraphModel` to a non-linear search.

Unlike single-dataset fitting, we now pass the `factor_graph.global_prior_model` as the model and 
the `factor_graph` itself as the analysis object.

This structure enables simultaneous fitting of multiple datasets in a consistent and scalable way.

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce 
an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("The search has finished run - you may now continue the notebook.")


"""
__Result__

The result object returned by this model-fit is a list of `Result` objects, because we used a factor graph.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.

For example, close inspection of the `max_log_likelihood_instance` of the two results shows that all parameters,
except the `effective_radius` of the source galaxy's `bulge`, are identical.
"""
print(result_list[0].max_log_likelihood_instance)
print(result_list[1].max_log_likelihood_instance)

"""
Plotting each result's tracer shows that the source appears different, owning to its different intensities.
"""
for result in result_list:
    tracer_plotter = aplt.TracerPlotter(
        tracer=result.max_log_likelihood_tracer, grid=result.grids.lp
    )
    tracer_plotter.subplot_tracer()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
The `Samples` object still has the dimensions of the overall non-linear search (in this case N=15). 

Therefore, the samples is identical in every result object.
"""
for result in result_list:
    plotter = aplt.NestPlotter(samples=result.samples)
    plotter.corner_anesthetic()

"""
__Wrap Up__

This simple example introduces the API for fitting multiple datasets with a shared model.

It should already be quite intuitive how this API can be adapted to fit more complex models, or fit different
datasets with different models. For example, an `AnalysisImaging` and `AnalysisInterferometer` can be combined, into
a single factor graph model, to simultaneously fit a imaging and interferometric data.

The `advanced/multi/modeling` package has more examples of how to fit multiple datasets with different models,
including relational models that vary parameters across datasets as a function of wavelength.
"""
