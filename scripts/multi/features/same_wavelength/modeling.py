"""
Modeling: Same Wavelength
=========================

This script fits a multiple `Imaging` datasets observed at the same wavelength of a 'galaxy-scale' strong lens with a
model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an MGE.

This script demonstrates how PyAutoLens's multi-dataset modeling tools can also simultaneously analyse datasets
observed at the same wavelength.

An example use case might be analysing undithered HST images before they are combined via the multidrizzing process,
to remove correlated noise in the data.

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
__Pixel Scales__

If observed at the same wavelength, it is likely the datasets have the same pixel-scale.

Nevertheless, we specify this as a list as there could be an exception.
"""
pixel_scales_list = [0.1, 0.1]

"""
__Dataset__

Load and plot each multi-wavelength strong lens dataset, using a list of their waveband colors.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "same_wavelength"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

dataset_list = [
    al.Imaging.from_fits(
        data_path=Path(dataset_path, f"image_{i}.fits"),
        psf_path=Path(dataset_path, f"psf_{i}.fits"),
        noise_map_path=Path(dataset_path, f"noise_map_{i}.fits"),
        pixel_scales=pixel_scales,
    )
    for i, pixel_scales in enumerate(pixel_scales_list)
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
__Analysis__

We create an `Analysis` object for every dataset.
"""
analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's light is an MGE with 1 x 20 Gaussians [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
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

"""
We now combine them using the factor analysis class, which allows us to fit the two datasets simultaneously.

Unlike other examples, no customization to the model is applied that, for example, adds more free parameters,
given that the datasets do not vary over wavelength.
"""
analysis_factor_list = []

for analysis in analysis_list:
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=True,
    )
    disk = af.Model(al.lp_linear.Exponential)

    galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

    model_analysis = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
The `info` of the model shows us there are two models each with linear light profiles.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=Path("multi", "modeling"),
    name="same_wavelength",
    unique_tag=dataset_name,
    n_live=100,
)

"""
__Model-Fit__
"""
result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result object returned by this model-fit is a list of `Result` objects, because we used a factor graph.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.
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
plotter = aplt.NestPlotter(samples=result_list.samples)
plotter.corner_anesthetic()

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results in **PyAutoLens**.
"""
