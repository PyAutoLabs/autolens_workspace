"""
Modeling Features: Mass Stellar Dark
====================================

The majority of example scripts fit a mass profile which represents the _total_ mass of the lens galaxy (its stars,
dark matter and other components combined). This typically uses an `Isothermal` or `PowerLaw` mass profile.

This script fits a mass model which decomposes the lens galaxy's mass into its stars and dark matter.

__Advantages__

Decomposed mass models measure direct properties of the stars and dark matter, for example the lens's stellar mass,
dark matter mass and the relative distribution between the two. Total mass profiles only inform us about the
superposition of these two components.

Decomposed mass models couple the lens galaxy's light profile to its stellar mass distribution, meaning that
additional information in the lens galaxy emission is used to constrain the mass model. Whilst total mass models
also fit the lens light, they do not couple it to the mass model and thus do not exploit this extra information.

Total mass models like the `Isothermal` and `PowerLaw` assume that the overall mass distribution of the lens galaxy
can be described using a single elliptical coordinate system. The stellar and dark components of a decomposed mass
model each have their own elliptical coordinate system, meaning that the mass model can be more complex and accurate.

__Disadvantages__

Assumptions must be made about how light and mass are coupled. This script assumes a constant mass-to-light raito,
however it is not clear this is a reliable assumption in many lens galaxies.

**PyAutoLens** supports more complex mass models which introduce a radial gradient into the mass-to-light ratio.
However, these are more complex and therefore are difficult to fit robustly. Furthermore, it is still not clear
whether the way they couple light to mass is a reliable assumption.

Performing ray-tracing with decomposed mass models is also more computationally expensive, meaning that the run times
of model-fits using these models is typically longer than total mass models.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a linear parametric `Sersic`.
 - The lens galaxy's stellar mass distribution is tied to the light model above.
 - The lens galaxy's dark matter mass distribution is a `NFW`.
 - The source galaxy's light is a linear parametric `SersicCore`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the strong lens dataset `mass_stellar_dark` via .fits files.
"""
dataset_name = "mass_stellar_dark"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

We compose a lens model where:

 - The lens galaxy's light and stellar mass is a linear parametric `Sersic` [7 parameters].
 
 - The lens galaxy's dark matter mass distribution is a `NFW` whose centre is aligned with the 
 `Sersic` bulge of the light and stellar mass model above [5 parameters].
 
 - The lens mass model also includes an `ExternalShear` [2 parameters].
 
 - The source galaxy's light is a linear parametric `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=22.

Note that for the stellar light and mass, we are using a "light and mass profile" via the `.lmp` package. This
profiles simultaneously acts like a light and mass profile.

For the dark matter, we use an `NFW`, which is a common mass profile to represent dark matter.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

bulge = af.Model(al.lmp.Sersic)
dark = af.Model(al.mp.NFW)
bulge.centre = dark.centre
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, dark=dark, shear=shear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

# Overall Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the lens model has both a `Sersic` light and mass profile and `NFW` dark matter profile, which 
are aligned.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a full 
description).
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="mass_stellar_dark",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Run Time__

The likelihood evaluation time for analysing stellar and dark matter mass models is longer than for total mass models
like the isothermal or power-law. This is because the deflection angles of these mass profiles are more expensive to
compute, requiring a Gaussian expansion or numerical calculation.

However, they have far fewer parameters than total mass models, when those models are also modeling the lens light. 
This is because many of the light and mass profile parameters are shared and fitted for simultaneously, reducing the
overall dimensionality of non-linear parameter space.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

These plots show that a decomposed stars and dark matter model is still able to produce ray-tracing and
the lensed source's emission.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grids.uniform
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.

These examples include a results API with specific tools for visualizing and analysing decomposed mass model,
for example 1D plots which separately show the density of stars and dark matter as a function of radius.

__Wrap Up__

Decomposed mass models have advantages and disavantages compared to total mass models.

The model which is best suited to your needs depends on the science you are hoping to undertake and the quality of the
data you are fitting.

In general, it is recommended that you first get fits going using total mass models, because they are simpler and make
fewer assumptions regarding how light is tied to mass. Once you have robust results, decomposed mass models can then
be fitted and compared in order to gain deeper insight.
"""
