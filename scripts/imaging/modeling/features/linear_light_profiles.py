"""
Modeling Features: Linear Light Profiles
========================================

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved for
via linear algebra every time the model is fitted to the data. This uses a process called an "inversion" and it
always computes the `intensity` values that give the best fit to the data (e.g. maximize the likelihood)
given the light profile's other parameters.

Each light profile's `intensity` parameter is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in this example, 3). They also removes
the degeneracies that occur between the `intnesity` and other light profile parameters (e.g. `effective_radius`,
`sersic_index`), which are some of the most difficult degeneracies for the model-fit to overcome!

Furthermore, the inversion has a neglible computational cost, thus we pretty much reduce the model complexity "for
free" and fit models more reliably and faster without any downside!

It is therefore recommend you always use linear light profiles to fit models over standard light profiles!

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a parametric `Sersic` bulge.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `Sersic`.

__Start Here Notebook__

If any code in this script is unclear, refer to the modeling `start_here.ipynb` notebook for more detailed comments.

__Notes__

This script is identical to `modeling/light_parametric__mass_total__source_lp.py` except that the light
profiles are switched to linear light profiles.
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

Load and plot the strong lens dataset `simple` via .fits files.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

We define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
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

 - The lens galaxy's light is a linear parametric `Sersic` bulge [6 parameters].
 
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a linear parametric `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=22.

__Model Cookbook__

A full description of model composition, including lens model customization, is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Source:

bulge = af.Model(al.lp_linear.Sersic)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the light profiles of the lens and source galaxies do not include an `intensity` parameter.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="linear_light_profiles",
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

For standard light profiles, the log likelihood evaluation time is of order ~0.01 seconds for this dataset.

For linear light profiles, the log likelihood evaluation increases to around ~0.05 seconds per likelihood evaluation.
This is still fast, but it does mean that the fit may take around five times longer to run.

However, because two free parameters have been removed from the model (the `intensity` of the lens bulge and 
source bulge), the total number of likelihood evaluations will reduce. Furthermore, the simpler parameter space
likely means that the fit will take less than 10000 per free parameter to converge.

Fits using standard light profiles and linear light profiles therefore take roughly the same time to run. However,
the simpler parameter space of linear light profiles means that the model-fit is more reliable, less susceptible to
converging to an incorrect solution and scales better if even more light profiles are included in the model.
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

This confirms that `intensity` parameters are not inferred by the model-fit.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

The lens and source galaxies appear similar to those in the data, confirming that the `intensity` values inferred by
the inversion process are accurate.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

search_plotter = aplt.DynestyPlotter(samples=result.samples)
search_plotter.cornerplot()

"""
__Intensities__

The intensities of linear light profiles are not a part of the model parameterization and therefore are not displayed
in the `model.results` file.

To extract the `intensity` values of a specific component in the model, we use the `max_log_likelihood_tracer`,
which has already had the inversion performed and therefore the galaxy light profiles have their solved for
`intensity`'s associated with them.
"""
fit = result.max_log_likelihood_fit

tracer = fit.tracer

lens_bulge = print(tracer.galaxies[0].bulge.intensity)
source_bulge = print(tracer.galaxies[1].bulge.intensity)


"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.

These examples show how the results API can be extended to investigate double Einstein ring results.
"""
