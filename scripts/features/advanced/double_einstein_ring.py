"""
Modeling Features: Double Einstein Ring
=======================================

A double Einstein ring lens is a strong lens system where there are two source galaxies at different redshifts 
behind the lens galaxy. They appear as two distinct Einstein rings in the image-plane, and can constrain 
Cosmological parameters in a way single Einstein ring lenses cannot.

To analyse these systems correctly the mass of the lens galaxy and the first source galaxy must be modeled 
simultaneously, and the emission of both source galaxies must be modeled simultaneously. 

This script illustrates the PyAutoLens API for modeling a double Einstein ring lens.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a double Einstein ring where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The first lens galaxy's total mass distribution is an `Isothermal`.
 - The second lens galaxy / first source galaxy's light is a linear parametric `ExponentialSph` and its mass a `IsothermalSph`.
 - The second source galaxy's light is a linear parametric `ExponentialSph`.

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

Load and plot the strong lens dataset `double_einstein_ring` via .fits files.

This dataset has a double Einstien ring, due to the two source galaxies at different redshifts behind the lens galaxy.
"""
dataset_name = "double_einstein_ring"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
Visualization of this dataset shows two distinct Einstein rings, which are the two source galaxies.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of both of the lensed source galaxies.
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

 - The first lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 
 - The second lens / first source galaxy's light is a linear parametric `ExponentialSph` and its mass 
 a `IsothermalSph` [6 parameters].

 - The second source galaxy's light is a linear parametric `ExponentialSph` [3 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=16.

Note that the galaxies are assigned redshifts of 0.5, 1.0 and 2.0. This ensures the multi-plane ray-tracing necessary
for the double Einstein ring lens system is performed correctly.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)
mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source 0:

bulge = af.Model(al.lp_linear.ExponentialCoreSph)
mass = af.Model(al.mp.IsothermalSph)

source_0 = af.Model(al.Galaxy, redshift=1.0, bulge=bulge, mass=mass)

# Source 1:

bulge = af.Model(al.lp_linear.ExponentialCoreSph)

source_1 = af.Model(al.Galaxy, redshift=2.0, bulge=bulge)

"""
__Cheating__

Initializing a double Einstein ring lens model is difficult, due to the complexity of parameter space. It is common to 
infer local maxima, which this script does if default broad priors on every model parameter are assumed.

To infer the correct model, we "cheat" and overwrite all of the priors of the model parameters to start centred on 
their true values.

For real data, we obviously do not know the true parameters and therefore cannot cheat in this way. Readers should
checkout the **PyAutoLens**'s advanced feature `chaining`, which chains together multiple non-linear searches. 

This feature is described in HowToLens chapter 3 and specific examples for a double Einstein ring are given in
the script `imaging/advanced/chaining/double_einstein_ring.py`.
"""
lens.bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.bulge.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.052, sigma=0.1)
lens.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.bulge.effective_radius = af.GaussianPrior(mean=0.8, sigma=0.2)
lens.bulge.sersic_index = af.GaussianPrior(mean=4.0, sigma=0.2)

lens.mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.052, sigma=0.1)
lens.mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.mass.einstein_radius = af.GaussianPrior(mean=1.5, sigma=0.2)

source_0.mass.centre_0 = af.GaussianPrior(mean=-0.15, sigma=0.2)
source_0.mass.centre_1 = af.GaussianPrior(mean=-0.15, sigma=0.2)
source_0.mass.einstein_radius = af.GaussianPrior(mean=0.4, sigma=0.1)
source_0.bulge.centre_0 = af.GaussianPrior(mean=-0.15, sigma=0.2)
source_0.bulge.centre_1 = af.GaussianPrior(mean=-0.15, sigma=0.2)
source_0.bulge.effective_radius = af.GaussianPrior(mean=0.1, sigma=0.1)

source_1.bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.2)
source_1.bulge.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.2)
source_1.bulge.effective_radius = af.GaussianPrior(mean=0.07, sigma=0.07)

"""
__Cosmology__

Double Einstein rings allow cosmological parameters to be constrained, because they provide information on the
different angular diameter distances between each source galaxy.

We therefore create a Cosmology as a `Model` object in order to make the cosmological parameter Omega_m a free 
parameter.
"""
cosmology = af.Model(al.cosmo.FlatwCDMWrap)

"""
By default, all parameters of a cosmology model are initialized as fixed values based on the Planck18 cosmology.

In order to make Omega_m a free parameter, we must manually overwrite its prior.
"""
cosmology.Om0 = af.GaussianPrior(mean=0.3, sigma=0.1)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1),
    cosmology=cosmology,
)

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms the model is composed of three galaxies, two of which are lensed source galaxies.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="double_einstein_ring",
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

The likelihood evaluation time for analysing double Einstein ring lens is quite a lot longer than single lens plane
lenses. This is because multi-plane ray-tracing calculations are computationally expensive. 

However, the real hit on run-time is the large number of free parameters in the model, which is often  10+ parameters
more than a single lens plane model. This means that the non-linear search takes longer to converge on a solution.
In this example, we cheated by initializing the priors on the model close to the correct solution. 

Combining pixelized source analyses with double Einstein ring lenses is very computationally expensive, because the
linear algebra calculations become significantly more expensive. This is not shown in this script, but is worth
baring in mind.
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

The search returns a result object, which whose `info` attribute shows the result in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this):
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

These plots show that the lens and both sources of the double Einstein ring were fitted successfully.
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

These examples show how the results API can be extended to investigate double Einstein ring results.

__Wrap Up__

Double Einstein ring systems can be fitted in **PyAutoLens**, however this script bypass the most difficult aspect
of fitting these systems by "cheating", and manually adjusting the priors to be near their true values.

Modeling real observations of double Einstein rings is one of the hardest lens modeling tasks, and requires an high
degree of lens modeling expertise to make a success.

If you have not already, I recommend you familiarize yourself with and use all of the following **PyAutoLens features
to model a real double Einstein ring:

 - Basis based light profiles (e.g. ``shapelets.ipynb` / `multi_gaussian_expansion.ipynb`): these allow one to fit
   complex lens and source morphologies whilst keeping the dimensionality of the problem low.
   
 - Search chaining (e.g. `imaging/advanced/chaining` and HowToLens chapter 3): by breaking the model-fit into a series
   of Nautilus searches models of gradually increasing complexity can be fitted.
   
 - Pixelizations (e.g. `pixelization.ipynb` and HowToLens chapter 4): to infer the cosmological parameters reliably
   the source must be reconstructed on an adaptive mesh to capture a irregular morphological features.
"""
