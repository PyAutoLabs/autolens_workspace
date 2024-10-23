"""
Modeling Features: Linear Light Profiles
========================================

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved for
via linear algebra every time the model is fitted to the data. This uses a process called an "inversion" and it
always computes the `intensity` values that give the best fit to the data (e.g. maximize the likelihood)
given the light profile's other parameters.

Based on the advantages below, we recommended you always use linear light profiles to fit models over standard
light profiles!

__Contents__

**Advantages & Disadvatanges:** Benefits and drawbacks of linear light profiles.
**Positive Only Solver:** How a positive solution to the light profile intensities is ensured.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Fit:** Perform a fit to a dataset using linear light profile with inputs for other light profile parameters.
**Intensities:** Access the solved for intensities of light profiles from the fit.
**Model:** Composing a model using linear light profiles and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Run Time:** Profiling of linear light profile run times and discussion of how they compare to standard light profiles.
**Model-Fit:** Performs the model fit using standard API.
**Result & Intensities:** Linear light profiles results, including how to access light profiles with solved for intensity values.
**Visualization:** Plotting images of model-fits using linear light profiles.
**Linear Objects (Source Code)**: Internal source code implementation of linear light profiles (for contributors).

__Advantages__

Each light profile's `intensity` parameter is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in this example by 2 dimensions).

This also removes the degeneracies that occur between the `intensity` and other light profile parameters
(e.g. `effective_radius`, `sersic_index`), which are difficult degeneracies for the non-linear search to map out
accurately. This produces more reliable lens model results and the fit converges in fewer iterations, speeding up the
overall analysis.

The inversion has a relatively small computational cost, thus we reduce the model complexity without much slow-down and
can therefore fit models more reliably and faster!

__Disadvantages__

Althought the computation time of the inversion is small, it is not non-negligable. It is approximately 3-4x slower
than using a standard light profile.

The gains in run times due to the simpler non-linear parameter space therefore are somewhat balanced by the slower
likelihood calculation.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physical.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a parametric linear `Sersic` bulge.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric linear `Sersic`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.

__Notes__

This script is identical to `modeling/start_here.py` except that the light profiles are switched to linear light 
profiles.
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

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Fit__

We now illustrate how to perform a fit to the dataset using linear light profils, using known light profile parameters.

The API follows closely the standard use of a `FitImaging` object, but simply uses linear light profiles (via the
`lp_linear` module) instead of standard light profiles. 

Note that the linear light profiles below do not have `intensity` parameters input and we use the true input values
of all other parameters for illustrative purposes.
"""
lens = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)


tracer = al.Tracer(galaxies=[lens, source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
By plotting the fit, we see that the linear light profiles have solved for `intensity` values that give a good fit
to the image. 
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()


"""
__Intensities__

The fit contains the solved for intensity values.

These are computed using a fit's `linear_light_profile_intensity_dict`, which maps each linear light profile 
in the model parameterization above to its `intensity`.

The code below shows how to use this dictionary, as an alternative to using the max_log_likelihood quantities above.
"""
lens_bulge = tracer.galaxies[0].bulge
source_bulge = tracer.galaxies[1].bulge

print(fit.linear_light_profile_intensity_dict)

print(
    f"\n Intensity of lens galaxy's bulge = {fit.linear_light_profile_intensity_dict[lens_bulge]}"
)

print(
    f"\n Intensity of source bulge (lp_linear.SersicCore) = {fit.linear_light_profile_intensity_dict[source_bulge]}"
)

"""
A `Tracer` where all linear light profile objects are replaced with ordinary light profiles using the solved 
for `intensity` values is also accessible from a fit.

For example, the linear light profile `Sersic` of the `bulge` component above has a solved for `intensity` of ~0.75. 

The `tracer` created below instead has an ordinary light profile with an `intensity` of ~0.75.

The benefit of using a tracer with standard light profiles is it can be visualized (linear light profiles cannot 
by default because they do not have `intensity` values).
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

print(tracer.galaxies[0].bulge.intensity)
print(tracer.galaxies[1].bulge.intensity)

"""
__Model__

We compose a lens model where:

 - The lens galaxy's light is a linear parametric `Sersic` bulge [6 parameters].
 
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a linear parametric `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.

Note how both the lens and source galaxies use linear light profiles, meaning that the `intensity` parameter of both
is no longer a free parameter in the fit.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Source:

bulge = af.Model(al.lp_linear.SersicCore)

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

In the `start_here.py` example 150 live points (`n_live=150`) were used to sample parameter space. For the linear
light profiles this is reduced to 100, as the simpler parameter space means we need fewer live points to map it out
accurately. This will lead to faster run times.
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="linear_light_profiles",
    unique_tag=dataset_name,
    n_live=100,
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
likely means that the fit will take less than 10000 per free parameter to converge. This is aided further
by the reduction in `n_live` to 100.

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
    tracer=result.max_log_likelihood_tracer, grid=result.grids.uniform
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
__Intensities__

The intensities of linear light profiles are not a part of the model parameterization and therefore are not displayed
in the `model.results` file.

To extract the `intensity` values of a specific component in the model, we use the `max_log_likelihood_tracer`,
which has already performed the inversion and therefore the galaxy light profiles have their solved for
`intensity`'s associated with them.
"""
tracer = result.max_log_likelihood_tracer

print(tracer.galaxies[0].bulge.intensity)
print(tracer.galaxies[1].bulge.intensity)

"""
Above, we access these values using the list index entry of each galaxy in the tracer. However, we may not be certain
of the order of the galaxies in the tracer, and therefore which galaxy index corresponds to the lens and source.

We can therefore use the model composition API to access these values.
"""
print(tracer.galaxies.lens.bulge.intensity)
print(tracer.galaxies.source.bulge.intensity)


"""
The `Tracer` contained in the `max_log_likelihood_fit` also has the solved for `intensity` values:
"""
fit = result.max_log_likelihood_fit

tracer = fit.tracer

print(tracer.galaxies.lens.bulge.intensity)
print(tracer.galaxies.source.bulge.intensity)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies, a tracer) cannot be plotted because they do not 
have an `intensity` value.

Therefore, the objects created above which replaces all linear light profiles with ordinary light profiles must be
used for visualization:
"""
tracer = result.max_log_likelihood_tracer

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=dataset.grid)
tracer_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=tracer.galaxies[0], grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Wrap Up__

Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.

In particular, checkout the results example `linear.py` which details how to extract all information about linear
light profiles from a fit.

__Result (Advanced)__

The code belows shows all additional results that can be computed from a `Result` object following a fit with a
linear light profile.

__Max Likelihood Inversion__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit`, which contains the
`Inversion` object we need.
"""
inversion = result.max_log_likelihood_fit.inversion

"""
This `Inversion` can be used to plot the reconstructed image of specifically all linear light profiles.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
# inversion_plotter.figures_2d(reconstructed_image=True)

"""
__Linear Objects (Internal Source Code)__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`. 

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).

- `Mapper`: The linear objected used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper` 
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectnagularMapper` is used for a `Rectangular` mesh).

In this example, two linear objects were used to fit the data:

 - An `Sersic` linear light profile for the source galaxy.
 ` A `Basis` containing 5 Gaussians for the lens galaxly's light. 
"""
print(inversion.linear_obj_list)

"""
To extract results from an inversion many quantities will come in lists or require that we specific the linear object
we with to use. 

Thus, knowing what linear objects are contained in the `linear_obj_list` and what indexes they correspond to
is important.
"""
print(f"LightProfileLinearObjFuncList (Basis) = {inversion.linear_obj_list[0]}")
print(f"LightProfileLinearObjFuncList (Sersic) = {inversion.linear_obj_list[1]}")

"""
Each of these `LightProfileLinearObjFuncList` objects contains its list of light profiles, which for the
`Sersic` is a single entry whereas for the `Basis` is 5 Gaussians.
"""
print(
    f"Linear Light Profile list (Basis -> x5 Gaussians) = {inversion.linear_obj_list[0].light_profile_list}"
)
print(
    f"Linear Light Profile list (Sersic) = {inversion.linear_obj_list[1].light_profile_list}"
)


"""
__Future Ideas / Contributions__

Not a lot tbh.
"""
