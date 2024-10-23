"""
Modeling Features: Multi Gaussian Expansion
===========================================

A multi Gaussian expansion (MGE) decomposes the lens light into ~15-100 Gaussians, where the `intensity` of every 
Gaussian is solved for via a linear algebra using a process called an "inversion" (see the `light_parametric_linear.py` 
feature for a full description of this).

This script fits a lens light model which uses an MGE consisting of 60 Gaussians. It is fitted to simulated data
where the lens galaxy's light has asymmetric and irregular features, which fitted poorly by symmetric light
profiles like the `Sersic`.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the light profile intensities is ensured.
**MGE Source Galaxy:** Discussion of using the MGE for the source galaxy, which is illustrated fully at the end of the example.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Basis:** How to create a basis of multiple light profiles, in this example Gaussians.
**Gaussians:** A visualization of the Gaussians in the Basis that make up the MGE.
**Linear Light Profiles:** How to create a basis of linear light profiles to perform the MGE.
**Fit:** Perform a fit to a dataset using linear light profile MGE.
**Intensities:** Access the solved for intensities of linear light profiles from the fit.
**Model:** Composing a model using an MGE and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Run Time:** Profiling of MGE run times and discussion of how they compare to standard light profiles.
**Model-Fit:** Performs the model fit using standard API.
**Result:** MGE results, including accessing light profiles with solved for intensity values.
**MGE Source:** Detailed illustration of using MGE source.
**Regularization:** API for applying regularization to MGE, which is not recommend but included for illustration.

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals, because they fail to capture 
irregular and asymmetric morphological of galaxies (e.g. isophotal twists, an ellipticity which varies radially). 
An MGE fully captures these features and can therefore much better represent the emission of complex lens galaxies.

The MGE model can be composed in a way that has fewer non-linear parameters than an elliptical Sersic. In this example,
two separate groups of Gaussians are used to represent the `bulge` and `disk` of the lens, which in total correspond
to just N=6 non-linear parameters (a `bulge` and `disk` comprising two linear Sersics has N=10 parameters). 

The MGE model parameterization is also composed such that neither the `intensity` parameters or any of the
parameters controlling the size of the Gaussians (their `sigma` values) are non-linear parameters sampled by Nautilus.
This removes the most significant degeneracies in parameter space, making the model much more reliable and efficient
to fit.

Therefore, not only does an MGE fit more complex galaxy morphologies, it does so using fewer non-linear parameters
in a much simpler non-linear parameter space which has far less significant parameter degeneracies!

__Disadvantages__

To fit an MGE model to the data, the light of the ~15-75 or more Gaussian in the MGE must be evaluated and compared
to the data. This is slower than evaluating the light of ~2-3 Sersic profiles, producing slower computational run
times (although the simpler non-linear parameter space will speed up the fit overall).

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For an MGE, this produces a positive-negative "ringing", where the
Gaussians alternate between large positive and negative values. This is clearly undesirable and unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physical.

__MGE Source Galaxy__

The MGE was designed to model the light of lens galaxies, because they are typically elliptical galaxies whose
morphology is better represented as many Gaussians. Their complex features (e.g. isophotal twists,
an ellipticity which varies radially) are accurately captured by an MGE.

The morphological features typically seen in source galaxies (e.g. disks, bars, clumps of star formation) are less
suited to an MGE. The source-plane of many lenses also often have multiple galaxies, whereas the MGE fitted
in this example assumes a single `centre`.

However, despite these limitations, an MGE turns out to be an extremely powerful way to model the source galaxies
of strong lenses. This is because, even if it struggles to capture the source's morphology, the simplification of
non-linear parameter space and removal of degeneracies makes it much easier to obtain a reliable lens model.
This is driven by the removal of any non-linear parameters which change the size of the source's light profile,
which are otherwise the most degenerate with the lens's mass model.

The second example in this script therefore uses an MGE source. We strongly recommend you read that example and adopt
MGE lens light models and source models, instead of the elliptical Sersic profiles, as soon as possible!

To capture the irregular and asymmetric features of the source's morphology, or reconstruct multiple source galaxies,
we recommend using a pixelized source reconstruction (see `autolens_workspace/modeling/imaging/features/pixelization.py`).
Combining this with an MGE for the len's light can be a very powerful way to model strong lenses!

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's bulge is a super position of 60 `Gaussian`` profiles.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear parametric `SersicCore`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the strong lens dataset `simple` via .fits files.
"""
dataset_name = "lens_light_asymmetric"
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
__Basis__

We first build a `Basis`, which is built from multiple light profiles (in this case, Gaussians). 

Below, we make a `Basis` out of 30 elliptical Gaussian light profiles which: 

 - All share the same centre and elliptical components.
 - The `sigma` size of the Gaussians increases in log10 increments.

Note that any light profile can be used to compose a Basis, but Gaussians are a good choice for lens galaxies
because they can capture the structure of elliptical galaxies.
"""
total_gaussians = 30

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".

mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# A list of linear light profile Gaussians will be input here, which will then be used to fit the data.

bulge_gaussian_list = []

# Iterate over every Gaussian and create it, with it centered at (0.0", 0.0") and assuming spherical symmetry.

for i in range(total_gaussians):
    gaussian = al.lp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=10 ** log10_sigma_list[i],
    )

    bulge_gaussian_list.append(gaussian)

# The Basis object groups many light profiles together into a single model component and is used to fit the data.

bulge = al.lp_basis.Basis(profile_list=bulge_gaussian_list)

"""
__Gaussians__

The `Basis` is composed of many Gaussians, each with different sizes (the `sigma` value) and therefore capturing
emission on different scales.

These Gaussians are visualized below using a `BasisPlotter`, which shows that the Gaussians expand in size as the
sigma value increases, in log10 increments.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

basis_plotter = aplt.BasisPlotter(basis=bulge, grid=grid)
basis_plotter.subplot_image()


"""
__Linear Light Profiles__

We now show Composing a basis of multiple Gaussians and use them to fit the lens galaxy's light in data.

This does not perform a model-fit via a non-linear search, and therefore requires us to manually specify and guess
suitable parameter values for the shapelets (e.g. the `centre`, `ell_comps`). However, Gaussians are
very flexible and will give us a decent looking lens fit even if we just guess sensible values
for each parameter. 

The one parameter that is tricky to guess is the `intensity` of each Gaussian. A wide range of positive `intensity` 
values are required to decompose the lens galaxy's light accurately. We certainly cannot obtain a good solution by 
guessing the `intensity` values by eye.

We therefore use linear light profile Gaussians, which determine the optimal value for each Gaussian's `intensity` 
via linear algebra. Linear light profiles are described in the `linear_light_profiles.py` example and you should
familiarize yourself with this example before using the multi-Gaussian expansion.

We therefore again setup a `Basis` in an analogous fashion to the previous example, but this time we use linear
Gaussians (via the `lp_linear.linear` module).
"""
total_gaussians = 30

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".

mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# A list of linear light profile Gaussians will be input here, which will then be used to fit the data.

bulge_gaussian_list = []

# Iterate over every Gaussian and create it, with it centered at (0.0", 0.0") and assuming spherical symmetry.

for i in range(total_gaussians):
    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        sigma=10 ** log10_sigma_list[i],
    )

    bulge_gaussian_list.append(gaussian)

# The Basis object groups many light profiles together into a single model component and is used to fit the data.

bulge = al.lp_basis.Basis(profile_list=bulge_gaussian_list)

"""
__Fit__

This is to illustrate the API for performing an MGE using standard autolens objects like the `Galaxy`, `Tracer`
and `FitImaging` 

Once we have a `Basis`, we can treat it like any other light profile in order to create a `Galaxy` and `Tracer` and 
use it to fit data.
"""
lens = al.Galaxy(
    redshift=0.5,
    bulge=bulge,
)

tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
By plotting the fit, we see that the MGE does a reasonable job at capturing the appearance of the lens galaxy.

The majority of residuals are due to the lensed source, which was not included in the model. There are faint
central residuals, which are due to the MGE not being a perfect fit to the lens galaxy's light. 

Given that there was no non-linear search to determine the optimal values of the Gaussians and the source galaxy
was omitted entirely, this is a pretty good fit!
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
We can use the `BasisPlotter` to plot each individual Gaussian in the reconstructed basis.

This plot shows each Gaussian has a unique positive `intensity` that was solved for via linear algebra.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

basis_plotter = aplt.BasisPlotter(basis=tracer.galaxies[0].bulge, grid=grid)
basis_plotter.subplot_image()

"""
__Intensities__

The fit contains the solved for intensity values.

These are computed using a fit's `linear_light_profile_intensity_dict`, which maps each linear light profile 
in the model parameterization above to its `intensity`.

The code below shows how to use this dictionary, as an alternative to using the max_log_likelihood quantities above.
"""
lens_bulge = fit.tracer.galaxies[0].bulge

print(
    f"\n Intensity of lens galaxy's first Gaussian in bulge = {fit.linear_light_profile_intensity_dict[lens_bulge.profile_list[0]]}"
)

"""
A `Tracer` where all linear light profile objects are replaced with ordinary light profiles using the solved 
for `intensity` values is also accessible from a fit.

For example, the first linear light profile of the MGE `bulge` component above printed it solved for intensity value,
but it was still represented as a linear light profile. 

The `tracer` created below instead has a standard light profile with an `intensity` actually set.

The benefit of using a tracer with standard light profiles is it can be visualized, as performed above (linear 
light profiles cannot by default because they do not have `intensity` values).
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

print(tracer.galaxies[0].bulge.profile_list[0].intensity)

"""
__Model__

The MGE above produced residuals, which we now rectify by fitting the MGE in a non-linear search, simultaneously
fitting the lens's mass and source galaxies.

We compose a lens model where:

 - The galaxy's bulge is 60 parametric linear `Gaussian` profiles [6 parameters]. 
 - The centres and elliptical components of the Gaussians are all linked together in two groups of 30.
 - The `sigma` size of the Gaussians increases in log10 increments.

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric linear `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.

Unlike the examples above, our MGE now comprises 2 * 30 Gaussians (instead of 1 * 30). Each group of 30
have their own elliptical components meaning the lens's light is descomposed into two distinct elliptical components,
which could be viewed as a bulge and disk.

Note that above we combine the MGE for the lens light with a linear light profile for the source, meaning these two
categories of models can be combined.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
total_gaussians = 30
gaussian_per_basis = 2

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    # A list of Gaussian model components whose parameters are customized belows.

    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    # Iterate over every Gaussian and customize its parameters.

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
        gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
        gaussian.ell_comps = gaussian_list[
            0
        ].ell_comps  # All Gaussians have same elliptical components.
        gaussian.sigma = (
            10 ** log10_sigma_list[i]
        )  # All Gaussian sigmas are fixed to values above.

    bulge_gaussian_list += gaussian_list

# The Basis object groups many light profiles together into a single model component.

bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This shows every single Gaussian light profile in the model, which is a lot of parameters! However, the vast
majority of these parameters are fixed to the values we set above, so the model actually has far fewer free
parameters than it looks!
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).

Owing to the simplicity of fitting an MGE we an use even fewer live points than other examples, reducing it to
75 live points, speeding up convergence of the non-linear search.
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="mge",
    unique_tag=dataset_name,
    n_live=75,
    number_of_cores=1,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Run Time__

The likelihood evaluation time for a multi-Gaussian expansion is significantly slower than standard / linear 
light profiles. This is because the image of every Gaussian must be computed and evaluated, and each must be blurred 
with the PSF. In this example, the evaluation time is ~0.5s, compared to ~0.01 seconds for standard light profiles.

Huge gains in the overall run-time however are made thanks to the models significantly reduced complexity and lower
number of free parameters. Furthermore, because there are not free parameters which scale the size of lens galaxy,
this produces significantly faster convergence by Nautilus that any other lens light model. We also use fewer live
points, further speeding up the model-fit.

Overall, it is difficult to state which approach will be faster overall. However, the MGE's ability to fit the data
more accurately and the less complex parameter due to removing parameters that scale the lens galaxy make it the 
superior approach.
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

The search returns a result object, which whose `info` attribute shows the result in a readable format (if this does 
not display clearly on your screen refer to `start_here.ipynb` for a description of how to fix this):

This confirms there are many `Gaussian`' in the lens light model and it lists their inferred parameters.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.

In particular, checkout the results example `linear.py` which details how to extract all information about linear
light profiles from a fit.
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
__Source MGE__

As discussed at the beginning of this tutorial, an MGE is an effective way to model the light of a source galaxy and 
get an initial estimate of the lens mass model.

This MGE source is used alongside the MGE lens light model, which offers a lot of flexibility in modeling the lens
and source galaxies. Note that the MGE source uses fewer Gaussians, as the MGE only needs to capture the main
structure of the source galaxy's light to obtain an accurate lens model.

We compose the model below, recreating the Gaussian's line-by-line in Python, so you can easily copy and paste
and reuse the code below in your own scripts.
"""
# Lens:

total_gaussians = 30
gaussian_per_basis = 2

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    # A list of Gaussian model components whose parameters are customized belows.

    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    # Iterate over every Gaussian and customize its parameters.

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
        gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
        gaussian.ell_comps = gaussian_list[
            0
        ].ell_comps  # All Gaussians have same elliptical components.
        gaussian.sigma = (
            10 ** log10_sigma_list[i]
        )  # All Gaussian sigmas are fixed to values above.

    bulge_gaussian_list += gaussian_list

# The Basis object groups many light profiles together into a single model component.

bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)
mass = af.Model(al.mp.Isothermal)
lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source:

total_gaussians = 20
gaussian_per_basis = 1

# By defining the centre here, it creates two free parameters that are assigned to the source Gaussians.

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

log10_sigma_list = np.linspace(-2, np.log10(1.0), total_gaussians)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    bulge_gaussian_list += gaussian_list

source_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
Printing the model info confirms the model has Gaussians for both the lens and source galaxies.
"""
print(model.info)

"""
We now fit this model, which includes the MGE source and lens light models.
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="mge_including_source",
    unique_tag=dataset_name,
    n_live=75,
    number_of_cores=1,
)

"""
__Run Time__

The likelihood evaluation time for a multi-Gaussian expansion for both lens and source galaxies is not much slower
than when just the lens galaxy uses an MGE.

However, the overall run-time will be even faster than before, as treating the source as an MGE further
reduces the complexity of parameter space ensuring Nautilus converges even faster.

For initial model-fits where the lens model parameters are not known, a lens + source MGE is possibly the best
model one can use. 
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
__Wrap Up__

A Multi Gaussian Expansion is a powerful tool for modeling the light of galaxies, and offers a compelling method to
fit complex light profiles with a small number of parameters.

For **PyAutoLens**'s advanced search chaining feature, it is common use the MGE to initialize the lens light and source 
models. The lens light model is then made more complex by using an MGE with more Gaussians and the source becomes a 
pixelized reconstruction.

Now you are familiar with MGE modeling, it is recommended you adopt this as your default lens modeling approach. 
However, it may not be suitable for lower resolution data, where the simpler Sersic profiles may be more appropriate.

__Regularization (Advanced / Unused)__

An MGE can be regularized, whereby smoothness is enforced on the `intensity` values of the Gaussians. However,
this feature proved to be problematic and is currently not used by any scientific analysis. It is still supported
and documented below for completeness, but it is recommended you skip over this section and do not use it in your own
modeling unless you have a specific reason to do so.

Regularization was implemented to avoid a "positive / negative" ringing effect in the lens light model reconstruction, 
whereby the  Gaussians went to a systematic solution which alternated between positive and negative values. 

Regularization was intended to smooth over the `intensity` values of the Gaussians, such that the solution would prefer
a positive-only solution. However, this did not work -- even with high levels of regularization, the Gaussians still
went to negative values. The solution also became far from optimal, often leaving significant residuals in the lens
light model reconstruction.

This problem was solved by switching to a positive-only linear algebra solver, which is the default used 
in **PyAutoLens** and was used for all fits performed above. The regularization feature is currently not used by
any scientific analysis and it is recommended you skip over the example below and do not use it in your own modeling.

However, its implementation is detailed below for completeness, and if you think you have a use for it in your own
modeling then go ahead! Indeed, even with a positive-only solver, it may be that regularization helps prevent 
overfitting in certain situations.

__Description__

There is one downside to `Basis` functions, we may compose a model with too much freedom. The `Basis` (e.g. our 20
Gaussians) may overfit noise in the data, or possible the lensed source galaxy emission -- neither of which we 
want to happen! 

To circumvent this issue, we have the option of adding regularization to a `Basis`. Regularization penalizes
solutions which are not smooth -- it is essentially a prior that says we expect the component the `Basis` represents
(e.g. a bulge or disk) to be smooth, in that its light changes smoothly as a function of radius.

Below, we compose and fit a model using Basis functions which includes regularization, which adds one addition 
parameter to the fit, the `coefficient`, which controls the degree of smoothing applied.
"""
# Lens:

regularization = af.Model(al.reg.Constant)
bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
    regularization=regularization,
)
mass = af.Model(al.mp.Isothermal)
lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source:

bulge = af.Model(al.lp_linear.SersicCore)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model, which has addition priors now associated with regularization.
"""
print(model.info)

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="mge_regularized",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)

"""
__Run Time__

Regularization has a small impact on the run-time of the model-fit, as the likelihood evaluation time does not
change and it adds only 1 additional parameter.
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
Finish.
"""
