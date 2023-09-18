"""
Modeling Features: Multi Gaussian Expansion
===========================================

A multi Gaussian expansion (MGE) decomposes the lens light into a super positive of ~15-100 Gaussians, where 
the `intensity` of every Gaussian is solved for via a linear algebra using a process called an "inversion"
(see the `light_parametric_linear.py` feature for a full description of this).

This script fits a lens light model which uses an MGE consisting of 60 Gaussians. It is fitted to simulated data
where the lens galaxy's light has asymmetric and irregular features, which are not well fitted by symmetric light
profiles like the `Sersic`.

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals, because they fail to capture 
irregular and asymmetric morphological of galaxies (e.g. isophotal twists, an ellipticity which varies radially). 
An MGE fully captures these features and can therefore much better represent the emission of complex lens galaxies.

The MGE model can be composed in a way that has fewer non-linear parameters than an elliptical Sersic. In this example,
two separate groups of Gaussians are used to represent the `bulge` and `disk` of the lens, which in total correspond
to just N=6 non-linear parameters (a `bulge` and `disk` comprising two linear Sersics would give N=10). 

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

__MGE Source Galaxy__

The MGE was designed to model the light of lens galaxies, because they are typically elliptical galaxies whose
morphology is better represented as a super position of Gaussians. Their complex featgures (e.g. isophotal twists,
an ellipticity which varies radially) are accurately captured by an MGE.

The morphological features typically seen in source galaxies (e.g. disks, bars, clumps of star formation) are less
suited to an MGE. The source-plane of many lenses also often have multiple galaxies, whereas the MGE fitted
in this example assumes a single `centre`.

However, despite these limitations, an MGE turns out to be an extremely powerful way to model the source galaxies
of strong lenses. This is because, even if it struggles to capture the source's morphology, the simplification of
non-linear parameter space and removal of degeneracies makes it much easier to obtain a reliable source model.
This is driven by the removal of any non-linear parameters which change the size of the source's light profile,
which are otherwise the most degenerate with the lens's mass model.

The second example in this script therefore uses an MGE source. We strongly recommend you read that example and adopt
MGE lens light models and source models, instead of the elliptical Sersic profiles, as soon as possible!

To capture the irregular and asymmetric features of the source's morphology, or reconstruct multiple source galaxies,
we recommend using a pixelized source reconstruction (see `autolens_workspace/imaging/modeling/features/pixelization.py`).
Combining this with an MGE for the len's light can be a very powerful way to model strong lenses!

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For an MGE, this produces a positive-negative "ringing", where the
Gaussians alternate between large positive and negative values. This is clearly undesirable and unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physical.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's bulge is a super position of 60 `Gaussian`` profiles.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `Sersic`.

__Start Here Notebook__

If any code in this script is unclear, refer to the modeling `start_here.ipynb` notebook for more detailed comments.
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

 - The galaxy's bulge is a superposition of 60 parametric linear `Gaussian` profiles [6 parameters]. 
 - The centres and elliptical components of the Gaussians are all linked together.
 - The `sigma` size of the Gaussians increases in log10 increments.

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric linear `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.

Note that above we combine the MGE for the lens light with a linear light profile for the source, meaning these two
categories of models can be combined with ease.

__Model Cookbook__

A full description of model composition, including lens model customization, is provided by the model cookbook: 

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
    light_profile_list=bulge_gaussian_list,
)

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)


# bulge = af.Model(al.lp.Sersic)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.Sersic)

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
75 live points, speeding up converge of the non-linear search.
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
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

search_plotter = aplt.NautilusPlotter(samples=result.samples)
search_plotter.cornerplot()

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
    light_profile_list=bulge_gaussian_list,
)
mass = af.Model(al.mp.Isothermal)
lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source:

total_gaussians = 20
gaussian_per_basis = 1

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
    light_profile_list=bulge_gaussian_list,
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

For **PyAutoLens**'s advanced chaining feature, it is common use the MGE to initialize the lens light and source 
models. The lens light model is then made more complex by using an MGE with more Gaussians and the source becomes a 
pixelized reconstruction.

Now you are familiar with MGE modeling, it is recommended you adopt this as your default lens modeling approach. 
However, it may not be suitable for lower resolution data, where the simpler Sersic profiles may be more appropriate.

__Regularization (Advanced / Unused)__

An MGE can be regularized, whereby smoothness is enforced on the `intensity` values of the Gaussians. This was 
implemented to avoid a "positive / negative" ringing effect in the lens light model reconstruction, whereby the 
Gaussians went to a systematic solution which alternated between positive and negative values. 

Regularization was intended to smooth over the `intensity` values of the Gaussians, such that the solution would prefer
a positive-only solution. However, this did not work -- even with high levels of regularization, the Gaussians still
went to negative values. The solution also became far from optimal, often leaving significant residuals in the lens
light model reconstruction.

This problem was solved by switching to a positive-only linear algebra solver, which is the default used 
in **PyAutoLens** and was used for all fits performed above. The regularization feature is currently not used by
any scientific analysis and it is recommended you skip over the example below and do not use it in your own modeling.

However, its implementation is detailed below for completeness, and if you think you have a use for it in your own
modeling then go ahead! Indeed, even with a positive-only solver, it may be that regularization helps prevent overfitting
in certain situations.

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
    light_profile_list=bulge_gaussian_list,
    regularization=regularization,
)
mass = af.Model(al.mp.Isothermal)
lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source:

bulge = af.Model(al.lp_linear.Sersic)
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
