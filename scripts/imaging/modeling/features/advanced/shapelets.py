"""
Modeling Features: Shapelets
============================

A shapelet is a basis function that is appropriate for capturing the exponential / disk-like features of a galaxy. It
has been employed in many strong lensing studies too model the light of the lensed source galaxy, as it can represent
features of disky star forming galaxies that a single Sersic function cannot.

- https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T
- https://iopscience.iop.org/article/10.1088/0004-637X/813/2/102 t

Shapelets are described in full in the following paper:

 https://arxiv.org/abs/astro-ph/0105178

This script performs a model-fit using shapelet, where it decomposes the source light into ~200
Shapelets. The `intensity` of every Shapelet is solved for via an inversion (see the `light_parametric_linear.py`
feature).

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals, because they fail to capture
irregular and asymmetric morphological of galaxies (e.g. isophotal twists, an ellipticity which varies radially).
Shapelets can capture these features and can therefore much better represent the emission of complex source galaxies.

The shapelet model can be composed in a way that has fewer non-linear parameters than an elliptical Sersic. In this
example, the ~200 shapelets which represent the `bulge` of the source are composed in a model corresponding to just
N=3 non-linear parameters (a `bulge` comprising a linear Sersic would give N=6).

Therefore, not only does a shapelet fit more complex source galaxy morphologies, it does so using fewer non-linear
parameters than the standard light profile models!

__Disadvantages__

- Computationally slow.
- Assume single centre

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For shapelets, this produces a positive-negative "ringing", where the
Gaussians alternate between large positive and negative values. This is clearly undesirable and unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physical.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
This script fits an `Imaging` dataset of a galaxy with a model where:

 - The source galaxy's bulge is a super position of `ShapeletCartesianSph`` profiles.

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

Load and plot the galaxy dataset `light_basis` via .fits files, which we will fit with 
the model.
"""
dataset_name = "simple__no_lens_light"
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

We first show how to compose a basis of multiple shapelets and use them to fit the source galaxy's light in data

This is to illustrate the API for fitting shapelets using standard autolens objects like the `Galaxy`, `Tracer`
and `FitImaging`.

This does not perform a model-fit via a non-linear search, and therefore requires us to manually specify and guess
suitable parameter values for the shapelets. However, shapelets can do a reasonable even if we just guess 
sensible parameter values.

We are applying shapelets to reconstruct the source galaxy's light, which means we need an accurate mass model of the
lens galaxy. We use the true lens mass model from the simulator script to do this, noting that later in the example
we will infer the lens mass model using a non-linear search.

__Basis__

We first build a `Basis`, which is built from multiple linear light profiles (in this case, shapelets). 

Below, we make a `Basis` out of 10 elliptical shapelet linear light profiles which: 

 - All share the same centre and elliptical components.
 - The size of the Shapelet basis is controlled by a `beta` parameter, which is the same for all shapelet basis 
   functions.
 
Note that any linear light profile can be used to compose a Basis. This includes Gaussians, which are often used to r
epresent the light of elliptical galaxies (see `modeling/features/multi_gaussian_expansion.py`).
"""
total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = []

n_count = 1
m_count = -1

for i in range(total_n + total_m):
    shapelet = al.lp_linear.ShapeletPolarSph(
        n=n_count, m=m_count, centre=(0.01, 0.01), beta=0.1
    )

    shapelets_bulge_list.append(shapelet)

    m_count += 2

    if m_count > n_count:
        n_count += 1
        m_count = -n_count

bulge = af.Model(
    al.lp_basis.Basis,
    light_profile_list=shapelets_bulge_list,
)

"""
Once we have a `Basis`, we can treat it like any other light profile in order to create a `Galaxy` and `Tracer` and 
use it to fit data.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=bulge,
)

tracer = al.Tracer(galaxies=[lens, source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
By plotting the fit, we see that the `Basis` does a reasonable job at capturing the appearance of the source galaxy,
with only faint residuals visible where the lensed source is located.

This is despite the beta parameter of the shapelets being a complete guess and not the optimal value for fitting the
source galaxy's light. 
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
Nevertheless, there are still residuals, which we now rectify by fitting the shapelets in a non-linear search, 
simultaneously fitting the lens's mass and source galaxies.

__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 - The source galaxy's bulge is a superposition of 10 parametric linear `ShapeletCartesianSph` profiles [3 parameters]. 
 - The centres of the Shapelets are all linked together.
 - The size of the Shapelet basis is controlled by a `beta` parameter, which is the same for all Shapelet basis 
   functions.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

Note how this Shapelet model can capture features more complex than a Sersic, but has fewer non-linear parameters
(N=3 compared to N=7 for a `Sersic`).

__Model Cookbook__

A full description of model composition, including lens model customization, is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = af.Collection(
    af.Model(al.lp_linear.ShapeletPolarSph) for _ in range(total_n + total_m)
)

n_count = 1
m_count = -1

for i, shapelet in enumerate(shapelets_bulge_list):
    shapelet.n = n_count
    shapelet.m = m_count

    m_count += 2

    if m_count > n_count:
        n_count += 1
        m_count = -n_count

    shapelet.centre = shapelets_bulge_list[0].centre
    shapelet.beta = shapelets_bulge_list[0].beta

bulge = af.Model(
    al.lp_basis.Basis,
    light_profile_list=shapelets_bulge_list,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the source galaxy is made of many `ShapeletCartesianSph` profiles.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="shapelets",
    unique_tag=dataset_name,
    n_live=150,
    #    force_x1_cpu=True,
    number_of_cores=4,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    settings_inversion=al.SettingsInversion(
        use_w_tilde=False, use_positive_only_solver=False
    ),
)

"""
__Run Time__

The likelihood evaluation time for a shapelets is significantly slower than standard light profiles.
This is because the image of every shapelets must be computed and evaluated, and each must be blurred with the PSF.
In this example, the evaluation time is ~0.37s, compared to ~0.01 seconds for standard light profiles.

Gains in the overall run-time however are made thanks to the models reduced complexity and lower
number of free parameters. The source is modeled with 3 free parameters, compared to 6+ for a linear light profile 
Sersic.

However, the multi-gaussian expansion (MGE) approach is even faster than shapelets. It uses fewer Gaussian basis
functions (speed up the likelihood evaluation) and has fewer free parameters (speeding up the non-linear search).
Furthermore, non of the free parameters scale the size of the source galaxy, which means the non-linear search
can converge faster.

I recommend you try using an MGE approach alongside shapelets. For many science cases, the MGE approach will be
faster and give higher quality results. Shapelets may perform better for irregular sources, but this is not
guaranteed.
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

Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=result.grid
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.cornerplot()

"""
__Lens Shapelets__

A model where the lens is modeled as shapelets can be composed and fitted as shown below.

I have not seen this model used in the literature, and am not clear on its advantages over a standard light profile
model. However, it is worth trying if you are fitting a lens galaxy with a complex morphology.

For most massive early-type galaxies, an MGE model will be faster and give higher quality results.
"""
# Lens:

bulge = af.Model(
    al.lp_basis.Basis,
    light_profile_list=shapelets_bulge_list,
)

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Source:

bulge = af.Model(
    al.lp_basis.Basis,
    light_profile_list=shapelets_bulge_list,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **Pyautolens**, which 
includes a dedicated tutorial for linear objects like basis functions.

__Regularization__

There is one downside to `Basis` functions, we may compose a model with too much freedom. The `Basis` (e.g. our 20
Gaussians) may overfit noise in the data, or possible the lensed source galaxy emission -- neither of which we 
want to happen! 

To circumvent this issue, we have the option of adding regularization to a `Basis`. Regularization penalizes
solutions which are not smooth -- it is essentially a prior that says we expect the component the `Basis` represents
(e.g. a bulge or disk) to be smooth, in that its light changes smoothly as a function of radius.

Below, we compose and fit a model using Basis functions which includes regularization, which adds one addition 
parameter to the fit, the `coefficient`, which controls the degree of smoothing applied.
"""
bulge = af.Model(
    al.lp_basis.Basis,
    light_profile_list=shapelets_bulge_list,
    regularization=al.reg.Constant,
)
galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model, which has addition priors now associated with regularization.
"""
print(model.info)

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="shapelets_regularized",
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
To learn more about Basis functions, regularization and when you should use them, checkout the 
following **HowToGalaxy** tutorials:

 - `howtogalaxy/chapter_2_lens_modeling/tutorial_5_linear_profiles.ipynb`.
 - `howtogalaxy/chapter_4_pixelizations/tutorial_4_bayesian_regularization.ipynb.
"""
