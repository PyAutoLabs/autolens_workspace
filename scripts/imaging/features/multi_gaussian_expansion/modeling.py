"""
Modeling Features: Multi Gaussian Expansion
===========================================

A multi Gaussian expansion (MGE) decomposes the lens light into ~15-100 Gaussians, where the `intensity` of every
Gaussian is solved for via a linear algebra using a process called an "inversion" (see the `linear_light_profiles.py`
feature for a full description of this).

This script performs lensing modeling using a lens light model which uses an MGE consisting of 60 Gaussians. It is
fitted to simulated data where the lens galaxy's light has asymmetric and irregular features, which fitted poorly by symmetric light
profiles like the `Sersic`.

__Contents__

- **Advantages & Disadvantages:** Symmetric light profiles (e.g.
- **Positive Only Solver:** Ensuring positive-only solutions for linear light profile intensities.
- **MGE Source Galaxy:** The MGE was designed to model the light of lens galaxies, because they are typically elliptical.
- **Model:** Compose the lens model fitted to the data.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
- **Model Cookbook:** A full description of model composition is provided by the model cookbook.
- **Search:** Configure the non-linear search used to fit the model.
- **Analysis:** Create the Analysis object that defines how the model is fitted to the data.
- **VRAM:** The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the.
- **Run Time:** Profiling the expected run time of the model-fit.
- **Result:** Overview of the results of the model-fit.
- **Source MGE:** As discussed at the beginning of this tutorial, an MGE is an effective way to model the light of a.
- **Lens Point Source:** Using a compact MGE to model an unresolved point-like source (e.g. an AGN) in the foreground lens.
- **Wrap Up:** Summary of the script and next steps.
- **Description:** There is one downside to `Basis` functions, we may compose a model with too much freedom.

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

Many codes which use linear algebra typically rely on a linear algebra solver which allows for positive and negative
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
we recommend using a pixelized source reconstruction (see `autolens_workspace/scripts/imaging/features/pixelization/modeling.py`).
Combining this with an MGE for the len's light can be a very powerful way to model strong lenses!

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's bulge is a super position of 60 `Gaussian` profiles.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear `SersicCore`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `imaging/start_here.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the strong lens dataset `simple` via .fits files.
"""
dataset_name = "lens_light_asymmetric"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/imaging/features/multi_gaussian_expansion/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Over Sampling__

Apply adaptive over sampling to ensure the lens galaxy light calculation is accurate, you can read up on over-sampling 
in more detail via the `autolens_workspace/*/guides/advanced/over_sampling.ipynb` notebook.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

We compose a lens model where:

 - The galaxy's bulge is 60 linear `Gaussian` profiles [6 parameters]. 
 - The centres and elliptical components of the Gaussians are all linked together in two groups of 30.
 - The `sigma` size of the Gaussians increases in log10 increments.

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's light is a linear `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.

Unlike the examples above, our MGE now comprises 2 * 30 Gaussians (instead of 1 * 30). Each group of 30
have their own elliptical components meaning the lens's light is decomposed into two distinct elliptical components,
which could be viewed as a bulge and disk.

Note that above we combine the MGE for the lens light with a linear light profile for the source, meaning these two
categories of models can be combined.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
total_gaussians = 30
gaussian_per_basis = 2

# The sigma values of the Gaussians will be fixed to values spanning a tenth of the pixel scale to the mask radius, 3.0".
mask_radius = 3.0
log10_sigma_list = np.linspace(
    np.log10(dataset.pixel_scales[0] / 10.0), np.log10(mask_radius), total_gaussians
)

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

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

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

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start_here.py` for a
full description).

Owing to the simplicity of fitting an MGE we an use even fewer live points than other examples, reducing it to
75 live points, speeding up convergence of the non-linear search.
"""
search = af.Nautilus(
    path_prefix=Path("imaging") / "features",
    name="mge",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=2000000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM
required by a model.

For each linear Gaussian light profile, extra VRAM is used. For around 60 linear Gaussians this  typically requires 
a modest amount of VRAM (e.g. 10–50 MB per batched likelihood). Models that use hundreds of Gaussians, especially in 
combination with a large batch size, may therefore exceed GBs of VRAM and require you to adjust the batch size to fit 
within your GPU's VRAM.

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

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

In particular, checkout the results example `linear.py` which details how to extract all information about linear
light profiles from a fit.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

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

# The sigma values of the Gaussians will be fixed to values spanning a tenth of the pixel scale to the mask radius, 3.0".
mask_radius = 3.0
log10_sigma_list = np.linspace(
    np.log10(dataset.pixel_scales[0] / 10.0), np.log10(mask_radius), total_gaussians
)

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

centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

log10_sigma_list = np.linspace(-4, np.log10(1.0), total_gaussians)

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
    path_prefix=Path("imaging") / "features",
    name="mge_including_source",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Run Time__

The likelihood evaluation time for a multi-Gaussian expansion for both lens and source galaxies is not much slower
than when just the lens galaxy uses an MGE.

However, the overall run-time will be even faster than before, as treating the source as an MGE further
reduces the complexity of parameter space ensuring Nautilus converges even faster.

For initial model-fits where the lens model parameters are not known, a lens + source MGE is possibly the best
model one can use. 

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Lens Point Source__

The MGE is not only suited to a galaxy's extended stellar light -- it is also an effective way to model a compact,
unresolved point-like component at the centre of the foreground lens galaxy, for example a nuclear starburst, an
active galactic nucleus (AGN) or an unresolved bulge.

Such a component is modeled as a `Basis` of a small number of linear Gaussians (here 10) which all share the same
`centre` and elliptical components. Their `sigma` values are fixed to a set of logarithmically spaced values between
0.01" and twice the pixel scale of the data. This keeps the basis compact relative to the resolution of the image, so
that it represents a realistic PSF-convolved point source. Fixing the sigmas and linking the centre and ellipticity
across all Gaussians keeps the parameter count low: the only free parameters are the shared (y, x) `centre` (given a
+/- 0.1" uniform prior) and the two shared elliptical components, for N=4 in total.

The point-source MGE is added to the lens galaxy as an additional light component alongside the extended `bulge` MGE
composed above, so the lens galaxy's light becomes the sum of its diffuse stellar emission and its compact nuclear
source. We recreate the Gaussians line-by-line below so you can copy and paste the code into your own scripts.
"""
total_point_gaussians = 10

# Sigma values span a tenth of the pixel scale up to twice the pixel scale, keeping the basis compact and point-like.

log10_sigma_list = np.linspace(
    np.log10(dataset.pixel_scales[0] / 10.0),
    np.log10(2.0 * dataset.pixel_scales[0]),
    total_point_gaussians,
)

# The centre is the only free parameter, shared by every Gaussian and given a +/- 0.1" uniform prior.

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

point_gaussian_list = af.Collection(
    af.Model(al.lp_linear.Gaussian) for _ in range(total_point_gaussians)
)

for i, gaussian in enumerate(point_gaussian_list):
    gaussian.centre.centre_0 = centre_0  # All Gaussians share the same y centre.
    gaussian.centre.centre_1 = centre_1  # All Gaussians share the same x centre.
    gaussian.ell_comps = point_gaussian_list[
        0
    ].ell_comps  # All Gaussians share the same elliptical components.
    gaussian.sigma = 10 ** log10_sigma_list[i]  # Fixed, log-spaced sigma values.

# The Basis groups the Gaussians into a single compact point-source light component.

point = af.Model(al.lp_basis.Basis, profile_list=point_gaussian_list)

# The point-source MGE is added to the lens galaxy alongside its extended `bulge` MGE, `mass` and `shear`.

lens = af.Model(
    al.Galaxy, redshift=0.5, bulge=bulge, point=point, mass=mass, shear=shear
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
Printing the model info confirms the lens galaxy now has both an extended `bulge` MGE and a compact `point` MGE.
"""
print(model.info)

"""
Recreating the Gaussians line-by-line is useful for understanding how the point-source basis is composed, but in
practice it is more convenient to use the `mge_point_model_from` helper, which builds exactly the same compact basis
in a single line. It takes the data's `pixel_scales` (which sets the maximum Gaussian `sigma` and therefore how
compact the source is), the number of Gaussians and the source `centre`:
"""
point = al.model_util.mge_point_model_from(
    pixel_scales=dataset.pixel_scales[0],
    total_gaussians=10,
    centre=(0.0, 0.0),
    sigma_min=dataset.pixel_scales[0] / 10.0,
)

lens = af.Model(
    al.Galaxy, redshift=0.5, bulge=bulge, point=point, mass=mass, shear=shear
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__Wrap Up__

A Multi Gaussian Expansion is a powerful tool for modeling the light of galaxies, and offers a compelling method to
fit complex light profiles with a small number of parameters.

For **PyAutoLens**'s advanced search chaining feature, it is common use the MGE to initialize the lens light and source 
models. The lens light model is then made more complex by using an MGE with more Gaussians and the source becomes a 
pixelized reconstruction.

Now you are familiar with MGE modeling, it is recommended you adopt this as your default lens modeling approach.
However, it may not be suitable for lower resolution data, where the simpler Sersic profiles may be more appropriate.

__Basis Regularization (Advanced / Unused)__

An MGE `Basis` can additionally carry a regularization term (e.g. `al.reg.Constant`) that penalises non-smooth
solutions for the Gaussian intensities. This is a research-only feature: it is not used by any production
scientific analysis, because the positive-only linear algebra solver used above already solves the
"positive/negative ringing" problem regularization was originally intended to address.

The code and rationale for the regularization branch have been moved out of this user-facing script to keep it
focused on the supported MGE workflow. If you want to experiment with adding a regularization to a Basis, see:

    autolens_workspace_developer/basis_regularization/mge_lens.py

That script is self-contained and runs the same regularized fit that used to live here.
"""
