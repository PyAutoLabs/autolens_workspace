"""
Modeling Features: Shapelets
============================

A shapelet is a basis function that is appropriate for capturing the exponential / disk-like features of a galaxy. It
has been employed in many strong lensing studies to model the light of the lensed source galaxy, because it can
represent features of disky star forming galaxies that a single Sersic function cannot.

- https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T
- https://iopscience.iop.org/article/10.1088/0004-637X/813/2/102 t

Shapelets are described in full in the following paper:

 https://arxiv.org/abs/astro-ph/0105178

This script performs a model-fit using shapelet, where it decomposes the galaxy light into ~20
Shapelets. The `intensity` of every Shapelet is solved for via linear algebra (see the `linear_light_profiles.py`
feature).

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using shapelets.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Basis:** How to create a basis of multiple light profiles, in this example shapelets.
**Coefficients:** A visualization of the real and imaginary shapelet coefficients in the Basis.
**Linear Light Profiles:** How to create a basis of linear light profiles to perform the shapelet decomposition.
**Fit:** Perform a fit to a dataset using linear light profile MGE.
**Intensities:** Access the solved for intensities of linear light profiles from the fit.
**Model:** Composing a model using shapelets and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Run Time:** Profiling of shapelet run times and discussion of how they compare to standard light profiles.
**Model-Fit:** Performs the model fit using standard API.
**Result:** Shaeplet results, including accessing light profiles with solved for intensity values.
**Cartesian Shapelets:** Using shapelets definedon a Cartesian coordinate system instead of polar coordinates.
**Lens Shapelets:** Using shapelets to decompose the lens galaxy instead of the source galaxy.
**Regularization:** API for applying regularization to shapelets, which is not recommend but included for illustration.

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals, because they fail to capture
irregular and asymmetric morphological of galaxies (e.g. isophotal twists, an ellipticity which varies radially).
Shapelets can capture some of these features and can therefore better represent the emission of complex source galaxies.

The shapelet model can be composed in a way that has fewer non-linear parameters than an elliptical Sersic. In this
example, the ~20 shapelets which represent the `bulge` of the source are composed in a model corresponding to just
N=3 non-linear parameters (a `bulge` comprising a linear Sersic would give N=6).

Therefore, shapelet fit more complex source galaxy morphologies using fewer non-linear parameters than the standard
light profile models!

__Disadvantages__

- There are many types of galaxy structure which shapelets may struggle to represent, such as a bar or assymetric
knots of star formation. They also rely on the galaxy have a distinct central over which the shapelets can be
centered, which is not the case of the galaxy is multiple merging systems or has bright companion galaxies.

- The linear algebra used to solve for the `intensity` of each shapelet has to allow for negative values of intensity
in order for shapelets to work. Negative surface brightnesses are unphysical, and are often inferred in a shapelet
decomposition, for example if the true galaxy has structure that cannot be captured by the shapelet basis. Other
approaches can force positive-only intensities on the solution, such as the Multi-Gaussian Expansion (MGE) or a pixelization.

- Computationally slower than standard light profiles like the Sersic.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
This script fits an `Imaging` dataset of a galaxy with a model where:

 - The source galaxy's bulge is a super position of `ShapeletCartesianSph`` profiles.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the galaxy dataset `light_basis` via .fits files, which we will fit with 
the model.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.4,
    outer_radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

Apply adaptive over sampling to ensure the lens galaxy light calculation is accurate, you can read up on over-sampling 
in more detail via the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Basis__

We first build a `Basis`, which is built from multiple light profiles (in this case, shapelets). 

Below, we make a `Basis` out of 20 elliptical polar shapelet light profiles which: 

 - All share the same centre and elliptical components.
 - The size of the Shapelet basis is controlled by a `beta` parameter, which is the same for all shapelet basis 
   functions.

Note that any light profile can be used to compose a Basis. This includes Gaussians, which are often used to 
represent the light of elliptical galaxies (see `modeling/features/multi_gaussian_expansion.py`).
"""
total_n = 5
total_m = sum(range(2, total_n + 1)) + 1

n_count = 1
m_count = -1

shapelets_bulge_list = []

shapelet_0 = al.lp.ShapeletPolar(
    n=0,
    m=0,
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=1.0,
    beta=1.0,
)

shapelets_bulge_list.append(shapelet_0)

for i in range(total_n + total_m):
    shapelet = al.lp.ShapeletPolarSph(
        n=n_count, m=m_count, centre=(0.0, 0.0), intensity=1.0, beta=1.0
    )

    shapelets_bulge_list.append(shapelet)

    m_count += 2

    if m_count > n_count:
        n_count += 1
        m_count = -n_count

bulge = al.lp_basis.Basis(profile_list=shapelets_bulge_list)

"""
__Coefficients__

The `Basis` is composed of many shapelets, each with different coefficients (n and m) values and a size parameter 
`beta`.

Each combination of coefficients creates shapelets with different radial and azimuthal features. They capture 
emission on different scales, with low coefficients corresponding to smooth features and high coefficients 
corresponding to more variable wave-like features. The size of the coefficients is determined by the input 
parameter `beta`, where larger values correspond to larger coefficients and therefore larger shapelets.

These coefficients are visualized below using a `BasisPlotter`.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

basis_plotter = aplt.BasisPlotter(basis=bulge, grid=grid)
basis_plotter.subplot_image()

"""
__Linear Light Profiles__

We now show Composing a basis of multiple shapelets and use them to fit the source galaxy's light in data.

This does not perform a model-fit via a non-linear search, and therefore requires us to manually specify and guess
suitable parameter values for the shapelets (e.g. the `centre`, `ell_comps`, `beta`). However, shapelets are
very flexible and will give us a decent looking source reconstruction even if we just guess sensible values
for each parameter. 

The one parameter that is tricky to guess is the `intensity` of each shapelet. A wide range of positive
and negative `intensity` values are required to decompose the source galaxy's light accurately. We certainly
cannot obtain a good solution by guessing the `intensity` values by eye.

We therefore use linear light profile shapelets, which determine the optimal value for each shapelet's `intensity` 
via linear algebra. Linear light profiles are described in the `linear_light_profiles.py` example and you should
familiarize yourself with this example before using shapelets.

We therefore again setup a `Basis` in an analogous fashion to the previous example, but this time we use linear
shapelets (via the `lp_linear.linear` module).
"""
total_n = 5
total_m = sum(range(2, total_n + 1)) + 1

n_count = 1
m_count = -1

shapelets_bulge_list = []

shapelet_0 = al.lp_linear.ShapeletPolar(
    n=0,
    m=0,
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    beta=1.0,
)

shapelets_bulge_list.append(shapelet_0)

for i in range(total_n + total_m):
    shapelet = al.lp_linear.ShapeletPolar(
        n=n_count,
        m=m_count,
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        beta=1.0,
    )

    shapelets_bulge_list.append(shapelet)

    m_count += 2

    if m_count > n_count:
        n_count += 1
        m_count = -n_count

bulge = al.lp_basis.Basis(profile_list=shapelets_bulge_list)

"""
__Fit__

We now illustrate the API for fitting shapelets using standard autolens objects like the `Galaxy`, `Tracer` 
and `FitImaging`.

Once we have a `Basis`, we can treat it like any other light profile in order to create a `Galaxy` and `Tracer` and 
use it to fit data.

We are applying shapelets to reconstruct the source galaxy's light, which means we need an accurate mass model of the
lens galaxy. We use the true lens mass model from the simulator script to do this, noting that later in the example
we will infer the lens mass model using a non-linear search.

__Positive Negative Solver__

In other examples which use linear algebra to fit the data, for example linear light profiles, the Multi Gaussian
Expansion (MGE) and pixelization, we use a `positive_only` solver, which forces all solved for intensities to be
positive. This is a physical and sensible approach, because the surface brightnesses of a galaxy cannot be negative.

Shapelets cannot be solved for using a `positive_only` solver, because the shapelets ability to decompose the
light of a galaxy relies on the ability to use negative intensities. This is because the shapelets are not
physically motivated light profiles, but instead a mathematical basis that can represent any light profile.

This means shapelets may include negative flux in the reconstructed source galaxy, which is unphysical, and
a disadvantage of using shapelets.

The `SettingsInversion` object below uses a `use_positive_only_solver=False` to allow for negative intensities.
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

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_positive_only_solver=False),
)

"""
By plotting the fit, we see that the shapelets do a reasonable job at capturing the appearance of the source galaxy,
with only faint residuals visible where the lensed source is located.

This is despite the beta parameter of the shapelets being a complete guess and not the optimal value for fitting the
source galaxy's light. 
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
We can use the `BasisPlotter` to plot each individual shapelet in the reconstructed basis.

This plot shows each shapelet has a unique `intensity` that was solved for via linear algebra.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

basis_plotter = aplt.BasisPlotter(basis=tracer.galaxies[1].bulge, grid=grid)
basis_plotter.subplot_image()

"""
__Intensities__

The fit contains the solved for intensity values.

These are computed using a fit's `linear_light_profile_intensity_dict`, which maps each linear light profile 
in the model parameterization above to its `intensity`.

The code below shows how to use this dictionary, as an alternative to using the max_log_likelihood quantities above.
"""
source_bulge = fit.tracer.galaxies[1].bulge

print(
    f"\n Intensity of source galaxy's first shapelet in bulge = {fit.linear_light_profile_intensity_dict[source_bulge.profile_list[0]]}"
)

"""
A `Tracer` where all linear light profile objects are replaced with ordinary light profiles using the solved 
for `intensity` values is also accessible from a fit.

For example, the first linear light profile of the shapelet `bulge` component above printed it solved for intensity 
value, but it was still represented as a linear light profile. 

The `tracer` created below instead has a standard light profile with an `intensity` actually set.

The benefit of using a tracer with standard light profiles is it can be visualized, as performed above (linear 
light profiles cannot by default because they do not have `intensity` values).
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

print(tracer.galaxies[1].bulge.profile_list[0].intensity)

"""
__Shapelet Cartesian__

The shapelets above were defined on a polar grid, which is suitable for modeling radially symmetric sources like
most galaxies.

An alternative approach is to define the shapelets on a Cartesian grid, which we plot the basis of below
and show an example fit.

These are generally not recommended for modeling galaxies, but may be better in certain situations.
"""
total_xy = 5

shapelets_bulge_list = []

for x in range(total_xy):
    for y in range(total_xy):
        shapelet = al.lp.ShapeletCartesian(
            n_y=y,
            n_x=x,
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.0),
            intensity=1.0,
            beta=1.0,
        )

        shapelets_bulge_list.append(shapelet)

bulge = al.lp_basis.Basis(profile_list=shapelets_bulge_list)

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

basis_plotter = aplt.BasisPlotter(basis=bulge, grid=grid)
basis_plotter.subplot_image()

"""
For fitting, we again use the linear light profile version of the Cartesian shapelets, which solves for the
optimal intensity of each shapelet via linear algebra.
"""
total_xy = 5

shapelets_bulge_list = []

for x in range(total_xy):
    for y in range(total_xy):
        shapelet = al.lp_linear.ShapeletCartesian(
            n_y=y, n_x=x, centre=(0.0, 0.0), ell_comps=(0.0, 0.0), beta=1.0
        )

        shapelets_bulge_list.append(shapelet)

bulge = al.lp_basis.Basis(profile_list=shapelets_bulge_list)

"""
__Fit__
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

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_positive_only_solver=False),
)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

tracer = fit.model_obj_linear_light_profiles_to_light_profiles

basis_plotter = aplt.BasisPlotter(basis=tracer.galaxies[1].bulge, grid=grid)
basis_plotter.subplot_image()

"""
__Wrap Up__

This script has illustrated how to use shapelets to model the light of galaxies.

Shapelets are a powerful basis function for capturing complex morphological features of galaxies that standard
light profiles struggle to represent. However, they do have drawbacks, such as the need to allow for negative
intensities in the solution, which is unphysical. 

As a rule of thumb, modeling is generally better if a pixelization is used to reconstruct the source galaxy's light,
but shapelets can be a useful middle-ground between standard light profiles and a pixelization.
"""
