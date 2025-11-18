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

 - The lens galaxy's light is a linear `Sersic` bulge.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear `Sersic`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.

__Notes__

This script is identical to `modeling/start_here.py` except that the light profiles are switched to linear light
profiles.
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
__Dataset__

Load and plot the strong lens dataset `simple` via .fits files.
"""
dataset_name = "simple"
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
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
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
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies, a tracer) cannot be plotted because they do not 
have an `intensity` value.

Therefore, the objects created above which replaces all linear light profiles with ordinary light profiles must be
used for visualization:
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=dataset.grid)
tracer_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=tracer.galaxies[0], grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Wrap Up__

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results in **PyAutoLens**.

In particular, checkout the results example `linear.py` which details how to extract all information about linear
light profiles from a fit.
"""
