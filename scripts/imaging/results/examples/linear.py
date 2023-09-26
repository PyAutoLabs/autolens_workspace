"""
Results 2: Linear
=================

This tutorial inspects an inferred model using linear light profiles which solve for the intensity via linear
algebra, in the form of both linear light profiles (via `lp_linear`) and a `Basis` of linear light profiles.

These objects mostly behave identically to ordinary light profiles, but due to the linear algebra have their own
specific functionality illustrated in this tutorial.

__Plot Module__

This example uses the **PyAutoLens** plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autolens_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.
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
__Model Fit__

The code below performs a model-fit using Nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!

Note that the model that is fitted has both a linear light profile `Sersic` and a `Basis` with 5 `Gaussian` 
profiles which is regularized.
"""
dataset_name = "lens_light_asymmetric"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

total_gaussians = 30
gaussian_per_basis = 2

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
mask_radius = 6.0
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

basis = af.Model(
    al.lp_basis.Basis,
    light_profile_list=bulge_gaussian_list,
    regularization=al.reg.ConstantZeroth(
        coefficient_neighbor=0.0, coefficient_zeroth=1.0
    ),
)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=basis, mass=al.mp.Isothermal)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.Sersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge_disk_linear]",
    unique_tag=dataset_name,
    n_live=100,
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
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
__Intensities__

The intensities of linear light profiles are not a part of the model parameterization and therefore are not displayed
in the `model.results` file.

To extract the `intensity` values of a specific component in the model, we use the `max_log_likelihood_tracer`,
which has already performed the inversion and therefore the galaxy light profiles have their solved for
`intensity`'s associated with them.
"""
tracer = result.max_log_likelihood_tracer

lens_bulge = print(tracer.galaxies[0].bulge.intensity)
source_bulge = print(tracer.galaxies[1].bulge.intensity)

"""
The `Tracer` contained in the `max_log_likelihood_fit` also has the solved for `intensity` values:
"""
fit = result.max_log_likelihood_fit

tracer = fit.tracer

lens_bulge = print(tracer.galaxies[0].bulge.intensity)
source_bulge = print(tracer.galaxies[1].bulge.intensity)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies, a tracer) cannot be plotted because they do not 
have an `intensity` value.

Therefore, the object created above which replaces all linear light profiles with ordinary light profiles must be
used for visualization:
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=dataset.grid)
tracer_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=tracer.galaxies[0], grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)


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
__Intensity Dict (Internal Source Code)__

The results above already allow you to compute the solved for intensity values.

These are computed using a fit's `linear_light_profile_intensity_dict`, which maps each linear light profile 
in the model parameterization above to its `intensity`.

The code below shows how to use this dictionary, as an alternative to using the max_log_likelihood quantities above.
"""
fit = result.max_log_likelihood_fit

print(
    f"\n Intensity of source bulge (lp_linear.Sersic) = {fit.linear_light_profile_intensity_dict[source_bulge]}"
)
print(
    f"\n Intensity of lens galaxy's first Gaussian in bulge = {fit.linear_light_profile_intensity_dict[lens_bulge.light_profile_list[0]]}"
)

"""
A `Tracer` where all linear light profile objects are replaced with ordinary light profiles using the solved 
for `intensity` values is also accessible from a fit.

For example, the linear light profile `Sersic` of the `bulge` component above has a solved for `intensity` of ~0.75. 

The `tracer` created below instead has an ordinary light profile with an `intensity` of ~0.75.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

print(tracer.galaxies[0].bulge.light_profile_list[0].intensity)
print(tracer.galaxies[1].bulge.intensity)

"""
Finish.
"""
