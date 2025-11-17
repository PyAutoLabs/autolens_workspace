"""
SLaM (Source, Light and Mass): Extra Galaxies
=============================================

This example shows how to use the SLaM pipelines to fit a lens where there are extra galaxies surrounding the main lens
galaxy, whose light and mass are both included in the lens model.

These systems likely constitute "group scale" lenses and therefore this script is the point where the galaxy-scale
SLaM pipelines can be adapted to group-scale lenses.

__Extra Galaxies Centres__

To set up a lens model including each extra galaxy with light and / or mass profile, we input manually the centres of
the extra galaxies.

In principle, a lens model including the extra galaxies could be composed without these centres. For example, if
there were two extra galaxies in the data, we could simply add two additional light and mass profiles into the model.
The modeling API does support this, but we will not use it in this example.

This is because models where the extra galaxies have free centres are often too complex to fit. It is likely the fit
will infer an inaccurate lens model and local maxima, because the parameter space is too complex.

For example, a common problem is that one of the extra galaxy light profiles intended to model a nearby galaxy instead
fit  one of the lensed source's multiple images. Alternatively, an extra galaxy's mass profile may recenter itself and
act as part of the main lens galaxy's mass distribution.

Therefore, when modeling extra galaxies we input the centre of each, in order to fix their light and mass profile
centres or set up priors centre around these values.

The `data_preparation` tutorial `autolens_workspace/*/data_preparation/imaging/examples/optional/extra_galaxies_centres.py`
describes how to create these centres. Using this script they have been output to the `.json` file we load below.

__Preqrequisites__

Before reading this script, you should have familiarity with the following key concepts:

- **Extra Galaxies**: How we include extra galaxies in the lens model, demonstrated in `features/extra_galaxies.ipynb`,
  as the exact same API is used here.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script  fits `Imaging` dataset  of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.
 - Two extra galaxies are included in the model, each with their light represented as a bulge with MGE light profile
   and their mass as a `IsothermalSph` profile.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/chaining.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
import sys
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
import slam_pipeline

"""
__Dataset__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

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
__Extra Galaxies Centres__
"""
extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "extra_galaxies_centres.json"))
)

print(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)] + extra_galaxies_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("imaging") / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__SOURCE LP PIPELINE__

The SOURCE LP PIPELINE is identical to the `start_here.ipynb` example, except the `extra_galaxies` are included in the
model.
"""
analysis = al.AnalysisImaging(dataset=dataset)

# Lens Light

centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

total_gaussians = 30
gaussian_per_basis = 2

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

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

lens_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

# Source Light

centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

total_gaussians = 30
gaussian_per_basis = 1

log10_sigma_list = np.linspace(-3, np.log10(1.0), total_gaussians)

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

# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:
    # Extra Galaxy Light

    total_gaussians = 10

    ### FUTURE IMPROVEMENT: Set the size based on each extra galaxy's size as opposed to the mask.

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    ### FUTURE IMPROVEMENT: Use elliptical Gaussians for the extra galaxies where the ellipticity is estimated beforehand.

    extra_galaxy_gaussian_list = []

    gaussian_list = af.Collection(
        af.Model(al.lp_linear.GaussianSph) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = extra_galaxy_centre[0]
        gaussian.centre.centre_1 = extra_galaxy_centre[1]
        gaussian.sigma = 10 ** log10_sigma_list[i]

    extra_galaxy_gaussian_list += gaussian_list

    extra_galaxy_bulge = af.Model(
        al.lp_basis.Basis, profile_list=extra_galaxy_gaussian_list
    )

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)

    extra_galaxy = af.Model(
        al.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge, mass=mass
    )

    extra_galaxy.mass.centre = extra_galaxy_centre

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
    extra_galaxies=extra_galaxies,
)

"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE (and every pipeline that follows) are identical to the `start_here.ipynb` example,
except the additional galaxies are passed to the pipeline.

The model components for the extra galaxies are set up using a trick with the model composition whereby all
extra galaxies used in the SOURCE LP PIPELINE are set up as a model, and the result is then used to fix their
light parameters to the results of the SOURCE LP PIPELINE.

This means that the extra galaxies model parameterization is identical to SOURCE LP PIPELINE, but the mass profile
priors are set using the results of the SOURCE LP PIPELINE.
"""
extra_galaxies = source_lp_result.model.extra_galaxies

for galaxy, result_galaxy in zip(
    extra_galaxies, source_lp_result.instance.extra_galaxies
):
    galaxy.bulge = result_galaxy.bulge

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.RectangularMagnification,
    extra_galaxies=extra_galaxies,
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__

As above, this pipeline also has the same API as the `start_here.ipynb` example.

The extra galaxies are passed from the SOURCE PIX PIPELINE, via the `source_pix_result_1` object, therefore there is 
no need to manually pass them below.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    settings_inversion=al.SettingsInversion(
        image_mesh_min_mesh_pixels_per_pixel=3,
        image_mesh_min_mesh_number=5,
        image_mesh_adapt_background_percent_threshold=0.1,
        image_mesh_adapt_background_percent_check=0.8,
    ),
)

source_pix_result_2 = slam_pipeline.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.RectangularMagnification,
    regularization=al.reg.AdaptiveBrightnessSplit,
)

"""
__LIGHT LP PIPELINE__

As above, this pipeline also has the same API as the `start_here.ipynb` example, except for the extra galaxies.

The extra galaxies use the same for loop trick used before the SOURCE PIX PIPELINE, however this now makes
the light profiles free parameters in the model and fixes their mass profiles to the results of the SOURCE PIX PIPELINE.
"""
analysis = al.AnalysisImaging(
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1)
)

centre_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
centre_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)

total_gaussians = 30
gaussian_per_basis = 2

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

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

lens_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

# EXTRA GALAXIES

extra_galaxies = source_lp_result.model.extra_galaxies

for galaxy, result_galaxy in zip(
    extra_galaxies, source_pix_result_1.instance.extra_galaxies
):
    galaxy.mass = result_galaxy.mass

light_result = slam_pipeline.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    lens_disk=None,
    extra_galaxies=extra_galaxies,
)

"""
__MASS TOTAL PIPELINE__

As above, this pipeline also has the same API as the `start_here.ipynb` example except for the extra galaxies.

The extra galaxies are set up using the same trick as the SOURCE PIX PIPELINE, .

The light profiles of the extra galaxies are fixed to the results of the LIGHT LP PIPELINE, meaning that the mass
profiles of the extra galaxies are free parameters in the model with their priors set using the results of the 
SOURCE PIPELINE.
"""
extra_galaxies = source_pix_result_1.model.extra_galaxies

for galaxy, result_galaxy in zip(extra_galaxies, light_result.instance.extra_galaxies):
    galaxy.bulge = result_galaxy.bulge

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

mass_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    extra_galaxies=extra_galaxies,
)

"""
__Output__

The `start_hre.ipynb` example describes how results can be output to hard-disk after the SLaM pipelines have been run.
Checkout that script for a complete description of the output of this script.
"""
