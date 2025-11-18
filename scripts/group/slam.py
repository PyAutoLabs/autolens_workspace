"""
SLaM (Source, Light and Mass): Extra Galaxies
=============================================

This example shows how to use the Source, Light and Mass (SLaM) automated lens modeling pipelines to fit a group-scale
strong lens, which includes extra galaxies surrounding the main lens whose light and mass are both included in the
lens model.

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

The `data_preparation` tutorial `autolens_workspace/*/imaging/data_preparation/examples/optional/extra_galaxies_centres.py`
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

If any code in this script is unclear, refer to the `guides/modeling/slam_start_here.ipynb` notebook.
"""

#
# from autoconf import jax_wrapper # Ensures JAX environment variables are set before other imports

# %matplotlib inline
# # from pyprojroot import here
# # workspace_path = str(here())
# # %cd $workspace_path
# # print(f"Working Directory has been set to `{workspace_path}`")
#
# import numpy as np
# import os
# import sys
# from pathlib import Path
# import autofit as af
# import autolens as al
# import autolens.plot as aplt
#
# # from autolens_workspace.scripts.advanced.chaining.slam.examples.extra_galaxies import source_pix_result_1
#
# # from autolens_workspace.scripts.guides.advanced.scaling_relation import extra_galaxy_centre
#
# sys.path.insert(0, os.getcwd())
# import slam_pipeline
# from autofit.aggregator.aggregator import Aggregator
#
#
# """
# __Dataset__
#
# Load, plot and mask the `Imaging` data.
# """
# dataset_name = "A_Obvious_COSJ095953+023319"
# dataset_path = Path(
#     "/mnt",
#     "ral",
#     "c4072114",
#     "PyAuto",
#     "group",
#     "dataset",
#     "imaging",
#     "groups",
#     dataset_name,
#     "F444W",
# )  # , "F444W"
#
# dataset = al.Imaging.from_fits(
#     data_path=dataset_path / "data.fits",
#     noise_map_path=dataset_path / "noise_map.fits",
#     psf_path=dataset_path / "psf.fits",
#     pixel_scales=0.06,
# )
#
# """
# __Extra Galaxies Centres and positions__
# """
# inner_extra_galaxies_centres = al.Grid2DIrregular(
#     al.from_json(file_path=Path(dataset_path, "inner_extra_galaxies_centres.json"))
# )
# outer_extra_galaxies_centres = al.Grid2DIrregular(
#     al.from_json(file_path=Path(dataset_path, "outer_extra_galaxies_centres.json"))
# )
# extra_galaxies_centres = np.vstack(
#     (inner_extra_galaxies_centres, outer_extra_galaxies_centres)
# )
#
# positions = al.Grid2DIrregular(
#     al.from_json(
#         file_path=Path(dataset_path, "..", "positions.json")
#     )  #'..' in between for A_obvious
# )
#
# """
# __Masks__
# """
# mask_radius = 3.8
# pixel_scales = 0.06
#
# """
# __Calculate the extra galaxies maximum distance to the centre to create a larger mask
# """
# dist_extra_galaxies = np.sqrt(
#     inner_extra_galaxies_centres[:, 0] ** 2 + inner_extra_galaxies_centres[:, 1] ** 2
# )
# dist_extra_galaxies = np.append(
#     dist_extra_galaxies,
#     np.sqrt(
#         outer_extra_galaxies_centres[:, 0] ** 2
#         + outer_extra_galaxies_centres[:, 1] ** 2
#     ),
# )
#
# mask_radius_larger = max(mask_radius, dist_extra_galaxies.max() + 0.5)
#
# mask_larger = al.Mask2D.circular(
#     shape_native=dataset.shape_native,
#     pixel_scales=dataset.pixel_scales,
#     radius=mask_radius_larger,
# )
#
# dataset_larger = dataset.apply_mask(mask=mask_larger)
#
# mask = al.Mask2D.circular(
#     shape_native=dataset.shape_native,
#     pixel_scales=dataset.pixel_scales,
#     radius=mask_radius,
# )
#
# dataset = dataset.apply_mask(mask=mask)
#
# dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
# dataset_plotter.subplot_dataset()
#
#
# """
# __Settings AutoFit__
#
# The settings of autofit, which controls the output paths, parallelization, database use, etc.
# """
# settings_search = af.SettingsSearch(
#     path_prefix=Path("imaging") / "slam",
#     unique_tag="A_Obvious_COSJ095953+023319_scal_rel",
#     info=None,
#     session=None,
# )
#
# """
# __Redshifts__
#
# The redshifts of the lens and source galaxies.
# """
# redshift_lens = 0.890
# redshift_source = 4.9923
#
# """
# __SOURCE LP PIPELINE__
#
# The SOURCE LP PIPELINE is identical to the `start_here.ipynb` example, except the `extra_galaxies` are included in the
# model.
# """
# positions_likelihood = al.PositionsLH(positions=positions, threshold=0.1)
#
# analysis = al.AnalysisImaging(
#     dataset=dataset, positions_likelihood_list=[positions_likelihood]
# )
#
# # Lens Light
#
# lens_bulge = al.model_util.mge_model_from(
#     mask_radius=mask_radius,
#     total_gaussians=30,
#     gaussian_per_basis=2,
#     centre_prior_is_uniform=True,
# )
#
# # Lens Mass
#
# lens_mass = af.Model(al.mp.Isothermal)
# lens_mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)
#
# # Source Light
#
# source_bulge = al.model_util.mge_model_from(
#     mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
# )
#
# # Extra Galaxies:
#
# extra_galaxies_list = []
#
# for extra_galaxy_centre in extra_galaxies_centres:
#     # Extra Galaxy Light
#
#     total_gaussians = 10
#
#     log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)
#
#     extra_galaxy_gaussian_list = []
#
#     gaussian_list = af.Collection(
#         af.Model(al.lp_linear.GaussianSph) for _ in range(total_gaussians)
#     )
#
#     for i, gaussian in enumerate(gaussian_list):
#         gaussian.centre.centre_0 = extra_galaxy_centre[0]
#         gaussian.centre.centre_1 = extra_galaxy_centre[1]
#         gaussian.sigma = 10 ** log10_sigma_list[i]
#
#     extra_galaxy_gaussian_list += gaussian_list
#
#     extra_galaxy_bulge = af.Model(
#         al.lp_basis.Basis, profile_list=extra_galaxy_gaussian_list
#     )
#
#     # Extra Galaxy Mass
#
#     mass = af.Model(al.mp.IsothermalSph)
#
#     mass.centre = extra_galaxy_centre
#     mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
#
#     extra_galaxy = af.Model(
#         al.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge, mass=mass
#     )
#
#     extra_galaxy.mass.centre = extra_galaxy_centre
#
#     extra_galaxies_list.append(extra_galaxy)
#
# extra_galaxies = af.Collection(extra_galaxies_list)
#
# source_lp_result_1 = slam_pipeline.source_lp.run_1(
#     settings_search=settings_search,
#     analysis=analysis,
#     lens_bulge=lens_bulge,
#     lens_disk=None,
#     mass=lens_mass,
#     shear=af.Model(al.mp.ExternalShear),
#     source_bulge=source_bulge,
#     extra_galaxies=extra_galaxies,
#     mass_centre=(0.0, 0.0),
#     redshift_lens=redshift_lens,
#     redshift_source=redshift_source,
# )
#
# extra_galaxies_list = []
#
# for i, extra_galaxy_centre in enumerate(extra_galaxies_centres):
#     # Extra Galaxy Light
#
#     total_gaussians = 10
#
#     log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)
#
#     centre_0 = af.GaussianPrior(mean=extra_galaxy_centre[0], sigma=0.1)
#     centre_1 = af.GaussianPrior(mean=extra_galaxy_centre[1], sigma=0.1)
#
#     extra_galaxy_gaussian_list = []
#
#     gaussian_list = af.Collection(
#         af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
#     )
#     for j, gaussian in enumerate(gaussian_list):
#         gaussian.centre.centre_0 = centre_0
#         gaussian.centre.centre_1 = centre_1
#         gaussian.ell_comps = gaussian_list[0].ell_comps
#         gaussian.sigma = 10 ** log10_sigma_list[j]
#
#     extra_galaxy_gaussian_list += gaussian_list
#
#     extra_galaxy_bulge = af.Model(
#         al.lp_basis.Basis, profile_list=extra_galaxy_gaussian_list
#     )
#
#     # Extra Galaxy Mass
#
#     mass = source_lp_result_1.instance.extra_galaxies[i].mass
#     mass.centre = extra_galaxy_centre
#
#     extra_galaxy = af.Model(
#         al.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge, mass=mass
#     )
#
#     extra_galaxies_list.append(extra_galaxy)
#
#
# extra_galaxies_free_centres = af.Collection(extra_galaxies_list)
#
# analysis = al.AnalysisImaging(
#     dataset=dataset_larger, positions_likelihood_list=[positions_likelihood]
# )
#
# source_lp_result_2 = slam_pipeline.source_lp.run_2(
#     settings_search=settings_search,
#     analysis=analysis,
#     source_lp_result=source_lp_result_1,
#     extra_galaxies=extra_galaxies_free_centres,
# )
#
#
# """
# __SOURCE PIX PIPELINE__
#
# The SOURCE PIX PIPELINE (and every pipeline that follows) are identical to the `start_here.ipynb` example.
#
# The model components for the extra galaxies (e.g. `lens_bulge` and `lens_disk`) are passed from the SOURCE LP PIPELINE,
# via the `source_lp_result` object, therefore you do not need to manually pass them below.
# """
# analysis = al.AnalysisImaging(
#     dataset=dataset,
#     adapt_image_maker=al.AdaptImageMaker(result=source_lp_result_1),
#     # positions_likelihood_list=[source_lp_result_2.positions_likelihood_from(
#     #   factor=3.0, minimum_threshold=0.2
#     # )],
#     positions_likelihood_list=[positions_likelihood],
# )
#
# extra_galaxies_fixed_list = []
#
# scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
# scaling_relation = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
#
# for i in range(len(extra_galaxies_centres)):
#     extra_galaxy_bulge = source_lp_result_2.instance.extra_galaxies[i].bulge
#
#     Luminosity_per_gaussian_list = []
#
#     for (
#         gaussian
#     ) in source_lp_result_1.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles.galaxies[
#         i + 2
#     ].bulge.profile_list:
#         q = gaussian.axis_ratio
#
#         Luminosity = 2 * np.pi * gaussian.sigma**2 / q * gaussian.intensity
#         Luminosity_per_gaussian_list.append(Luminosity)
#
#     total_luminosity = np.sum(Luminosity_per_gaussian_list) / pixel_scales**2
#
#     mass = af.Model(al.mp.Isothermal)
#     mass.einstein_radius = scaling_factor * total_luminosity**scaling_relation
#     mass.centre = source_lp_result_2.instance.extra_galaxies[i].bulge.centre
#     mass.ell_comps = source_lp_result_2.instance.extra_galaxies[i].bulge.ell_comps
#
#     extra_galaxy = af.Model(
#         al.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge, mass=mass
#     )
#     extra_galaxies_fixed_list.append(extra_galaxy)
#
# extra_galaxies_fixed_centres = af.Collection(extra_galaxies_fixed_list)
#
# source_pix_result_1 = slam_pipeline.source_pix.run_1(
#     settings_search=settings_search,
#     analysis=analysis,
#     source_lp_result=source_lp_result_1,
#     extra_galaxies=extra_galaxies_fixed_centres,
#     mesh_init=al.mesh.Voronoi,
# )
#
# """
# __SOURCE PIX PIPELINE 2 (with lens light)__
#
# As above, this pipeline also has the same API as the `start_here.ipynb` example.
#
# The extra galaxies are passed from the SOURCE PIX PIPELINE, via the `source_pix_result_1` object, therefore you do not
# need to manually pass them below.
# """
# analysis = al.AnalysisImaging(
#     dataset=dataset,
#     adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
#     # positions_likelihood_list=[source_pix_result_1.positions_likelihood_from(
#     #    factor=3.0, minimum_threshold=0.2),
#     positions_likelihood_list=[positions_likelihood],
#     settings_inversion=al.SettingsInversion(
#         image_mesh_min_mesh_pixels_per_pixel=3,
#         image_mesh_min_mesh_number=5,
#         image_mesh_adapt_background_percent_threshold=0.1,
#         image_mesh_adapt_background_percent_check=0.8,
#     ),
# )
#
# # extra_galaxies = source_pix_result_1.instance.extra_galaxies
#
# source_pix_result_2 = slam_pipeline.source_pix.run_2(
#     settings_search=settings_search,
#     analysis=analysis,
#     source_lp_result=source_lp_result_1,
#     source_pix_result_1=source_pix_result_1,
#     extra_galaxies=source_pix_result_1.instance.extra_galaxies,
#     image_mesh=al.image_mesh.Hilbert,
#     mesh=al.mesh.Voronoi,
#     regularization=al.reg.AdaptiveBrightnessSplit,
# )
#
# """
# __LIGHT LP PIPELINE__
#
# As above, this pipeline also has the same API as the `start_here.ipynb` example.
#
# The extra galaxies are passed from the SOURCE PIX PIPELINE, via the `source_pix_result_1` object, therefore you do not
# need to manually pass them below.
# """
#
# analysis = al.AnalysisImaging(
#     dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1)
# )
#
# lens_bulge = al.model_util.mge_model_from(
#     mask_radius=mask_radius,
#     total_gaussians=30,
#     gaussian_per_basis=2,
#     centre_prior_is_uniform=True,
# )
#
# # Inner extra galaxies
#
# extra_galaxies_list = []
#
# for i, extra_galaxy_centre in enumerate(inner_extra_galaxies_centres):
#     # Extra Galaxy Light
#
#     total_gaussians = 10
#
#     log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)
#
#     centre_0 = af.GaussianPrior(mean=extra_galaxy_centre[0], sigma=0.1)
#     centre_1 = af.GaussianPrior(mean=extra_galaxy_centre[1], sigma=0.1)
#
#     extra_galaxy_gaussian_list = []
#
#     gaussian_list = af.Collection(
#         af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
#     )
#
#     for j, gaussian in enumerate(gaussian_list):
#         gaussian.centre.centre_0 = centre_0
#         gaussian.centre.centre_1 = centre_1
#         gaussian.ell_comps = gaussian_list[0].ell_comps
#         gaussian.sigma = 10 ** log10_sigma_list[j]
#
#     extra_galaxy_gaussian_list += gaussian_list
#
#     extra_galaxy_bulge = af.Model(
#         al.lp_basis.Basis, profile_list=extra_galaxy_gaussian_list
#     )
#
#     # Extra Galaxy Mass
#
#     mass = source_pix_result_1.instance.extra_galaxies[i].mass
#
#     extra_galaxy = af.Model(
#         al.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge, mass=mass
#     )
#
#     extra_galaxies_list.append(extra_galaxy)
#
# # Outer extra galaxies
#
# for i, extra_galaxy_centre in enumerate(outer_extra_galaxies_centres):
#
#     # Extra Galaxy Light
#     bulge = source_pix_result_1.instance.extra_galaxies[
#         i + len(inner_extra_galaxies_centres)
#     ].bulge
#
#     # Extra Galaxy Mass
#     mass = source_pix_result_1.instance.extra_galaxies[
#         i + len(inner_extra_galaxies_centres)
#     ].mass
#
#     extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
#
#     extra_galaxies_list.append(extra_galaxy)
#
# extra_galaxies = af.Collection(extra_galaxies_list)
#
# light_result_1 = slam_pipeline.light_lp.run_1(
#     settings_search=settings_search,
#     analysis=analysis,
#     source_result_for_lens=source_pix_result_1,
#     source_result_for_source=source_pix_result_2,
#     extra_galaxies=extra_galaxies,
#     lens_bulge=lens_bulge,
#     lens_disk=None,
# )
#
# """
# __MASS TOTAL PIPELINE__
#
# As above, this pipeline also has the same API as the `start_here.ipynb` example.
#
# The extra galaxies are passed from the SOURCE PIX PIPELINE, via the `source_pix_result_1` object, therefore you do not
# need to manually pass them below.
# """
# analysis = al.AnalysisImaging(
#     dataset=dataset,
#     adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
#     # positions_likelihood_list=[source_pix_result_2.positions_likelihood_from(
#     #    factor=3.0, minimum_threshold=0.2
#     # )],
#     positions_likelihood_list=[positions_likelihood],
# )
#
# # Extra galaxies
# # extra_galaxies = source_lp_result.model.extra_galaxies
# # for galaxy, result_galaxy in zip(extra_galaxies, light_result.instance.extra_galaxies):
# #   galaxy.bulge = result_galaxy.bulge
#
# extra_galaxies_list = []
#
# for i, extra_galaxy_centre in enumerate(extra_galaxies_centres):
#     # Extra Galaxy Light
#
#     extra_galaxy_bulge = light_result_1.instance.extra_galaxies[i].bulge
#
#     # Extra Galaxy Mass
#     # mass = af.Model(al.mp.IsothermalSph)
#     # mass = source_pix_result_1.model.extra_galaxies[i].mass
#
#     # mass.centre = light_result_2.instance.extra_galaxies[i].bulge.centre
#     # mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
#
#     Luminosity_per_gaussian_list = []
#
#     for (
#         gaussian
#     ) in light_result_1.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles.galaxies[
#         i + 2
#     ].bulge.profile_list:
#         q = gaussian.axis_ratio
#
#         Luminosity = 2 * np.pi * gaussian.sigma**2 / q * gaussian.intensity
#         Luminosity_per_gaussian_list.append(Luminosity)
#
#     total_luminosity = np.sum(Luminosity_per_gaussian_list) / pixel_scales**2
#
#     mass = af.Model(al.mp.Isothermal)
#     mass.einstein_radius = scaling_factor * total_luminosity**scaling_relation
#     mass.centre = light_result_1.instance.extra_galaxies[i].bulge.centre
#     mass.ell_comps = light_result_1.instance.extra_galaxies[i].bulge.ell_comps
#
#     extra_galaxy = af.Model(
#         al.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge, mass=mass
#     )
#
#     extra_galaxies_list.append(extra_galaxy)
#
# extra_galaxies = af.Collection(extra_galaxies_list)
#
# mass_result = slam_pipeline.mass_total.run(
#     settings_search=settings_search,
#     analysis=analysis,
#     source_result_for_lens=source_pix_result_1,
#     source_result_for_source=source_pix_result_2,
#     light_result=light_result_1,
#     mass=af.Model(al.mp.PowerLaw),
#     extra_galaxies=extra_galaxies,
# )
#
# """
# __Output__
#
# The `start_hre.ipynb` example describes how results can be output to hard-disk after the SLaM pipelines have been run.
# Checkout that script for a complete description of the output of this script.
# """
