# """
# Chaining: SIE to Power-law
# ==========================
#
# This script chains two searches to fit multiple `Imaging` datasets of a 'galaxy-scale' strong lens with
# a model where:
#
#  - The lens galaxy's light is omitted.
#  - The lens galaxy's total mass distribution is an `PowerLaw`.
#  - The source galaxy's light is a parametric `SersicCore`.
#
# The two searches break down as follows:
#
#  1) Models the lens galaxy's mass as an `Isothermal` and the source galaxy's light as an `Sersic`.
#  2) Models the lens galaxy's mass an an `PowerLaw` and the source galaxy's light as an `Sersic`.
#
# This is an advanced script and assumes previous knowledge of the core **PyAutoLens** API for search chaining. Thus,
# certain parts of code are not documented to ensure the script is concise.
#
# Checkout `imaging/chaining/sie_to_power_law.py` for a detailed description of this search chaining script.
# """
# # %matplotlib inline
# # from pyprojroot import here
# # workspace_path = str(here())
# # %cd $workspace_path
# # print(f"Working Directory has been set to `{workspace_path}`")
#
# from os import path
# import autofit as af
# import autolens as al
# import autolens.plot as aplt
#
# """
# __Colors__
#
# The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).
# """
# color_list = ["g", "r"]
#
# """
# __Pixel Scales__
# """
# pixel_scales_list = [0.08, 0.12]
#
# """
# __Dataset__
#
# Load and plot each multi-wavelength strong lens dataset, using a list of their waveband colors.
# """
# dataset_type = "multi"
# dataset_label = "imaging"
# dataset_name = "simple__no_lens_light"
#
# dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)
#
# dataset_list = [
#     al.Imaging.from_fits(
#         data_path=path.join(dataset_path, f"{color}_data.fits"),
#         psf_path=path.join(dataset_path, f"{color}_psf.fits"),
#         noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
#         pixel_scales=pixel_scales,
#     )
#     for color, pixel_scales in zip(color_list, pixel_scales_list)
# ]
#
# for dataset in dataset_list:
#
#     dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
#     dataset_plotter.subplot_dataset()
#
# """
# __Mask__
#
# Mask every multi-wavelength imaging dataset.
# """
# mask_list = [
#     al.Mask2D.circular(
#         shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
#     )
#     for dataset in dataset_list
# ]
#
#
# dataset_list = [
#     dataset.apply_mask(mask=mask)
#     for imaging, mask in zip(dataset_list, mask_list)
# ]
#
# for dataset in dataset_list:
#
#     dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
#     dataset_plotter.subplot_dataset()
#
# """
# __Paths__
#
# The path the results of all chained searches are output:
# """
# path_prefix = path.join("multi", "chaining", "sie_to_power_law")
#
# """
# __Analysis (Search 1)__
#
# We create an `Analysis` object for every dataset.
# """
# analysis_1_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]
# analysis_1 = sum(analysis_1_list)
# analysis_1.n_cores = 1
#
# """
# __Model (Search 1)__
#
# Search 1 fits a lens model where:
#
#  - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
#  - The source galaxy's light is a parametric `SersicCore`, where the `intensity` parameter of the source galaxy
#  for each individual waveband of imaging is a different free parameter [8 parameters].
#
# The number of free parameters and therefore the dimensionality of non-linear parameter space is N=15.
# """
# lens = af.Model(
#     al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
# )
# source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)
#
# model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source))
#
# """
# We now make the intensity a free parameter across every analysis object.
# """
# analysis_1 = analysis_1.with_free_parameters(model_1.galaxies.source.bulge.intensity)
#
# """
# __Search + Analysis + Model-Fit (Search 1)__
#
# We now create the non-linear search, analysis and perform the model-fit using this model.
# """
# search_1 = af.Nautilus(
#     path_prefix=path_prefix, name="search[1]__sie", unique_tag=dataset_name, n_live=100
# )
#
# result_1_list = search_1.fit(model=model_1, analysis=analysis_1)
#
# """
# __Analysis (Search 2)__
#
# We now create the list of analyses for search 2, including positions to avoid unphysica invdersion solutions.
# """
# analysis_2_list = [
#     al.AnalysisImaging(
#         dataset=dataset,
#         positions_likelihood=result_1_list[0].positions_likelihood_from(
#             factor=3.0, minimum_threshold=0.2
#         ),
#     )
#     for dataset in dataset_list
# ]
# analysis_2 = sum(analysis_2_list)
# analysis_2.n_cores = 1
#
# """
# __Model (Search 2)__
#
# We use the results of search 1 to create the lens model fitted in search 2, where:
#
#  - The lens galaxy's total mass distribution is an `PowerLaw` with `ExternalShear` [8 parameters: priors
#  initialized from search 1].
#  - The source galaxy's light is again a parametric `Sersic` where the `intensity` varies across wavelength
#   [8 parameters: priors initialized from search 1].
#
# The number of free parameters and therefore the dimensionality of non-linear parameter space is N=16.
#
# The overall model is accessible via the entire `result_list`, (e.g. `result_1_list.model`), which we can
# use to set up the lens galaxy via prior passing.
# """
# source = result_1_list.model.galaxies.source
#
# """
# However, we cannot use this to pass the lens galaxy, because its mass model must change from an `Isothermal`
# to an `PowerLaw`.
# """
# mass = af.Model(al.mp.PowerLaw)
# mass.take_attributes(result_1_list.model.galaxies.lens.mass)
# shear = result_1_list.model.galaxies.lens.shear
#
# lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
#
# model_2 = af.Collection(galaxies=af.Collection(lens=lens, source=source))
#
# """
# __Search + Analysis + Model-Fit (Search 2)__
#
# We now create the non-linear search, analysis and perform the model-fit using this model.
# """
# search_2 = af.Nautilus(
#     path_prefix=path_prefix,
#     name="search[2]__power_law",
#     unique_tag=dataset_name,
#     n_live=100,
# )
#
# result_2_list = search_2.fit(model=model_2, analysis=analysis_2)
#
# """
# __Wrap Up__
#
# In this example, we passed used prior passing to initialize a lens mass model as an `Isothermal` and
# passed its priors to then fit the more complex `PowerLaw` model.
#
# This removed difficult-to-fit degeneracies from the non-linear parameter space in search 1, providing a more robust
# and efficient model-fit.
#
# __Pipelines__
#
# Advanced search chaining uses `pipelines` that chain together multiple searches to perform complex lens modeling
# in a robust and efficient way.
#
# The following example pipelines fits a power-law, using the same approach demonstrated in this script of first
# fitting an `Isothermal`:
#
#  `autolens_workspace/imaging/chaining/pipelines/mass_total__source_lp.py`
#
# __SLaM (Source, Light and Mass)__
#
# An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens modeling
# processing into a series of fits that first perfect the source model, then the lens light model and finally the lens
# mass model.
#
# The SLaM pipelines assume an `Isothermal` throughout the Source and Light pipelines, and only switch to a
# more complex mass model (like the `PowerLaw`) in the final Mass pipeline.
# """
