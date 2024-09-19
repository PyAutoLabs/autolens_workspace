# """
# Chaining: Parametric To Pixelization
# ====================================
#
# This script chains two searches to fit multiple `Imaging` datasets of a 'galaxy-scale' strong lens with a model where:
#
#  - The lens galaxy's light is omitted.
#  - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
#  - The source galaxy's light is an `Exponential`.
#
# The two searches break down as follows:
#
#  1) Model the source galaxy using a parametric `Sersic` and lens galaxy mass as an `Isothermal`.
#  2) Models the source galaxy using an `Inversion` and lens galaxy mass as an `Isothermal`.
#
# This is an advanced script and assumes previous knowledge of the core **PyAutoLens** API for search chaining. Thus,
# certain parts of code are not documented to ensure the script is concise.
#
# Checkout `imaging/chaining/parametric_to_pixelization.py` for a detailed description of this search chaining script.
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
# path_prefix = path.join("multi", "chaining", "parametric_to_pixelization")
#
# """
# __Analysis (Search 1)__
#
# We create an `Analysis` object for every dataset.
# """
# analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]
# analysis = sum(analysis_list)
# analysis.n_cores = 1
#
# """
# __Model (Search 1)__
#
# Search 1 fits a lens model where:
#
#  - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
#  - The source galaxy's light is a parametric `SersicCore`, where the `intensity` parameter of the source galaxy
#  for each individual waveband of imaging is a different free parameter [8 parameters].
#
# The number of free parameters and therefore the dimensionality of non-linear parameter space is N=13.
# """
# lens = af.Model(
#     al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
# )
# source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)
#
# model = af.Collection(
#                     galaxies=af.Collection(lens=lens, source=source)
#                     )
#
# """
# We now make the intensity a free parameter across every analysis object.
# """
# analysis = analysis.with_free_parameters(model.galaxies.source.bulge.intensity)
#
# """
# __Search+ Model-Fit (Search 1)__
#
# We now create the non-linear search, analysis and perform the model-fit using this model.
# """
# search_1 = af.Nautilus(
#     path_prefix=path_prefix,
#     name="search[1]__parametric",
#     unique_tag=dataset_name,
#     n_live=100,
# )
#
# result_1_list = search_1.fit(model=model, analysis=analysis)
#
# """
# __Analysis + Positions (Search 2)__
#
# We now create the list of analyses for search 2, including positions to avoid unphysica invdersion solutions.
# """
# analysis_list = [
#     al.AnalysisImaging(
#         dataset=dataset,
#         positions_likelihood=result_1_list[0].positions_likelihood_from(
#             factor=3.0, minimum_threshold=0.2
#         ),
#     )
#     for dataset in dataset_list
# ]
# analysis = sum(analysis_list)
# analysis.n_cores = 1
#
# """
# __Model (Search 2)__
#
# We use the results of search 1 to create the lens model fitted in search 2, where:
#
#  - The lens galaxy's total mass distribution is again an `Isothermal` and `ExternalShear` [7 parameters:
#  priors initialized from search 1].
#  - The source galaxy's light uses an `Overlay` image mesh [2 parameters].
#  - The source-galaxy's light uses a `Delaunay` mesh [0 parameters].
#  - This pixelization is regularized using a `ConstantSplit` scheme which smooths every source pixel
#  equally, where its `regularization_coefficient` varies across the datasets [2 parameter].
#
# The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
#
# The overall model is accessible via the entire `result_list`, (e.g. `result_1_list.model`), which we can
# use to set up the lens galaxy via prior passing.
# """
# lens = result_1_list.model.galaxies.lens
# pixelization = af.Model(
#     al.Pixelization,
#     image_mesh=al.image_mesh.Overlay,
#     mesh=al.mesh.Delaunay,
#     regularization=al.reg.ConstantSplit,
# )
#
# source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
# model = af.Collection(
#                     galaxies=af.Collection(lens=lens, source=source)
#                     )
# #
#
# """
# We now make the regularization coefficient a free parameter across every analysis object.
# """
# analysis = analysis.with_free_parameters(
#     model.galaxies.source.pixelization.regularization.coefficient
# )
#
# """
# __Search + Model-Fit__
#
# We now create the non-linear search and perform the model-fit using this model.
# """
# search_2 = af.Nautilus(
#     path_prefix=path_prefix,
#     name="search[2]__pixelization",
#     unique_tag=dataset_name,
#     n_live=80,
# )
#
# result_2_list = search_2.fit(model=model, analysis=analysis)
#
# """
# __Wrap Up__
#
# In this example, we passed used prior passing to initialize an `Isothermal` + `ExternalShear` lens mass model
# using a parametric source and pass this model to a second search which modeled the source using an `Inversion`.
#
# This was more computationally efficient than just fitting the `Inversion` by itself and helped to ensure that the
# `Inversion` did not go to an unphysical mass model solution which reconstructs the source as a demagnified version
# of the lensed image.
#
# __Pipelines__
#
# Advanced search chaining uses `pipelines` that chain together multiple searches to perform complex lens modeling
# in a robust and efficient way.
#
# The following example pipelines fits an inversion, using the same approach demonstrated in this script of first fitting
# a parametric source:
#
#  `autolens_workspace/imaging/chaining/pipelines/mass_total__source_pixelization.py`
#
#  __SLaM (Source, Light and Mass)__
#
# An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens modeling
# processing into a series of fits that first perfect the source model, then the lens light model and finally the lens
# mass model.
#
# The SLaM pipelines begin with a parametric Source pipeline, which then switches to an inversion Source pipeline,
# exploiting the chaining technique demonstrated in this example.
# """
