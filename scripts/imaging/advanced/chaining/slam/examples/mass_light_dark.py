"""
SLaM (Source, Light and Mass): Mass Light Dark
==============================================

This example shows how to use the SLaM pipelines to end with a mass model which decomposes the lens into its
stars and dark matter, using a light plus dark matter mass model.

Unlike other example SLaM pipelines, which end with the MASS TOTAL PIPELINE, this script ends with the
MASS LIGHT DARK PIPELINE.

__Model__

Using a SOURCE LP PIPELINE, LIGHT PIPELINE and a MASS LIGHT DARK PIPELINE this SLaM script  fits `Imaging` dataset of
a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge with a linear parametric `Sersic` light profile.
 - The lens galaxy's stellar mass distribution is a bulge tied to the light model above.
 - The lens galaxy's dark matter mass distribution is modeled as a `NFWMCRLudlow`.
 - The source galaxy's light is a `Pixelization`.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `source_pixelization`
 `light_lp`
 `mass_light_dark`

Check them out for a detailed description of the analysis!

__Start Here Notebook__

If any code in this script is unclear, refer to the `slam/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import sys
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
import slam

"""
__Dataset__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "mass_stellar_dark"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=path.join("imaging", "slam"),
    unique_tag=dataset_name,
    info=None,
    number_of_cores=1,
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

The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light, which in 
this example:

 - Uses a linear parametric `Sersic` bulge for the lens galaxy's light.

 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 __Settings__:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS LIGHT DARK 
 PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=dataset)

bulge = af.Model(al.lp_linear.Sersic)

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp_linear.SersicCore),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE uses two searches to initialize a robust model for the `Pixelization` that
reconstructs the source galaxy's light. 

This pixelization adapts its source pixels to the morphology of the source, placing more pixels in its 
brightest regions. To do this, an "adapt image" is required, which is the lens light subtracted image meaning
only the lensed source emission is present.

The SOURCE LP Pipeline result is not good enough quality to set up this adapt image (e.g. the source
may be more complex than a simple light profile). The first step of the SOURCE PIX PIPELINE therefore fits a new
model using a pixelization to create this adapt image.

The first search, which is an initialization search, fits an `Overlay` image-mesh, `Delaunay` mesh 
and `AdaptiveBrightnessSplit` regularization.

__Adapt Images / Image Mesh Settings__

If you are unclear what the `adapt_images` and `SettingsInversion` inputs are doing below, refer to the 
`autolens_workspace/*/imaging/advanced/chaining/pix_adapt/start_here.py` example script.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
    positions_likelihood=source_lp_result.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

source_pix_result_1 = slam.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Delaunay,
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__

The second search, which uses the mesh and regularization used throughout the remainder of the SLaM pipelines,
fits the following model:

- Uses a `Hilbert` image-mesh. 

- Uses a `Delaunay` mesh.

 - Uses an `AdaptiveBrightnessSplit` regularization.

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.

The `Hilbert` image-mesh and `AdaptiveBrightness` regularization adapt the source pixels and regularization weights
to the source's morphology.

Below, we therefore set up the adapt image using this result.
"""
adapt_image_maker = al.AdaptImageMaker(result=source_pix_result_1)
adapt_image = adapt_image_maker.adapt_images.galaxy_name_image_dict[
    "('galaxies', 'source')"
]

over_sampling = al.OverSamplingUniform.from_adapt(
    data=adapt_image,
    noise_map=dataset.noise_map,
)

dataset = dataset.apply_over_sampling(
    over_sampling=al.OverSamplingDataset(pixelization=over_sampling)
)

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

source_pix_result_2 = slam.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.Delaunay,
    regularization=al.reg.AdaptiveBrightnessSplit,
)


"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PIX PIPELINE.

In this example it:

 - Uses a linear parametric `Sersic` bulge [Fixed from SOURCE LP PIPELINE].

 - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].

 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values]. 
"""
bulge = af.Model(al.lp_linear.Sersic)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
)

light_result = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=bulge,
    lens_disk=None,
)

"""
__MASS LIGHT DARK PIPELINE__

The MASS LIGHT DARK PIPELINE uses one search to fits a complex lens mass model to a high level of 
accuracy, using the source model of the SOURCE PIPELINE and the lens light model of the LIGHT LP PIPELINE to 
initialize the model priors . 

In this example it:

 - Uses a linear parametric `Sersic` bulge for the lens galaxy's light and its stellar mass [11 parameters: fixed from 
 LIGHT LP PIPELINE].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` whose centre is aligned with bulge of 
 the light and stellar mass model above [5 parameters].

 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the MASS 
 LIGHT DARK PIPELINE.
 
__Settings__:

 - adapt: We may be using adapt features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
 hyper dataset if required.

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    positions_likelihood=source_pix_result_2.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

lp_chain_tracer = al.util.chaining.lp_chain_tracer_from(
    light_result=light_result, settings_search=settings_search
)

lens_bulge = af.Model(al.lmp.Sersic)
dark = af.Model(al.mp.NFWMCRLudlow)

dark.centre = lens_bulge.centre

mass_result = slam.mass_light_dark.run(
    settings_search=settings_search,
    analysis=analysis,
    lp_chain_tracer=lp_chain_tracer,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    dark=dark,
)

"""
Finish.
"""
