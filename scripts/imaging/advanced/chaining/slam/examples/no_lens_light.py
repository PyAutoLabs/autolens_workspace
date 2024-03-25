"""
SLaM (Source, Light and Mass): No Lens Light
============================================

This example shows how to use the SLaM pipelines to fit a lens where the lens light is not present in the data.
This means that the LIGHT PIPELINE is omitted from the pipeline completely.

__Model__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE and a MASS TOTAL PIPELINE this SLaM script fits `Imaging`
of a strong lens system, where in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy is reconstructed using a `Hilbert` image-mesh, `Delaunay` mesh and `ConstantSplit` 
   regularization scheme.

This uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `mass_total`

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
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple__no_lens_light"
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

 - Uses a parametric `Sersic` bulge for the source's light.
 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
 
__Settings__:
 
 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the SOURCE INVERSION 
 PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=dataset)

source_lp_results = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=None,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.Sersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=0.5,
    redshift_source=1.0,
)

"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE uses two searches to initialize a robust model for the `Pixelization` that
reconstructs the source galaxy's light. 

The first search, which is an initialization search, fits an `Overlay` image-mesh, `Delaunay` mesh and `Constant` 
regularization. 

The second search, which uses the mesh and regularization used throughout the remainder of the SLaM pipelines,
fits the following model:

- Uses a `Hilbert` image-mesh. 
- Uses a `Delaunay` mesh.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_results.last),
    positions_likelihood=source_lp_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    settings_inversion=al.SettingsInversion(
        image_mesh_min_mesh_pixels_per_pixel=3,
        image_mesh_min_mesh_number=5,
        image_mesh_adapt_background_percent_threshold=0.1,
        image_mesh_adapt_background_percent_check=0.8,
    ),
)

source_pix_results = slam.source_pix.run(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_results=source_lp_results,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.Delaunay,
    regularization=al.reg.AdaptiveBrightnessSplit,
)

"""
__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize the model priors. 

In this example it:

 - Uses an `PowerLaw` model for the lens's total mass distribution [The centre if unfixed from (0.0, 0.0)].
 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINES through to the MASS 
 PIPELINE.
 
__Settings__:

 - adapt: We may be using adapt features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
 hyper dataset if required.

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_results[0]),
    positions_likelihood=source_pix_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

mass_results = slam.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_results=source_pix_results,
    light_results=None,
    mass=af.Model(al.mp.PowerLaw),
)

"""
Finish.
"""
