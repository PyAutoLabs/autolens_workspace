"""
SLaM (Source, Light and Mass): Source Pixelized
===============================================

Using two source pipelines, a light pipeline and a mass pipeline this SLaM modeling script  fits `Imaging` dataset 
of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with a parametric `Sersic` light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy's light is a `Pixelization`.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`

Check them out for a detailed description of the analysis!
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
from scripts.imaging.advanced.chaining.slam import slam

"""
__Dataset__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple__source_x2"
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
settings_autofit = af.SettingsSearch(
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
__Adapt Setup__

The `SetupAdapt` determines which hyper-mode features are used during the model-fit.
"""
setup_adapt = al.SetupAdapt(
    mesh_pixels_fixed=1500,
)

"""
__SOURCE LP PIPELINE__

The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light, which in 
this example:
 
 - Uses a parametric `Sersic` bulge for the lens galaxy's light.
 
 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 __Settings__:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=dataset)

bulge = af.Model(al.lp.Sersic)

source_lp_results = slam.source_lp.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    lens_bulge=bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.Sersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE uses two searches to initialize a robust model for the `Pixelization` that
reconstructs the source galaxy's light. 

The first search, which is an initialization search, fits a `DelaunayMagnification` mesh with `Constant` 
regularization. 

The second search, which uses the mesh and regularization used throughout the remainder of the SLaM pipelines,
fits the following model:

 - Uses a `DelaunayBrightnessImage` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_result=source_lp_results.last,
    positions_likelihood=source_lp_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

source_pix_results = slam.source_pix.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_lp_results=source_lp_results,
    mesh=al.mesh.DelaunayBrightnessImage,
    regularization=al.reg.AdaptiveBrightnessSplit,
)

"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
In this example it:

 - Uses a parametric `Sersic` bulge [Fixed from SOURCE LP PIPELINE].
 
 - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].
 
 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].
 
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].   
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_result=source_pix_results.last,
    positions_likelihood=source_pix_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

bulge = af.Model(al.lp.Sersic)

light_results = slam.light_lp.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_pix_results,
    lens_bulge=bulge,
    lens_disk=None,
)

"""
__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize the model priors and the lens 
light model of the LIGHT LP PIPELINE. 

In this example it:

 - Uses a parametric `Sersic` bulge [fixed from LIGHT LP PIPELINE].

 - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
 PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].
 
 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].
 
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS TOTAL 
 PIPELINE.

__Settings__:

 - adapt: We may be using adapt features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
 hyper dataset if required.

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_result=light_results.last,
    positions_likelihood=source_pix_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)


mass_results = slam.mass_total.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_pix_results,
    light_results=light_results,
    mass=af.Model(al.mp.PowerLaw),
)

"""
Finish.
"""
