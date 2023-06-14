"""
SLaM (Source, Light and Mass): Light Parametric + Mass Total + Source Parametric
================================================================================

SLaM pipelines break the analysis of 'galaxy-scale' strong lenses down into multiple pipelines which focus on modeling
a specific aspect of the strong lens, first the Source, then the (lens) Light and finally the Mass. Each of these
pipelines has it own inputs which which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS LIGHT DARK PIPELINE.

Using a SOURCE LP PIPELINE, LIGHT PIPELINE and a MASS LIGHT DARK PIPELINE this SLaM script  fits `Imaging` dataset of
a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge with a parametric `Sersic` light profile.
 - The lens galaxy's stellar mass distribution is a bulge tied to the light model above.
 - The lens galaxy's dark matter mass distribution is modeled as a `NFWMCRLudlow`.
 - The source galaxy's light is a parametric `Inversion`.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `source_pixelization`
 `light_lp`
 `mass_total/mass_light_dark`

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

Load the `Imaging` data, define the `Mask2D` and plot them.
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
settings_autofit = af.SettingsSearch(
    path_prefix=path.join("imaging", "slam"),
    unique_tag=dataset_name,
    info=None,
    number_of_cores=1,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
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
__SOURCE LP PIPELINE (with lens light)__

The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:

 - Uses a parametric `Sersic` bulge for the lens galaxy's light.

 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

 __Settings__:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS LIGHT DARK 
 PIPELINE).
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
__SOURCE PIX PIPELINE (with lens light)__

The SOURCE PIX PIPELINE (with lens light) uses four searches to initialize a robust model for the `Inversion` 
that reconstructs the source galaxy's light. It begins by fitting a `DelaunayMagnification` mesh with `Constant` 
regularization, to set up the model and hyper images, and then:

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
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PIX PIPELINE.
In this example it:

 - Uses a parametric `Sersic` bulge [Do not use the results of the SOURCE LP PIPELINE to initialize priors].

 - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
 
"""
analysis = al.AnalysisImaging(dataset=dataset, adapt_result=source_pix_results.last)

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
__MASS LIGHT DARK PIPELINE (with lens light)__

The MASS LIGHT DARK PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of 
accuracy, using the source model of the SOURCE PIPELINE and the lens light model of the LIGHT LP PIPELINE to 
initialize the model priors . In this example it:

 - Uses a parametric `Sersic` bulge for the lens galaxy's light and its stellar mass [12 parameters: fixed from 
 LIGHT LP PIPELINE].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` whose centre is aligned with bulge of 
 the light and stellar mass model above [5 parameters].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the MASS 
 LIGHT DARK PIPELINE.
 
__Settings__:

 - adapt: We may be using hyper features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
 hyper dataset if required.

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_result=source_pix_results.last,
    positions_likelihood=source_pix_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

lens_bulge = af.Model(al.lmp.Sersic)
dark = af.Model(al.mp.NFWMCRLudlow)

dark.centre = lens_bulge.centre

mass_results = slam.mass_light_dark.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_pix_results,
    light_results=light_results,
    lens_bulge=lens_bulge,
    lens_disk=None,
    dark=dark,
)

"""
Finish.
"""
