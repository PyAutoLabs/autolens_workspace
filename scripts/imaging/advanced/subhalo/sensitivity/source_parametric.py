"""
SLaM (Source, Light and Mass): Subhalo Source Parametric Sensitivity Mapping
============================================================================

SLaM pipelines break the analysis of 'galaxy-scale' strong lenses down into multiple pipelines which focus on modeling
a specific aspect of the strong lens, first the Source, then the (lens) Light and finally the Mass. Each of these
pipelines has it own inputs which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS TOTAL PIPELINE.

Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE, MASS TOTAL PIPELINE and SUBHALO PIPELINE this SLaM script
fits `Imaging` of a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge with a parametric `Sersic` light profile.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - A dark matter subhalo near The lens galaxy mass is included as a`NFWMCRLudlowSph`.
 - The source galaxy is an `Inversion`.

It ends by performing sensitivity mapping of the data using the above model, so as to determine where in the data
subhalos of a given mass could have been detected if present.

This uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`
 `subhalo/detection`

Check them out for a full description of the analysis!
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
dataset_name = "light_sersic_exp__dark_matter_subhalo"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
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
    number_of_cores=2,
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
setup_adapt = al.SetupAdapt()

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
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
In this example it:

 - Uses a parametric `Sersic` bulge for the lens galaxy's light [Do not use the results of the SOURCE LP PIPELINE to 
 initialize priors].

 - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].

 - Uses the `Sersic` model representing a bulge for the source's light [fixed from SOURCE LP PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
"""
analysis = al.AnalysisImaging(dataset=dataset, adapt_result=source_lp_results.last)

bulge = af.Model(al.lp.Sersic)

light_results = slam.light_lp.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_lp_results,
    lens_bulge=bulge,
    lens_disk=None,
)

"""
__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors and the lens light
model of the LIGHT LP PIPELINE. 

In this example it:

 - Uses a parametric `Sersic` bulge for the lens galaxy's light [fixed from LIGHT LP PIPELINE].

 - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
 PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].
 
 - Uses the `Sersic` model representing a bulge for the source's light [priors initialized from SOURCE 
 PARAMETRIC PIPELINE].
 
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS TOTAL PIPELINE.
"""
analysis = al.AnalysisImaging(dataset=dataset, adapt_result=source_lp_results.last)

mass_results = slam.mass_total.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_lp_results,
    light_results=light_results,
    mass=af.Model(al.mp.PowerLaw),
)

"""
__SUBHALO PIPELINE (sensitivity mapping)__

The SUBHALO PIPELINE (sensitivity mapping) performs sensitivity mapping of the data using the lens model
fitted above, so as to determine where subhalos of what mass could be detected in the data. A full description of
Sensitivity mapping if given in the script `sensitivity_mapping.py`.

Each model-fit performed by sensitivity mapping creates a new instance of an `Analysis` class, which contains the
data simulated by the `simulate_cls` for that model. This requires us to write a wrapper around the 
PyAutoLens `AnalysisImaging` class.
"""


class AnalysisImagingSensitivity(al.AnalysisImaging):
    def __init__(self, dataset):
        # TODO : PRELOADS, need to make sure w_tilde isnt repeated over and over.

        super().__init__(dataset=dataset)

        self.adapt_galaxy_image_path_dict = (
            mass_results.last.adapt_galaxy_image_path_dict
        )
        self.adapt_model_image = mass_results.last.adapt_model_image


subhalo_results = slam.subhalo.sensitivity_mapping_imaging(
    settings_autofit=settings_autofit,
    analysis_cls=AnalysisImagingSensitivity,
    mask=mask,
    psf=dataset.psf,
    mass_results=mass_results,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)
"""
Finish.
"""
