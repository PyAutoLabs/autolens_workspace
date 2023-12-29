"""
SLaM (Source, Light and Mass): Start Here
=========================================

This scripts gives an introduce to the Source, (lens) Light and Mass (SLaM) pipelines. These are advanced modeling
pipelines which use many aspects of core PyAutoLens functionality to automate the modeling of strong lenses.

Before reading this script, you should be familiar with non-linear search chaining, illustrated in the
`imaging/advanced/chaining` package, and have experience using have looked at multiple examples explaining why its
a powerful tool to chain models together (e.g. going from a light profile source to pixelixed source).

The Source, (lens) Light and Mass pipelines (SLaM) chain together between 3-6 searches in a carefully crafted way
which fully exploit the different benefits of search chaining. They provide a fully automated framework for fitting
large samples of strong lenses with complex strong lens models. They are the default pipelines used by many PyAutoLens
scientific publications.

__Pipeline Structure__

Each pipeline that makes up the overall SLaM pipelines focuses on a specific aspect of the strong lens model: 

 - Source: The first pipeline(s) infer a robust source model. For pixelized sources, this means robust values for the
   mesh and regularization. For light profile sources, this means estimates of its initial parameters.
   
 - Light: The next pipeline infers the lens light, using fixed source and mass models from the previous pipeline(s).
   
 - Mass: The final pipeline infers a mass model of any complexity, using source and lens light models initialized
   from the previous pipeline(s).

The models fitted in earlier pipelines determine the models used in later pipelines. For example, if the SOURCE
PIPELINE uses a pixelized `Voronoi` mesh for the source, this will be used in the subsequent MASS TOTAL PIPELINE.

__Design Choices__

There are many design choices that go into the SLaM pipelines, which we discuss now.

The SLaM pipelines are designed around pixelixed source modeling. Pixelized sources are necessary for fitting complex
mass models, which the SLaM pipelines automates the fitting of. However, the SLaM pipelines support fitting of
light profile sources, and using the SLaM pipelines in this way will still provide automated and robust lens modeling.

We now list the design considerations which dictate the ordering of the SLaM pipelines, which were driven by the use
of pixelized source modeling:

 - Starting With the Source: The SLaM pipelines begin with the Source Pipeline because complex mass models (w.g. a
   `PowerLaw` or stars plus dark matter model) cannot be fitted accurately using a light profiles for the source. A
   robust pixelized source model (e.g. the mesh and regularization) can be set up using a simpler mass
   model (e.g. `Isothermal` plus `Shear`).

 - Positions: Pixelized source modeling requires the positions of the multiple images of the lensed source(s) to be
   input into the analysis, to prevent unphysical pixelized source reconstructions. The SLaM pipelines can determine
   these positions automatically, using the mass and source model of the results of the SOURCE LP PIPELINE. However,
   WE RECOMMEND users still input positions manually, as this provides a more robust analysis.

 - Adapt Images: Advanced pixelized source models use "adapt images" to adapt the mesh and regularization weights to
   the source's morphology. This functionality is discussed in `advanced/chaining/pix_adapt`. The SLaM pipelines
   gradually improve the lens model and therefore gradually improve these adapt images, ensuring this adaptive
   functionality can produce the best adaptation to the source's properties.
   
 - Lens Light Before Mass: Fitting complex lens light models requires clean deblending of the lens and lensed source
   emission. This requires a robust pixelized source model, which simultaneously reconstructs the source light when
   fitting the lens light. This deblending can be performed accurate using a simpler mass model. Therefore, the SLaM
   pipelines refine the lens light before fitting a complex mass model. Certain mass models fit stars plus dark
   models and therefore require this very accurate lens light model to be inferred first.
   
 - Mass Last: The SLaM pipelines fit the most complex mass models last. This is because they require the most accurate
   pixelized source model possible, and the most accurate lens light model possible.

__This Script__

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

__Start Here Notebook__

If any code in this script is unclear, refer to the `chaining/start_here.ipynb` notebook.
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
    settings_search=settings_search,
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

The first search, which is an initialization search, fits an `Overlay` image-mesh, `Delaunay` mesh and `Constant` 
regularization. 

The second search, which uses the mesh and regularization used throughout the remainder of the SLaM pipelines,
fits the following model:

- Uses a `KMeans` image-mesh. 
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
    adapt_images=source_lp_results.last.adapt_images,
    positions_likelihood=source_lp_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

source_pix_results = slam.source_pix.run(
    settings_search=settings_search,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_lp_results=source_lp_results,
    image_mesh=al.image_mesh.KMeans,
    mesh=al.mesh.Delaunay,
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
    adapt_images=source_pix_results[0].adapt_images,
    positions_likelihood=source_pix_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

bulge = af.Model(al.lp.Sersic)

light_results = slam.light_lp.run(
    settings_search=settings_search,
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
    settings_search=settings_search,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_pix_results,
    light_results=light_results,
    mass=af.Model(al.mp.PowerLaw),
)

"""
Finish.
"""
