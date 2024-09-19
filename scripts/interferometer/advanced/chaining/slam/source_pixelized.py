"""
SLaM (Source, Light and Mass): Mass Total + Source Inversion
============================================================

SLaM pipelines break the analysis of 'galaxy-scale' strong lenses down into multiple pipelines which focus on modeling
a specific aspect of the strong lens, first the Source, then the (lens) Light and finally the Mass. Each of these
pipelines has it own inputs which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a linear parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS TOTAL PIPELINE.

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE and a MASS TOTAL PIPELINE this SLaM script fits `Interferometer` 
of a strong lens system, where in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy is an `Inversion`.

This uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `mass_total`

Check them out for a detailed description of the analysis!

__Run Times and Settings__

The run times of an interferometer `Inversion` depend significantly on the following settings:

 - `transformer_class`: whether a discrete Fourier transform (`TransformerDFT`) or non-uniform fast Fourier Transform
 (`TransformerNUFFT) is used to map the inversion's image from real-space to Fourier space.

 - `use_linear_operators`: whether the linear operator formalism or matrix formalism is used for the linear algebra.

The optimal settings depend on the number of visibilities in the dataset:

 - For N_visibilities < 1000: `transformer_class=TransformerDFT` and `use_linear_operators=False` gives the fastest
 run-times.
 - For  N_visibilities > ~10000: use `transformer_class=TransformerNUFFT`  and `use_linear_operators=True`.

The dataset modeled by default in this script has just 200 visibilties, therefore `transformer_class=TransformerDFT`
and `use_linear_operators=False`.

The script `autolens_workspace/*/interferometer/run_times.py` allows you to compute the run-time of an inversion
for your interferometer dataset. It does this for all possible combinations of settings and therefore can tell you
which settings give the fastest run times for your dataset.

__Start Here Notebook__

If any code in this script is unclear, refer to the `autolens_workspace/imaging/advanced/chaining/slam/start_here.ipynb`
notebook.
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

Load the `Interferometer` data, define the visibility and real-space masks and plot them.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(151, 151), pixel_scales=0.05, radius=3.0
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
__Inversion Settings (Run Times)__

The run times of an interferometer `Inversion` depend significantly on the following settings:

 - `transformer_class`: whether a discrete Fourier transform (`TransformerDFT`) or non-uniform fast Fourier Transform
 (`TransformerNUFFT) is used to map the inversion's image from real-space to Fourier space.

 - `use_linear_operators`: whether the linear operator formalism or matrix formalism is used for the linear algebra.

The optimal settings depend on the number of visibilities in the dataset:

 - For N_visibilities < 1000: `transformer_class=TransformerDFT` and `use_linear_operators=False` gives the fastest
 run-times.
 - For  N_visibilities > ~10000: use `transformer_class=TransformerNUFFT`  and `use_linear_operators=True`.

The dataset modeled by default in this script has just 200 visibilties, therefore `transformer_class=TransformerDFT`
and `use_linear_operators=False`. If you are using this script to model your own dataset with a different number of
visibilities, you should update the options below accordingly.

The script `autolens_workspace/*/interferometer/run_times.py` allows you to compute the run-time of an inversion
for your interferometer dataset. It does this for all possible combinations of settings and therefore can tell you
which settings give the fastest run times for your dataset.
"""
settings_inversion = al.SettingsInversion(
    use_linear_operators=False,
    image_mesh_min_mesh_pixels_per_pixel=3,
    image_mesh_min_mesh_number=5,
    image_mesh_adapt_background_percent_threshold=0.1,
    image_mesh_adapt_background_percent_check=0.8,
)

"""
We now plot the `Interferometer` object which is used to fit the lens model.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=path.join("interferometer", "slam"),
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
__SOURCE LP PIPELINE__

The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light, which in
this example:

 - Uses a linear parametric `Sersic` bulge for the source's light (omitting a disk / envelope).
 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
 
__Settings__:
 
 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the SOURCE INVERSION 
 PIPELINE).
"""
analysis = al.AnalysisInterferometer(dataset=dataset)

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=None,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp_linear.SersicCore),
    mass_centre=(0.0, 0.0),
    redshift_lens=0.5,
    redshift_source=1.0,
)

"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE uses two searches to initialize a robust model for the `Pixelization` that
reconstructs the source galaxy's light. It begins by fitting an `Overlay` image-mesh, `Delaunay` mesh and `Constant` 
regularization, to set up the model and hyper images, and then:

- Uses a `Hilbert` image-mesh. 
- Uses a `Delaunay` mesh.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood=source_lp_result.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
    settings_inversion=settings_inversion,
)

source_pix_result_1 = slam.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Delaunay,
)

analysis = al.AnalysisInterferometer(
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
__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize the model priors. In this 
example it:

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
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    positions_likelihood=source_pix_result_1.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    settings_inversion=settings_inversion,
)

mass_result = slam.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=None,
    mass=af.Model(al.mp.PowerLaw),
)

"""
Finish.
"""
