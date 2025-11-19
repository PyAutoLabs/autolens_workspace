"""
SLaM (Source, Light and Mass): Mass Total + Source Parametric
=============================================================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for fitting a
lens model where the source is a modeled using a Multi Gaussian Expansion (MGE).

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that
guide before working through this example.

This example only provides documentation specific to the use of an MGE source, describing how the pipeline
differs from the standard SLaM pipelines described in the SLaM start here guide.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines
  and how they integrate into strong-lens modeling.

You can still run the script without fully understanding the guide, but reviewing it later will
make the structure and choices of the SLaM workflow clearer.

__Interferometer SLaM Description__

The `slam_start_here` notebook provides a detailed description of the SLaM pipelines, but it does this using CCD
imaging data.

There is no dedicated example which provides full descriptions of the SLaM pipelines using interferometer data, however,
the concepts and API described in the `slam_start_here` are identical to what is required for interferometer data.
Therefore, by reading the `slam_start_here` example you will fully understand everything required to use this
interferometer SLaM script.

__Model__

Using a SOURCE LP PIPELINE and a MASS TOTAL PIPELINE this SLaM script fits `Interferometer` of a strong lens system, where
in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy's light is an MGE.

This uses the SLaM pipelines:

 `source_lp`
 `mass_total`

Check them out for a full description of the analysis!
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import sys
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
import slam_pipeline

"""
__Dataset + Masking__ 

Load the `Interferometer` data, define the visibility and real-space masks and plot them.
"""
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(151, 151), pixel_scales=0.05, radius=mask_radius
)


dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("interferometer", "slam"),
    unique_tag=dataset_name,
    info=None,
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

The SOURCE LP PIPELINE is identical to the `slam_start_here.ipynb` example.
"""
analysis = al.AnalysisInterferometer(dataset=dataset)

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=None,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=0.5,
    redshift_source=1.0,
)


"""
Compared to the `slam_start_here` example, this SLaM pipeline skips the SOURCE PIX PIPELINE because the MGE
is a parametric profile and skips the LIGHT LP PIPELINE because there is no lens light to model.

__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE is again identical to the `slam_start_here.ipynb` example, however because 
`source_result_for_source` uses an MGE model, the source light model is now an MGE instead of a pixelization.
"""
analysis = al.AnalysisInterferometer(dataset=dataset)

mass_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_lp_result,
    source_result_for_source=source_lp_result,
    light_result=None,
    mass=af.Model(al.mp.PowerLaw),
)

"""
Finish.
"""
