"""
Multi Gaussian Expansion: SLaM
==============================

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

__Model__

Using a SOURCE LP PIPELINE, LIGHT PIPELINE and a MASS TOTAL PIPELINE this SLaM script fits a strong
lens system, where in the final model:

 - The lens galaxy's light is a bulge with an MGE.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy's light is an MGE.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `light_lp`
 `mass_total`

Check them out for a detailed description of the analysis!
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
__Dataset__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("imaging") / "slam",
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
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=True,
)

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
Compared to the `slam_start_here` example, this SLaM pipeline skips the SOURCE PIX PIPELINE because the MGE
is a parametric profile.

__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE is setup identically to the `slam_start_here.ipynb` example, however because 
`source_result_for_source` uses an MGE model, the source light model is now an MGE instead of a pixelization.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=True,
)

light_results = slam_pipeline.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_lp_result,
    source_result_for_source=source_lp_result,
    lens_bulge=bulge,
    lens_disk=None,
)

"""
__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE is again identical to the `slam_start_here.ipynb` example, noting again that the
`source_result_for_source` uses an MGE model.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

mass_results = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_lp_result,
    source_result_for_source=source_lp_result,
    light_result=light_results,
    mass=af.Model(al.mp.PowerLaw),
)

"""
Finish.
"""
