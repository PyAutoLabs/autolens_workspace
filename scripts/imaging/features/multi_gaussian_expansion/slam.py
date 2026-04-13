"""
Multi Gaussian Expansion: SLaM
==============================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for fitting a
lens model where the source is modeled using a Multi Gaussian Expansion (MGE).

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that
guide before working through this example.

This example only provides documentation specific to the use of an MGE source, describing how the pipeline
differs from the standard SLaM pipelines described in the SLaM start here guide.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
**Model:** Compose the lens model fitted to the data.
**SOURCE LP PIPELINE:** Identical to `slam_start_here.py` with `gaussian_per_basis=1` for both the lens and source MGE.
**LIGHT LP PIPELINE:** Identical to `slam_start_here.py`, except.
**MASS TOTAL PIPELINE:** Identical to `slam_start_here.py`, except.
**Dataset:** Load and plot the strong lens dataset.
**Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
**Redshifts:** The redshifts of the lens and source galaxies.
**SLaM Pipeline:** The code below calls the full SLaM PIPELINE.

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

__Start Here Notebook__

If any code in this script is unclear, refer to the `slam_start_here` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__SOURCE LP PIPELINE__

Identical to `slam_start_here.py` with `gaussian_per_basis=1` for both the lens and source MGE models.

Because the source is parametric (an MGE), the SOURCE PIX PIPELINE is skipped. The LIGHT LP and MASS TOTAL
pipelines use `source_lp_result` directly as both the lens and source initialization result.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    redshift_lens: float,
    redshift_source: float,
    n_batch: int = 50,
) -> af.Result:
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

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=None,
                mass=af.Model(al.mp.Isothermal),
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
                disk=None,
            ),
        ),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__LIGHT LP PIPELINE__

Identical to `slam_start_here.py`, except:

 - `source_result_for_lens` and `source_result_for_source` are both set to `source_lp_result`, because
   there is no SOURCE PIX PIPELINE for an MGE source.
 - The analysis does not use adapt images (not required for a parametric MGE source).
"""


def light_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    source_lp_result: af.Result,
    n_batch: int = 20,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=True,
    )

    source = al.util.chaining.source_custom_model_from(
        result=source_lp_result, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=None,
                mass=source_lp_result.instance.galaxies.lens.mass,
                shear=source_lp_result.instance.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Identical to `slam_start_here.py`, except:

 - `source_result_for_lens` and `source_result_for_source` are both set to `source_lp_result`, because
   there is no SOURCE PIX PIPELINE for an MGE source.
 - The analysis does not use adapt images or positions likelihood (not required for a parametric MGE source).
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    light_result: af.Result,
    n_batch: int = 20,
) -> af.Result:
    # Total mass model for the lens galaxy.
    mass = af.Model(al.mp.PowerLaw)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    bulge = light_result.instance.galaxies.lens.bulge
    disk = light_result.instance.galaxies.lens.disk

    source = al.util.chaining.source_from(result=source_lp_result)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=bulge,
                disk=disk,
                mass=mass,
                shear=source_lp_result.model.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

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

aplt.subplot_imaging_dataset(dataset=dataset)

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
__SLaM Pipeline__

The code below calls the full SLaM PIPELINE. See the documentation string above each Python function for
a description of each pipeline step.
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    source_lp_result=source_lp_result,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    light_result=light_result,
)

"""
Finish.
"""
