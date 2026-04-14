"""
Multi Gaussian Expansion: Group SLaM
=====================================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for fitting a
group-scale strong lens where all light profiles use Multi Gaussian Expansion (MGE) models.

The standard group SLaM pipeline (``group/slam.py``) already uses MGE by default for all galaxies. This
feature script serves as a reference implementation documenting the MGE-specific choices and explaining
why MGE is the default approach for group-scale modeling.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that
guide before working through this example.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
**This Script:** Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE and MASS TOTAL PIPELINE.
**SOURCE LP PIPELINE:** Fits lens light and source light using MGE, with Isothermal mass and ExternalShear.
**LIGHT LP PIPELINE:** Refits lens light with a fresh MGE, mass and source fixed from SOURCE LP.
**MASS TOTAL PIPELINE:** Refits the mass using a PowerLaw, light and source fixed from previous stages.
**Dataset:** Load and plot the strong lens dataset.
**Settings AutoFit:** The settings of autofit.
**SLaM Pipeline:** The code below calls the full SLaM PIPELINE.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines.

- **Group Modeling** (`group/modeling`)
  How we model group-scale strong lenses, including extra galaxies.

- **MGE Feature** (`imaging/features/multi_gaussian_expansion`)
  The Multi Gaussian Expansion light profile and its advantages.

__This Script__

Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE and a MASS TOTAL PIPELINE this SLaM modeling script
fits `Imaging` data of a group-scale strong lens where in the final model:

 - Each main lens galaxy has a free MGE bulge and a `PowerLaw` total mass.
 - Each extra galaxy has a free MGE bulge and an `IsothermalSph` mass with bounded Einstein radius.
 - The source galaxy's light is an MGE.

Because the source is parametric (an MGE), the SOURCE PIX PIPELINE is skipped. This is a simpler
pipeline than the full group SLaM (which transitions to a pixelized source), and is appropriate
when an MGE source model is sufficient.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__SOURCE LP PIPELINE__

Fits the lens light, source light, lens mass and external shear simultaneously using MGE models.

For group-scale lenses:
 - Each main lens galaxy gets a 20-Gaussian MGE with `gaussian_per_basis=1` and uniform centre priors.
 - Each extra galaxy gets a 10-Gaussian MGE with centres fixed to the observed positions and a bounded
   `IsothermalSph` mass.
 - The source galaxy gets a 20-Gaussian MGE with Gaussian centre priors.
 - Only `lens_0` carries an `ExternalShear`.

The MGE source means the SOURCE PIX PIPELINE is not needed, significantly simplifying the overall pipeline.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    main_lens_centres,
    extra_galaxies_centres,
    redshift_lens: float,
    redshift_source: float,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    # Main Lens Galaxies:

    lens_models = []

    for i, centre in enumerate(main_lens_centres):

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=1,
            centre_prior_is_uniform=True,
        )

        mass = af.Model(al.mp.Isothermal)

        lens = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=bulge,
            mass=mass,
            shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
        )

        lens_models.append(lens)

    # Extra Galaxies:

    extra_galaxies_list = []

    for centre in extra_galaxies_centres:

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius, total_gaussians=10, centre_fixed=centre
        )

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

        extra_galaxy = af.Model(
            al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=mass
        )
        extra_galaxies_list.append(extra_galaxy)

    extra_galaxies = af.Collection(extra_galaxies_list) if extra_galaxies_list else None

    # Source:

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )

    source = af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge)

    # Overall Lens Model:

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
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

Refits the lens light with a fresh MGE model, keeping the mass and source fixed from the SOURCE LP result.

For group-scale lenses:
 - Each main lens galaxy gets a fresh 20-Gaussian MGE with uniform centre priors.
 - Each extra galaxy gets a fresh 10-Gaussian MGE with centres fixed to the observed positions.
 - The source is fixed as an instance from the SOURCE LP result.
 - Mass profiles are fixed as instances from the SOURCE LP result.
"""


def light_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    main_lens_centres,
    extra_galaxies_centres,
    source_lp_result: af.Result,
    redshift_lens: float,
    n_batch: int = 20,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    # Main Lens Galaxies (fresh MGE, mass fixed):

    lens_models = []

    n_main = sum(
        1 for k in vars(source_lp_result.instance.galaxies) if k.startswith("lens_")
    )

    for i in range(n_main):
        lens_instance = getattr(source_lp_result.instance.galaxies, f"lens_{i}")

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=1,
            centre_prior_is_uniform=True,
        )

        lens = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=bulge,
            mass=lens_instance.mass,
            shear=lens_instance.shear,
        )

        lens_models.append(lens)

    # Extra Galaxies (fresh MGE, mass fixed):

    extra_galaxies_list = []

    if source_lp_result.instance.extra_galaxies is not None:
        for i, centre in enumerate(extra_galaxies_centres):
            extra_instance = source_lp_result.instance.extra_galaxies[i]

            bulge = al.model_util.mge_model_from(
                mask_radius=mask_radius, total_gaussians=10, centre_fixed=centre
            )

            extra_galaxy = af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=bulge,
                mass=extra_instance.mass,
            )
            extra_galaxies_list.append(extra_galaxy)

    extra_galaxies = af.Collection(extra_galaxies_list) if extra_galaxies_list else None

    # Source (fixed from SOURCE LP):

    source = al.util.chaining.source_custom_model_from(
        result=source_lp_result, source_is_model=False
    )

    # Overall Lens Model:

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
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

Refits the total mass distribution using a `PowerLaw` for the main lens galaxies, with light and source
fixed from previous stages.

For group-scale lenses:
 - Each main lens galaxy's mass is upgraded to a `PowerLaw`.
 - Each extra galaxy keeps its `IsothermalSph` mass with bounded Einstein radius.
 - Light profiles are fixed as instances from the LIGHT LP result.
 - The source is fixed from the SOURCE LP result.
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    light_result: af.Result,
    extra_galaxies_centres,
    redshift_lens: float,
    n_batch: int = 20,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    n_main = sum(
        1 for k in vars(light_result.instance.galaxies) if k.startswith("lens_")
    )

    # Main Lens Galaxies (fixed light, free PowerLaw mass):

    lens_models = []

    for i in range(n_main):
        light_lens_instance = getattr(light_result.instance.galaxies, f"lens_{i}")
        source_lp_lens_model = getattr(source_lp_result.model.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.PowerLaw),
            mass_result=source_lp_lens_model.mass,
            unfix_mass_centre=True,
        )

        lens = af.Model(
            al.Galaxy,
            redshift=light_lens_instance.redshift,
            bulge=light_lens_instance.bulge,
            mass=mass,
            shear=source_lp_result.model.galaxies.lens_0.shear
            if i == 0
            else None,
        )

        lens_models.append(lens)

    # Extra Galaxies (fixed light, free bounded mass):

    extra_galaxies_list = []

    if light_result.instance.extra_galaxies is not None:
        for i, centre in enumerate(extra_galaxies_centres):
            light_extra = light_result.instance.extra_galaxies[i]

            mass = af.Model(al.mp.IsothermalSph)
            mass.centre = centre
            mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

            extra_galaxy = af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=light_extra.bulge,
                mass=mass,
            )
            extra_galaxies_list.append(extra_galaxy)

    extra_galaxies = af.Collection(extra_galaxies_list) if extra_galaxies_list else None

    # Source (fixed from SOURCE LP):

    source = al.util.chaining.source_from(result=source_lp_result)

    # Overall Lens Model:

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
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
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Galaxy Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("group") / "slam",
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
    main_lens_centres=main_lens_centres,
    extra_galaxies_centres=extra_galaxies_centres,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    main_lens_centres=main_lens_centres,
    extra_galaxies_centres=extra_galaxies_centres,
    source_lp_result=source_lp_result,
    redshift_lens=redshift_lens,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    light_result=light_result,
    extra_galaxies_centres=extra_galaxies_centres,
    redshift_lens=redshift_lens,
)

"""
Finish.
"""
