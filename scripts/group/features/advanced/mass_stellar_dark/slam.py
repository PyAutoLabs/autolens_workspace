"""
SLaM (Source, Light and Mass): Group Mass Stellar Dark
======================================================

This script adapts the SLaM (Source, Light and Mass) pipelines to a group-scale strong lens where each main
lens galaxy is decomposed into a stellar component (tied to its light via a mass-to-light ratio) and a dark
matter halo.

This script is the group analogue of:

 - `guides/modeling/slam_start_here.py` — the canonical single-galaxy SLaM walkthrough.
 - `scripts/imaging/features/advanced/mass_stellar_dark/slam.py` — the single-galaxy decomposed-mass SLaM.
 - `scripts/group/features/advanced/double_source_plane_lens/slam.py` — the group SLaM with two source planes (the
   `lens_dict` plumbing in that script is the structural template used here, simplified for a single source
   plane).

Each pipeline stage is a plain inline Python function. Per-lens priors are chained via
`al.util.chaining.mass_from`, image positions are derived automatically via `positions_likelihood_from`, and
MGE light profiles are constructed via `al.model_util.mge_model_from`.

__Group-Specific Differences From Standard SLaM__

 - The lens-plane (z=0.5) is composed via the group `lens_dict` convention: one `af.Model(al.Galaxy)` entry per
   main lens galaxy centre, with the `ExternalShear` attached only to `lens_0`.
 - Each pipeline iterates over the main lens galaxies via `lens_{i}` keys rather than referencing a single
   `lens` attribute.
 - The MASS LIGHT DARK pipeline constructs the per-galaxy `lmp.Sersic + NFWSph` manually rather than calling
   `al.util.chaining.mass_light_dark_from`, which only supports the single-`lens` layout.

__This Script__

Using a SOURCE LP, SOURCE PIX 1, SOURCE PIX 2, LIGHT LP and a MASS LIGHT DARK pipeline, this group SLaM script
fits an `Imaging` dataset where in the final model:

 - Each main lens galaxy's light is a `Sersic` linear light profile.
 - Each main lens galaxy's stellar mass distribution is a `Sersic` tied to its OWN light via a
   `mass_to_light_ratio` (one per galaxy, free parameters).
 - Each main lens galaxy's dark matter mass distribution is an `NFWSph` aligned with the bulge centre.
 - The first main lens galaxy carries an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__Helpers__

`build_lens_dict_source_lp` constructs an `af.Model` lens_dict for the SOURCE LP pipeline: each main lens
galaxy has an MGE bulge plus a free `Isothermal` mass profile, with `ExternalShear` on `lens_0`.

`build_lens_dict_light_lp` constructs an `af.Model` lens_dict for the LIGHT LP pipeline: each main lens galaxy
has a `lp_linear.Sersic` bulge (chosen because the MASS LIGHT DARK pipeline requires a `LightMassProfile`
sharing the same profile type), with mass and shear fixed from the SOURCE PIX 1 result.

`build_lens_dict_mass_light_dark` constructs the final MASS LIGHT DARK lens_dict: each lens galaxy's bulge is
swapped to `lmp.Sersic` with priors carried over from LIGHT LP, plus a new `NFWSph` dark halo.
"""


def n_lens_from(result) -> int:
    return len(
        [name for name in vars(result.instance.galaxies) if name.startswith("lens_")]
    )


def lens_galaxy_image_dict(result, n_lens: int) -> dict:
    full = al.galaxy_name_image_dict_via_result_from(result=result)
    return {
        f"('galaxies', 'lens_{i}')": full[f"('galaxies', 'lens_{i}')"]
        for i in range(n_lens)
    }


def build_lens_dict_source_lp(
    main_lens_centres,
    redshift_lens: float,
    mask_radius: float,
    sigma_min: float,
    total_gaussians: int = 20,
    gaussian_per_basis: int = 2,
):
    lens_dict = {}
    for i, centre in enumerate(main_lens_centres):
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=total_gaussians,
            gaussian_per_basis=gaussian_per_basis,
            centre_prior_is_uniform=True,
            centre=(centre[0], centre[1]),
            sigma_min=sigma_min,
        )

        mass = af.Model(al.mp.Isothermal)
        mass.centre = (centre[0], centre[1])

        kwargs = dict(redshift=redshift_lens, bulge=bulge, mass=mass)
        if i == 0:
            kwargs["shear"] = af.Model(al.mp.ExternalShear)

        lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, **kwargs)
    return lens_dict


"""
__SOURCE LP PIPELINE__

Initial fit using MGE bulges + `Isothermal` mass for each main lens galaxy, with an MGE source. This
constrains the source-plane geometry and the per-galaxy total-mass before subsequent pipelines decompose it.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    main_lens_centres,
    mask_radius: float,
    redshift_lens: float,
    redshift_source: float,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_dict = build_lens_dict_source_lp(
        main_lens_centres=main_lens_centres,
        redshift_lens=redshift_lens,
        mask_radius=mask_radius,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
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
__SOURCE PIX PIPELINE 1__

Pixelize the source using an initial mesh / regularization. Each main lens galaxy's mass is freed with priors
chained from the SOURCE LP pipeline via `al.util.chaining.mass_from`. Adapt images are stitched per-lens.

__Adapt Image S/N Cap__

The source adapt image is capped at a signal-to-noise of 3.0 before it is used by the adaptive
image-mesh and the adaptive regularization. Without the cap the brightest peak dominates the
weights (they scale as a power of the adapt image), so fainter multiply-imaged features get too
few source pixels and too little regularization weight. Capping makes every feature above S/N 3.0
count equally. The cap is applied to an explicit copy so the raw S/N image is untouched.
"""


def source_pix_1(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    mesh_init,
    regularization_init,
    n_batch: int = 20,
) -> af.Result:
    n_lens = n_lens_from(source_lp_result)

    galaxy_image_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_dict)

    positions_likelihood = source_lp_result.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood],
        use_jax=True,
    )

    lens_dict = {}
    for i in range(n_lens):
        lens_inst = getattr(source_lp_result.instance.galaxies, f"lens_{i}")
        lens_model_mass = getattr(source_lp_result.model.galaxies, f"lens_{i}").mass

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.Isothermal),
            mass_result=lens_model_mass,
            unfix_mass_centre=True,
        )

        kwargs = dict(
            redshift=lens_inst.redshift,
            bulge=lens_inst.bulge,
            mass=mass,
        )
        if i == 0:
            kwargs["shear"] = source_lp_result.model.galaxies.lens_0.shear

        lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, **kwargs)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Refines the source pixelization with an adapt mesh derived from the SOURCE PIX 1 source reconstruction. Each
main lens galaxy's bulge, mass and (for `lens_0`) shear are fixed to the SOURCE PIX 1 instance.
"""


def source_pix_2(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    source_pix_result_1: af.Result,
    mesh,
    regularization,
    n_batch: int = 20,
) -> af.Result:
    n_lens = n_lens_from(source_lp_result)

    galaxy_image_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    lens_dict = {}
    for i in range(n_lens):
        lens_inst_lp = getattr(source_lp_result.instance.galaxies, f"lens_{i}")
        lens_inst_pix = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        kwargs = dict(
            redshift=lens_inst_lp.redshift,
            bulge=lens_inst_lp.bulge,
            mass=lens_inst_pix.mass,
        )
        if i == 0:
            kwargs["shear"] = lens_inst_pix.shear

        lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, **kwargs)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh,
                    regularization=regularization,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__LIGHT LP PIPELINE__

Refits each main lens galaxy's light using a `lp_linear.Sersic` bulge (replacing the SOURCE LP MGE bulge). The
linear `Sersic` is the light profile type the MASS LIGHT DARK pipeline pairs with `lmp.Sersic` for the
stellar-mass coupling. Mass + shear + source pixelization are fixed from SOURCE PIX 1.
"""


def light_lp(
    settings_search: af.SettingsSearch,
    dataset,
    main_lens_centres,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    redshift_lens: float,
    n_batch: int = 20,
) -> af.Result:
    n_lens = n_lens_from(source_result_for_lens)

    galaxy_image_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_dict)

    analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

    lens_dict = {}
    for i, centre in enumerate(main_lens_centres):
        bulge = af.Model(al.lp_linear.Sersic)
        bulge.centre = (centre[0], centre[1])

        lens_inst = getattr(source_result_for_lens.instance.galaxies, f"lens_{i}")

        kwargs = dict(
            redshift=redshift_lens,
            bulge=bulge,
            mass=lens_inst.mass,
        )
        if i == 0:
            kwargs["shear"] = source_result_for_lens.instance.galaxies.lens_0.shear

        lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, **kwargs)

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS LIGHT DARK PIPELINE__

The terminal pipeline. Each main lens galaxy's bulge is swapped from the LIGHT LP `lp_linear.Sersic` to a
`lmp.Sersic` (light + stellar mass coupled via `mass_to_light_ratio`); priors on the bulge geometry / intensity
are carried over via `take_attributes`. A separate `NFWSph` dark halo is added per galaxy with its centre fixed
to the bulge centre.

This pipeline is the per-galaxy generalisation of the imaging
`scripts/imaging/features/advanced/mass_stellar_dark/slam.py` MASS LIGHT DARK stage. Because
`al.util.chaining.mass_light_dark_from` accesses `light_result.instance.galaxies.lens.<name>` directly (a
hardcoded single-lens path), the per-galaxy decomposition is constructed manually here.
"""


def mass_light_dark(
    settings_search: af.SettingsSearch,
    dataset,
    main_lens_centres,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: af.Result,
    n_batch: int = 20,
) -> af.Result:
    n_lens = n_lens_from(light_result)

    galaxy_image_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_result_for_source.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    lens_dict = {}
    for i, centre in enumerate(main_lens_centres):
        bulge = af.Model(al.lmp.Sersic)
        bulge.take_attributes(source=getattr(light_result.model.galaxies, f"lens_{i}"))
        bulge.centre = (centre[0], centre[1])
        bulge.mass_to_light_ratio = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

        dark = af.Model(al.mp.NFWSph)
        dark.centre = (centre[0], centre[1])
        dark.kappa_s = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        dark.scale_radius = af.UniformPrior(lower_limit=5.0, upper_limit=50.0)

        kwargs = dict(
            redshift=light_result.instance.galaxies.lens_0.redshift,
            bulge=bulge,
            dark=dark,
        )
        if i == 0:
            kwargs["shear"] = source_result_for_lens.model.galaxies.lens_0.shear

        lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, **kwargs)

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

    search = af.Nautilus(
        name="mass_light_dark[1]",
        **settings_search.search_dict,
        n_live=250,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

Load, plot and mask the `Imaging` data.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset") / "group" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/mass_stellar_dark/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

mask_radius = 3.7

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("group") / "slam" / "mass_stellar_dark",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Mesh Shape__
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__SLaM Pipeline__

The code below runs the full group decomposed-mass SLaM pipeline. See the docstring above each function for a
description of each stage.
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    main_lens_centres=main_lens_centres,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    mesh_init=af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape),
    regularization_init=al.reg.Adapt,
)

"""
__Adaptive Pixelization Over-Sampling__

From SOURCE PIX PIPELINE 2 onwards the pixelization grid is over-sampled adaptively. The source's
signal-to-noise map from the previous pixelized fit (the same map that becomes the adapt image, read before
the S/N 3.0 cap is applied) is thresholded at S/N 3.0: pixels above it, the bright lensed source, use a
sub-size of 4 and every other pixel uses a sub-size of 2. This concentrates the extra over-sampling where the
source is bright and the pixelization gains the most accuracy from it, and keeps the rest of the mask cheap.

The map returned by `galaxy_name_image_dict_via_result_from` is already signal divided by noise, so it is
thresholded directly.

SOURCE PIX PIPELINE 1 keeps the dataset's default uniform sub-size. Its adapt image comes from the parametric
source fit of the SOURCE LP PIPELINE, which does not yet trace the lensed source well enough to steer
over-sampling.
"""
galaxy_image_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)

# Bound before the cap: the over-sampling map below uses the raw (uncapped) S/N image.
source_image_raw = galaxy_image_dict["('galaxies', 'source')"]

# Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
adapt_image_snr_cap = 3.0

source_adapt_image = galaxy_image_dict["('galaxies', 'source')"].copy()
source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
galaxy_image_dict["('galaxies', 'source')"] = source_adapt_image

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_dict)

signal_to_noise_threshold = 3.0

over_sample_size_pixelization = al.Array2D(
    values=np.where(source_image_raw > signal_to_noise_threshold, 4, 2),
    mask=dataset.mask,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=over_sample_size_pixelization,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape),
    regularization=al.reg.Adapt,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset,
    main_lens_centres=main_lens_centres,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    redshift_lens=redshift_lens,
)

mass_result = mass_light_dark(
    settings_search=settings_search,
    dataset=dataset,
    main_lens_centres=main_lens_centres,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)
