"""
SLaM (Source, Light and Mass): Stellar and Dark Mass (Multi Galaxy)
===================================================================

The SLaM pipeline for a multi-galaxy lens whose deflectors' mass is decomposed into stellar and dark components.

Read `multi_galaxy/slam.py` first — the multi-galaxy SLaM baseline — and `guides/modeling/slam_start_here`
before it. This script documents only what the decomposition changes.

__What Changes__

One stage. The baseline's five become five, not six: `MASS TOTAL` is replaced by `MASS LIGHT DARK`.

Everything before it is the baseline's, unchanged — `source_lp[1]`, both SOURCE PIX stages and `light[1]` all
fit a total mass model, because the decomposition is not tractable until the light and source are settled. The
stage functions are copied rather than imported; `multi_galaxy/slam.py` is a script, so importing it would
execute its whole pipeline as a side effect.

The final stage swaps each deflector's `lp_linear.Sersic` bulge for an `lmp.Sersic` — light and stellar mass
coupled through a `mass_to_light_ratio` — carrying the geometry across with `take_attributes`, and adds an
`NFWSph` halo per galaxy.

__Why light[1] Matters More Here__

In the baseline, `light[1]` produces an accurate light model and the mass stage that follows is independent of
it. Here they are not independent: the stellar mass **is** the light, scaled. An error in a deflector's light
model propagates directly into its stellar mass and therefore into the decomposition.

This is the strongest version of the argument `multi_galaxy/slam.py` already makes for that stage.

__The Mass-To-Light Ratio Is Tied__

The two deflectors share one `mass_to_light_ratio` in the final stage, for the reason
`multi_galaxy/features/advanced/mass_stellar_dark/modeling.py` gives: the two galaxies' ratios are
near-degenerate with each other, and tying them removes that direction rather than leaving the search to explore
it. Their dark halos are not tied.

__An API Note__

`al.util.chaining.mass_light_dark_from` reads `light_result.instance.galaxies.lens.<name>` — a hardcoded
single-lens path. It cannot be used for a model with `lens_0`, `lens_1`, ..., so the final stage builds each
galaxy's decomposition by hand. The group pipeline does the same.

__Contents__

- **Helpers:** The deflector-count helper the baseline uses.
- **Source LP Pipeline:** Light and mass per deflector, and the source.
- **Source Pix Pipeline 1 & 2:** The pixelized source.
- **Light LP Pipeline:** A fresh MGE per deflector — load-bearing here.
- **Mass Light Dark Pipeline:** The decomposition.
- **Dataset, Centres, Mask:** Set up.
- **Mesh Shape:** The pixelization classes.
- **SLaM Pipeline:** Run the five stages in order.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


def n_main_from(result) -> int:
    """
    The number of co-dominant deflectors in a result's model. Identical to the helper in `multi_galaxy/slam.py`.
    """
    return sum(1 for key in vars(result.instance.galaxies) if key.startswith("lens_"))


"""
__SOURCE LP PIPELINE__

Identical to `multi_galaxy/slam.py`. A total mass model per deflector, with centres fixed and Einstein radii
capped.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    main_lens_centres,
    redshift_lens: float,
    redshift_source: float,
    upper_einstein_radius: float = 3.0,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_dict = {}

    for i, centre in enumerate(main_lens_centres):

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
            centre=(centre[0], centre[1]),
            centre_sigma=0.1,
            sigma_min=dataset.pixel_scales[0] / 10.0,
        )

        mass = af.Model(al.mp.Isothermal)
        mass.centre = (centre[0], centre[1])
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0, upper_limit=upper_einstein_radius
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=bulge,
            disk=None,
            point=None,
            mass=mass,
        )

    shear_galaxy = af.Model(
        al.Galaxy,
        redshift=redshift_lens,
        shear=af.Model(al.mp.ExternalShear),
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=shear_galaxy,
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
        ),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=150 + 50 * len(lens_dict),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Identical to `multi_galaxy/slam.py`: the first pixelized source, producing a better adapt image, with the mass
centres released.

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
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    lens_dict = {}

    for i in range(n_main_from(source_lp_result)):

        lens_instance = getattr(source_lp_result.instance.galaxies, f"lens_{i}")
        lens_model = getattr(source_lp_result.model.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.Isothermal),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=lens_instance.bulge,
            disk=lens_instance.disk,
            point=lens_instance.point,
            mass=mass,
        )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_lp_result.model.galaxies.shear_galaxy,
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
        n_live=150 + 50 * (n_main_from(source_lp_result) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Identical to `multi_galaxy/slam.py`: the final pixelized source on the improved adapt image, everything else
fixed.
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
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    lens_dict = {}

    for i in range(n_main_from(source_pix_result_1)):

        lp_instance = getattr(source_lp_result.instance.galaxies, f"lens_{i}")
        pix_instance = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lp_instance.redshift,
            bulge=lp_instance.bulge,
            disk=lp_instance.disk,
            point=lp_instance.point,
            mass=pix_instance.mass,
        )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_pix_result_1.instance.galaxies.shear_galaxy,
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

Identical to `multi_galaxy/slam.py` in code, and more load-bearing here than in any other pipeline in the
package.

The stellar mass fitted in the next stage **is** this stage's light, scaled by a mass-to-light ratio. An error
in a deflector's light model does not merely leave residuals — it propagates straight into that galaxy's stellar
mass, and from there into the decomposition the pipeline exists to produce.
"""


def light_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

    lens_dict = {}

    for i in range(n_main_from(source_result_for_lens)):

        lens_instance = getattr(source_result_for_lens.instance.galaxies, f"lens_{i}")

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
            centre=tuple(lens_instance.mass.centre),
            sigma_min=dataset.pixel_scales[0] / 10.0,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=bulge,
            disk=None,
            point=None,
            mass=lens_instance.mass,
        )

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.instance.galaxies.shear_galaxy,
            source=source,
        ),
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150 + 100 * (n_main_from(source_result_for_lens) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS LIGHT DARK PIPELINE__

The terminal stage, replacing the baseline's `MASS TOTAL`.

Each deflector's bulge is swapped from `light[1]`'s MGE to an `lmp.Sersic`, so its light and stellar mass are
one profile. An `NFWSph` halo is added per galaxy, centred on the same point. The source is fixed from
`source_pix[2]`.

The two galaxies' `mass_to_light_ratio` values are tied. Their halos are not.

`al.util.chaining.mass_light_dark_from` cannot be used — it reads a hardcoded single-lens path — so the
composition is built by hand, as the group pipeline also does.
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
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

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

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=light_result.instance.galaxies.lens_0.redshift,
            bulge=bulge,
            dark=dark,
        )

    # Tie the mass-to-light ratio across every deflector.

    for i in range(1, len(lens_dict)):
        lens_dict[f"lens_{i}"].bulge.mass_to_light_ratio = lens_dict[
            "lens_0"
        ].bulge.mass_to_light_ratio

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.model.galaxies.shear_galaxy,
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_light_dark[1]",
        **settings_search.search_dict,
        n_live=250,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

The `mass_stellar_dark` multi-galaxy dataset.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/mass_stellar_dark/simulator.py",
        ],
        check=True,
    )

pixel_scale = 0.05

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=pixel_scale,
)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.

The radius matters more here than in the baseline: the stellar and dark components separate in the outskirts,
where their profiles diverge, so a tighter mask removes the pixels the final stage depends on.
"""
mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("multi_galaxy")
    / "features"
    / "advanced"
    / "mass_stellar_dark"
    / "slam",
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

The pixelization classes used by the SOURCE PIX stages, as in `multi_galaxy/slam.py`.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mesh_init = af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape)
regularization_init = al.reg.Adapt

mesh = af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape)
regularization = al.reg.Adapt

"""
__SLaM Pipeline__

The five stages, in order.
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    main_lens_centres=main_lens_centres,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    mesh_init=mesh_init,
    regularization_init=regularization_init,
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
signal_to_noise_threshold = 3.0

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)
source_image_raw = galaxy_image_name_dict["('galaxies', 'source')"]

over_sample_size_pixelization = al.Array2D(
    values=np.where(source_image_raw > signal_to_noise_threshold, 4, 2),
    mask=dataset.mask,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=over_sample_size_pixelization,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=mesh,
    regularization=regularization,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
)

mass_result = mass_light_dark(
    settings_search=settings_search,
    dataset=dataset,
    main_lens_centres=main_lens_centres,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)

"""
__Result__

`mass_result` holds the decomposed model.

Beyond the multi-galaxy checks in `multi_galaxy/slam.py`, the decomposition-specific ones are in
`multi_galaxy/features/advanced/mass_stellar_dark/modeling.py`'s `__Result__` section: the components' relative
contributions per galaxy, the shared mass-to-light ratio against its prior bounds, and the two dark halos
against each other.

One more is specific to running this as a pipeline: compare each deflector's total mass here against its
`Isothermal` from `source_pix[1]`. The decomposition should divide the same total, not find a different one.
"""
print(mass_result.info)

aplt.subplot_fit_imaging(fit=mass_result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/slam.py` — the baseline pipeline these stages are copied from.
 - `multi_galaxy/features/advanced/mass_stellar_dark/chaining.py` — the same total-then-decompose idea in two
   searches rather than five.
 - `multi_galaxy/features/advanced/mass_stellar_dark/modeling.py` — the decomposition and the tying choice.
 - `guides/modeling/slam_start_here` — what each stage is for, in full.
"""
