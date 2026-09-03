"""
SLaM (Source, Light and Mass): DSPL
===================================

This script adapts the SLaM (Source, Light and Mass) pipelines to a Double Source Plane Lens (DSPL) system, where
there are two source galaxies at different redshifts behind the lens galaxy. Both source galaxies must be modeled
simultaneously because the first source (at the intermediate redshift) acts as both a background source of the lens
galaxy and as an additional deflector for the second (higher-redshift) source.

This script is the DSPL analogue of `guides/modeling/slam_start_here.py`. It follows the same API conventions —
each pipeline stage is a plain inline Python function, priors are chained via `al.util.chaining.mass_from`, image
positions are derived automatically via `positions_likelihood_from`, and MGE light profiles are constructed via
`al.model_util.mge_model_from`.

__DSPL-Specific Differences From Standard SLaM__

 - There are two source galaxies (`source_0` at redshift 1.0, `source_1` at redshift 2.0). `source_0` is a light
   source AND a mass deflector for `source_1`.
 - The SOURCE LP PIPELINE is split into two searches. The first fits lens + `source_0` only, providing a stable
   starting point. The second frees `source_0`'s mass and adds `source_1`'s light.
 - The SOURCE PIX PIPELINE has an extra search: one pixelizes `source_0` while `source_1` is a bare ray-tracing
   galaxy, and the next pixelizes `source_1` with `source_0`'s mass fixed from the previous search.
 - Two `PositionsLH` likelihoods are used once both sources are active, one per source-plane redshift.
 - Adapt images are stitched across pipeline stages (e.g. `source_0`'s adapt image comes from a different result
   than `source_1`'s).

__This Script__

Using a SOURCE LP PIPELINE and SOURCE PIX PIPELINE, this DSPL SLaM modeling script fits an `Imaging` dataset of a
double source-plane lens (DSPL) where in the final model:

 - The lens galaxy's light is a bulge with MGE light profile.
 - The lens galaxy's total mass distribution is an `Isothermal` plus an `ExternalShear`.
 - The first source galaxy's light is a `Pixelization` and its mass is an `Isothermal`.
 - The second source galaxy's light is a `Pixelization`.

Optional LIGHT LP and MASS TOTAL stages are left as a follow-up exercise.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__SOURCE LP PIPELINE 1__

The first SOURCE LP PIPELINE search initializes a model where `source_1` is ignored and only the lens galaxy and
`source_0` are fit. This single-plane fit provides robust initial priors for the lens light, lens mass, shear and
`source_0` light before the more complex DSPL model is introduced.

The model:
 - Lens light: MGE with 2 x 20 Gaussians.
 - Lens mass: `Isothermal` + `ExternalShear`.
 - `source_0` light: MGE with 1 x 20 Gaussians.
"""


def source_lp_1(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    redshift_lens: float,
    redshift_source_0: float,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    source_0_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=False,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                mass=af.Model(al.mp.Isothermal),
                shear=af.Model(al.mp.ExternalShear),
            ),
            source_0=af.Model(
                al.Galaxy,
                redshift=redshift_source_0,
                bulge=source_0_bulge,
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
__SOURCE LP PIPELINE 2__

The second SOURCE LP PIPELINE search introduces the second source galaxy. The lens bulge / mass / shear and
`source_0` light are fixed to the instance values from search 1, and we free:

 - `source_0`'s mass: `Isothermal` with a prior tightly centred on the origin (the first source typically sits
   close to the lens centre).
 - `source_1`'s light: MGE with 1 x 20 Gaussians.

A `PositionsLH` for `source_0` is attached to the analysis — automatically derived from the search-1 result via
`positions_likelihood_from` — to prevent unphysical mass models during the DSPL fit. No `source_1` positions are
available yet (this is the first search that fits `source_1`); they are introduced from the pixelized results
further down the pipeline.
"""


def source_lp_2(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    source_lp_result_1: af.Result,
    redshift_source_1: float,
    n_batch: int = 30,
) -> af.Result:
    positions_likelihood_source_0 = source_lp_result_1.positions_likelihood_from(
        factor=3.0,
        minimum_threshold=0.3,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[positions_likelihood_source_0],
    )

    source_0_mass = af.Model(al.mp.Isothermal)
    source_0_mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source_0_mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source_0_mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    source_1_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=False,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result_1.instance.galaxies.lens.redshift,
                bulge=source_lp_result_1.instance.galaxies.lens.bulge,
                mass=source_lp_result_1.instance.galaxies.lens.mass,
                shear=source_lp_result_1.instance.galaxies.lens.shear,
            ),
            source_0=af.Model(
                al.Galaxy,
                redshift=source_lp_result_1.instance.galaxies.source_0.redshift,
                bulge=source_lp_result_1.instance.galaxies.source_0.bulge,
                mass=source_0_mass,
            ),
            source_1=af.Model(
                al.Galaxy,
                redshift=redshift_source_1,
                bulge=source_1_bulge,
            ),
        ),
    )

    search = af.Nautilus(
        name="source_lp[2]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1 — source_0__

Pixelizes `source_0` while `source_1` is present only as a bare galaxy for ray-tracing purposes. The lens mass is
freed with priors initialized from the SOURCE LP PIPELINE result, and `source_1` is not fit (no light, no mass)
so this search constrains the lens mass via `source_0` alone.

Adapt images come from the SOURCE LP PIPELINE result 1 (the single-plane fit).

__Adapt Image S/N Cap__

The source adapt image is capped at a signal-to-noise of 3.0 before it is used by the adaptive
image-mesh and the adaptive regularization. Without the cap the brightest peak dominates the
weights (they scale as a power of the adapt image), so fainter multiply-imaged features get too
few source pixels and too little regularization weight. Capping makes every feature above S/N 3.0
count equally. The cap is applied to an explicit copy so the raw S/N image is untouched.
"""


def source_pix_1_source_0(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result_1: af.Result,
    source_lp_result_2: af.Result,
    redshift_source_1: float,
    mesh_init,
    regularization_init,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result_1
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_0_adapt_image = galaxy_image_name_dict["('galaxies', 'source_0')"].copy()
    source_0_adapt_image[source_0_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_image_name_dict["('galaxies', 'source_0')"] = source_0_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    positions_likelihood_source_0 = source_lp_result_1.positions_likelihood_from(
        factor=3.0,
        minimum_threshold=0.2,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood_source_0],
    )

    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.Isothermal),
        mass_result=source_lp_result_2.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    shear = source_lp_result_2.model.galaxies.lens.shear

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result_2.instance.galaxies.lens.redshift,
                bulge=source_lp_result_2.instance.galaxies.lens.bulge,
                mass=mass,
                shear=shear,
            ),
            source_0=af.Model(
                al.Galaxy,
                redshift=source_lp_result_2.instance.galaxies.source_0.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
            source_1=af.Model(
                al.Galaxy,
                redshift=redshift_source_1,
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]_source_0",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1 — source_1__

Pixelizes `source_1`. `source_0`'s mass is freed with priors initialized from the previous pixelized search, so
the second source plane constrains the lensing by the first source. The lens mass is fixed from
`source_pix_result_1_source_0`.

Two `PositionsLH` are attached — one per source plane — to prevent unphysical reconstructions.

Adapt images are stitched: the lens and `source_1` adapt images come from the LP pipeline result 2 (this is
`source_1`'s first pixelized fit, so its adapt image is seeded from its light-profile model image in the LP
result); `source_0`'s adapt image comes from the pixelized search above. All three are required because
`regularization_init` is adaptive (`al.reg.Adapt`) — every pixelized galaxy needs an adapt image.
"""


def source_pix_1_source_1(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result_2: af.Result,
    source_pix_result_1_source_0: af.Result,
    mesh_init,
    regularization_init,
    n_batch: int = 20,
) -> af.Result:
    lp2_dict = al.galaxy_name_image_dict_via_result_from(result=source_lp_result_2)
    pix_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1_source_0
    )

    galaxy_name_image_dict = {
        "('galaxies', 'lens')": lp2_dict["('galaxies', 'lens')"],
        "('galaxies', 'source_0')": pix_dict["('galaxies', 'source_0')"],
        "('galaxies', 'source_1')": lp2_dict["('galaxies', 'source_1')"],
    }

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_0_adapt_image = galaxy_name_image_dict["('galaxies', 'source_0')"].copy()
    source_0_adapt_image[source_0_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_name_image_dict["('galaxies', 'source_0')"] = source_0_adapt_image

    source_1_adapt_image = galaxy_name_image_dict["('galaxies', 'source_1')"].copy()
    source_1_adapt_image[source_1_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_name_image_dict["('galaxies', 'source_1')"] = source_1_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

    positions_likelihood_source_0 = (
        source_pix_result_1_source_0.positions_likelihood_from(
            factor=3.0,
            minimum_threshold=0.2,
            plane_redshift=source_lp_result_2.instance.galaxies.source_0.redshift,
        )
    )
    positions_likelihood_source_1 = source_lp_result_2.positions_likelihood_from(
        factor=3.0,
        minimum_threshold=0.2,
        plane_redshift=source_lp_result_2.instance.galaxies.source_1.redshift,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            positions_likelihood_source_0,
            positions_likelihood_source_1,
        ],
    )

    source_0_mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.Isothermal),
        mass_result=source_lp_result_2.model.galaxies.source_0.mass,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result_2.instance.galaxies.lens.redshift,
                bulge=source_lp_result_2.instance.galaxies.lens.bulge,
                mass=source_pix_result_1_source_0.instance.galaxies.lens.mass,
                shear=source_pix_result_1_source_0.instance.galaxies.lens.shear,
            ),
            source_0=af.Model(
                al.Galaxy,
                redshift=source_lp_result_2.instance.galaxies.source_0.redshift,
                mass=source_0_mass,
            ),
            source_1=af.Model(
                al.Galaxy,
                redshift=source_lp_result_2.instance.galaxies.source_1.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]_source_1",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

The final SOURCE PIX PIPELINE search fits both source galaxies simultaneously with adaptive pixelizations.
Lens mass, shear and `source_0`'s mass are all fixed to the maximum-likelihood instances of the previous
pixelized searches; only the pixelization regularization parameters are free.

The `RectangularBilinearAdaptImage` (or equivalent) mesh uses the high-quality adapt images built up over the earlier
pipeline stages to adapt each source-plane pixelization to its reconstructed morphology.
"""


def source_pix_2(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result_2: af.Result,
    source_pix_result_1_source_0: af.Result,
    source_pix_result_1_source_1: af.Result,
    mesh,
    regularization,
    n_batch: int = 20,
) -> af.Result:
    lp2_dict = al.galaxy_name_image_dict_via_result_from(result=source_lp_result_2)
    pix0_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1_source_0
    )
    pix1_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1_source_1
    )

    galaxy_name_image_dict = {
        "('galaxies', 'lens')": lp2_dict["('galaxies', 'lens')"],
        "('galaxies', 'source_0')": pix0_dict["('galaxies', 'source_0')"],
        "('galaxies', 'source_1')": pix1_dict["('galaxies', 'source_1')"],
    }

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_0_adapt_image = galaxy_name_image_dict["('galaxies', 'source_0')"].copy()
    source_0_adapt_image[source_0_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_name_image_dict["('galaxies', 'source_0')"] = source_0_adapt_image

    source_1_adapt_image = galaxy_name_image_dict["('galaxies', 'source_1')"].copy()
    source_1_adapt_image[source_1_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_name_image_dict["('galaxies', 'source_1')"] = source_1_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result_2.instance.galaxies.lens.redshift,
                bulge=source_lp_result_2.instance.galaxies.lens.bulge,
                mass=source_pix_result_1_source_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1_source_1.instance.galaxies.lens.shear,
            ),
            source_0=af.Model(
                al.Galaxy,
                redshift=source_lp_result_2.instance.galaxies.source_0.redshift,
                mass=source_pix_result_1_source_1.instance.galaxies.source_0.mass,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh,
                    regularization=regularization,
                ),
            ),
            source_1=af.Model(
                al.Galaxy,
                redshift=source_lp_result_2.instance.galaxies.source_1.redshift,
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
__Dataset__

Load, plot and mask the `Imaging` data.
"""
dataset_name = "double_source_plane_lens"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/imaging/features/advanced/double_source_plane_lens/simulator.py",
        ],
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
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("imaging") / "slam_dspl",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens galaxy and the two source galaxies. Multi-plane ray-tracing uses these to compute the
correct deflection angles between each plane.
"""
redshift_lens = 0.5
redshift_source_0 = 1.0
redshift_source_1 = 2.0

"""
__Mesh Shape__

The pixelization mesh shape is fixed before modeling; see `features/pixelization/modeling` for details.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__SLaM Pipeline__

The code below runs the full DSPL SLaM pipeline. See the docstring above each function for a description of
each stage.
"""
source_lp_result_1 = source_lp_1(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
    redshift_source_0=redshift_source_0,
)

source_lp_result_2 = source_lp_2(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    source_lp_result_1=source_lp_result_1,
    redshift_source_1=redshift_source_1,
)

source_pix_result_1_source_0 = source_pix_1_source_0(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result_1=source_lp_result_1,
    source_lp_result_2=source_lp_result_2,
    redshift_source_1=redshift_source_1,
    mesh_init=af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape),
    regularization_init=al.reg.Adapt,
)

source_pix_result_1_source_1 = source_pix_1_source_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result_2=source_lp_result_2,
    source_pix_result_1_source_0=source_pix_result_1_source_0,
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

This lens has two source planes; the threshold is applied to the first source plane's (`source_0`)
signal-to-noise map, from its pixelized fit in SOURCE PIX PIPELINE 1.

SOURCE PIX PIPELINE 1 keeps the dataset's default uniform sub-size. Its adapt image comes from the parametric
source fit of the SOURCE LP PIPELINE, which does not yet trace the lensed source well enough to steer
over-sampling.
"""
signal_to_noise_threshold = 3.0

source_image_raw = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1_source_0
)["('galaxies', 'source_0')"]

over_sample_size_pixelization = al.Array2D(
    values=np.where(source_image_raw > signal_to_noise_threshold, 4, 2),
    mask=dataset.mask,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=over_sample_size_pixelization
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result_2=source_lp_result_2,
    source_pix_result_1_source_0=source_pix_result_1_source_0,
    source_pix_result_1_source_1=source_pix_result_1_source_1,
    mesh=af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape),
    regularization=al.reg.Adapt,
)
