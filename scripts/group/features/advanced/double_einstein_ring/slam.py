"""
SLaM (Source, Light and Mass): Group Double Einstein Ring
=========================================================

This script adapts the SLaM (Source, Light and Mass) pipelines to a group-scale Double Source Plane Lens (DSPL)
system, where two source galaxies at different redshifts behind multiple main lens galaxies form a double
Einstein ring.

This script is the group + DSPL analogue of `guides/modeling/slam_start_here.py` and
`scripts/imaging/features/advanced/double_einstein_ring/slam.py`. Each pipeline stage is a plain inline Python
function, priors are chained via `al.util.chaining.mass_from`, image positions are derived automatically via
`positions_likelihood_from`, and MGE light profiles are constructed via `al.model_util.mge_model_from`.

__Group + DSPL-Specific Differences From Standard SLaM__

 - The lens-plane (z=0.5) is composed via the group `lens_dict` convention: one `af.Model(al.Galaxy)` entry per
   main lens galaxy centre, with the `ExternalShear` attached only to `lens_0`.
 - There are two source galaxies (`source_0` at redshift 1.0, `source_1` at redshift 2.0). `source_0` is a light
   source AND a mass deflector for `source_1`.
 - The SOURCE LP PIPELINE is split into two searches: the first fits all main lens galaxies + `source_0` only;
   the second frees `source_0`'s mass and adds `source_1`'s light.
 - The SOURCE PIX PIPELINE has an extra search: one pixelizes `source_0` while `source_1` is a bare ray-tracing
   galaxy, and the next pixelizes `source_1` with `source_0`'s mass fixed from the previous search.
 - Two `PositionsLH` likelihoods are used once both sources are active, one per source-plane redshift.
 - Adapt images are stitched across pipeline stages and across multiple main lens galaxies.

__This Script__

Using a SOURCE LP PIPELINE and SOURCE PIX PIPELINE, this DSPL group SLaM modeling script fits an `Imaging`
dataset of a group-scale double Einstein ring system where in the final model:

 - Each main lens galaxy's light is a bulge with an MGE light profile.
 - Each main lens galaxy's total mass distribution is an `Isothermal`. `lens_0` carries an `ExternalShear`.
 - The first source galaxy's light is a `Pixelization` and its mass is an `Isothermal`.
 - The second source galaxy's light is a `Pixelization`.

Optional LIGHT LP and MASS TOTAL stages are left as a follow-up exercise.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__Helpers__

`build_lens_dict_model` constructs a dictionary of `af.Model(al.Galaxy)` entries — one per main lens galaxy
centre — with each galaxy's bulge built from `mge_model_from` and a free `Isothermal` mass profile. The shear
is added to `lens_0` only, mirroring the canonical group convention.

`lens_dict_instance` and `lens_dict_model` extract per-lens result objects from a search result for forward
passing to the next stage.
"""


def build_lens_dict_model(
    main_lens_centres,
    redshift_lens: float,
    mask_radius: float,
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
        )

        mass = af.Model(al.mp.Isothermal)
        mass.centre = (centre[0], centre[1])

        kwargs = dict(
            redshift=redshift_lens,
            bulge=bulge,
            mass=mass,
        )
        if i == 0:
            kwargs["shear"] = af.Model(al.mp.ExternalShear)

        lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, **kwargs)
    return lens_dict


def lens_dict_from_result(result, n_lens: int, *, attr: str = "instance"):
    """Return a dict of lens-galaxy result objects (instance or model) keyed by `lens_{i}`."""
    galaxies = getattr(result, attr).galaxies
    return {f"lens_{i}": getattr(galaxies, f"lens_{i}") for i in range(n_lens)}


def galaxy_image_dict_for_all_lenses(result, n_lens: int):
    """Stitch per-lens galaxy-name image-dict entries from `galaxy_name_image_dict_via_result_from`."""
    full = al.galaxy_name_image_dict_via_result_from(result=result)
    out = {}
    for i in range(n_lens):
        key = f"('galaxies', 'lens_{i}')"
        out[key] = full[key]
    return out


"""
__SOURCE LP PIPELINE 1__

The first SOURCE LP PIPELINE search initializes a model where `source_1` is ignored and only the main lens
galaxies and `source_0` are fit. This single-plane fit provides robust initial priors for each main lens
galaxy's light, mass, the external shear on `lens_0`, and `source_0`'s light before the more complex DSPL model
is introduced.

Model:
 - Per main lens galaxy: MGE bulge (2 x 20 Gaussians), `Isothermal` mass. `lens_0` also has an `ExternalShear`.
 - `source_0` light: MGE with 1 x 20 Gaussians.
 - `source_1`: absent.
"""


def source_lp_1(
    settings_search: af.SettingsSearch,
    dataset,
    main_lens_centres,
    mask_radius: float,
    redshift_lens: float,
    redshift_source_0: float,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_dict = build_lens_dict_model(
        main_lens_centres=main_lens_centres,
        redshift_lens=redshift_lens,
        mask_radius=mask_radius,
    )

    source_0_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=False,
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
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

The second SOURCE LP PIPELINE search introduces `source_1`. Each main lens galaxy's bulge, mass and (for
`lens_0`) shear are fixed to the search-1 instance values, and `source_0`'s light is also fixed. New free
parameters:

 - `source_0`'s mass: `Isothermal` with a narrow prior centred near the origin (the first source typically sits
   close to the lens centroid).
 - `source_1`'s light: MGE with 1 x 20 Gaussians.

A `PositionsLH` for `source_0` is attached to the analysis to prevent unphysical mass models during the DSPL
fit.
"""


def source_lp_2(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result_1: af.Result,
    mask_radius: float,
    redshift_source_1: float,
    n_batch: int = 30,
) -> af.Result:
    n_lens = len(
        [
            name
            for name in vars(source_lp_result_1.instance.galaxies)
            if name.startswith("lens_")
        ]
    )

    positions_likelihood_source_0 = source_lp_result_1.positions_likelihood_from(
        factor=3.0,
        minimum_threshold=0.3,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[positions_likelihood_source_0],
        use_jax=True,
    )

    lens_dict = {}
    for i in range(n_lens):
        lens_inst = getattr(source_lp_result_1.instance.galaxies, f"lens_{i}")
        kwargs = dict(
            redshift=lens_inst.redshift,
            bulge=lens_inst.bulge,
            mass=lens_inst.mass,
        )
        if i == 0:
            kwargs["shear"] = lens_inst.shear

        lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, **kwargs)

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
            **lens_dict,
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

Pixelizes `source_0` while `source_1` is present only as a bare galaxy for ray-tracing purposes. Each main
lens galaxy's mass is freed with priors initialized from the SOURCE LP PIPELINE result 2, and `source_1` is not
fit (no light, no mass).

Adapt images for each main lens galaxy come from the SOURCE LP PIPELINE result 1.
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
    n_lens = len(
        [
            name
            for name in vars(source_lp_result_2.instance.galaxies)
            if name.startswith("lens_")
        ]
    )

    galaxy_name_image_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result_1
    )
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

    positions_likelihood_source_0 = source_lp_result_1.positions_likelihood_from(
        factor=3.0,
        minimum_threshold=0.2,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood_source_0],
        use_jax=True,
    )

    lens_dict = {}
    for i in range(n_lens):
        lens_inst = getattr(source_lp_result_2.instance.galaxies, f"lens_{i}")
        lens_model_mass = getattr(source_lp_result_2.model.galaxies, f"lens_{i}").mass
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
            kwargs["shear"] = source_lp_result_2.model.galaxies.lens_0.shear

        lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, **kwargs)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
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

Pixelizes `source_1`. `source_0`'s mass is freed with priors initialized from the previous pixelized search.
Each main lens galaxy's mass is fixed from `source_pix_result_1_source_0`.

Two `PositionsLH` are attached — one per source plane — to prevent unphysical reconstructions.

Adapt images are stitched: per-lens and `source_1` adapt images come from the LP pipeline result 2 (this is
`source_1`'s first pixelized fit, so its adapt image is seeded from its light-profile model image in the LP
result); `source_0`'s adapt image comes from the pixelized search above. All are required because
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
    n_lens = len(
        [
            name
            for name in vars(source_lp_result_2.instance.galaxies)
            if name.startswith("lens_")
        ]
    )

    lp2_dict = al.galaxy_name_image_dict_via_result_from(result=source_lp_result_2)
    pix0_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1_source_0
    )

    galaxy_name_image_dict = {}
    for i in range(n_lens):
        key = f"('galaxies', 'lens_{i}')"
        galaxy_name_image_dict[key] = lp2_dict[key]
    galaxy_name_image_dict["('galaxies', 'source_0')"] = pix0_dict[
        "('galaxies', 'source_0')"
    ]
    galaxy_name_image_dict["('galaxies', 'source_1')"] = lp2_dict[
        "('galaxies', 'source_1')"
    ]

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
        use_jax=True,
    )

    source_0_mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.Isothermal),
        mass_result=source_lp_result_2.model.galaxies.source_0.mass,
        unfix_mass_centre=True,
    )

    lens_dict = {}
    for i in range(n_lens):
        lens_inst_lp = getattr(source_lp_result_2.instance.galaxies, f"lens_{i}")
        lens_inst_pix = getattr(
            source_pix_result_1_source_0.instance.galaxies, f"lens_{i}"
        )

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
Per-lens mass, shear (on `lens_0`) and `source_0`'s mass are all fixed to the maximum-likelihood instances of
the previous pixelized searches; only the pixelization regularization parameters are free.
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
    n_lens = len(
        [
            name
            for name in vars(source_lp_result_2.instance.galaxies)
            if name.startswith("lens_")
        ]
    )

    lp2_dict = al.galaxy_name_image_dict_via_result_from(result=source_lp_result_2)
    pix0_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1_source_0
    )
    pix1_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1_source_1
    )

    galaxy_name_image_dict = {}
    for i in range(n_lens):
        key = f"('galaxies', 'lens_{i}')"
        galaxy_name_image_dict[key] = lp2_dict[key]
    galaxy_name_image_dict["('galaxies', 'source_0')"] = pix0_dict[
        "('galaxies', 'source_0')"
    ]
    galaxy_name_image_dict["('galaxies', 'source_1')"] = pix1_dict[
        "('galaxies', 'source_1')"
    ]

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    lens_dict = {}
    for i in range(n_lens):
        lens_inst_lp = getattr(source_lp_result_2.instance.galaxies, f"lens_{i}")
        lens_inst_pix = getattr(
            source_pix_result_1_source_1.instance.galaxies, f"lens_{i}"
        )

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

Load, plot and mask the group double Einstein ring `Imaging` dataset.
"""
dataset_name = "double_einstein_ring"
dataset_path = Path("dataset") / "group" / dataset_name

if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/double_einstein_ring/simulator.py",
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

mask_radius = 4.0

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
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("group") / "slam_dspl",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens-plane and the two source galaxies.
"""
redshift_lens = 0.5
redshift_source_0 = 1.0
redshift_source_1 = 2.0

"""
__Mesh Shape__

The pixelization mesh shape is fixed before modeling.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__SLaM Pipeline__

The code below runs the full DSPL group SLaM pipeline. See the docstring above each function for a description
of each stage.
"""
source_lp_result_1 = source_lp_1(
    settings_search=settings_search,
    dataset=dataset,
    main_lens_centres=main_lens_centres,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
    redshift_source_0=redshift_source_0,
)

source_lp_result_2 = source_lp_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result_1=source_lp_result_1,
    mask_radius=mask_radius,
    redshift_source_1=redshift_source_1,
)

source_pix_result_1_source_0 = source_pix_1_source_0(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result_1=source_lp_result_1,
    source_lp_result_2=source_lp_result_2,
    redshift_source_1=redshift_source_1,
    mesh_init=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
    regularization_init=al.reg.Adapt,
)

source_pix_result_1_source_1 = source_pix_1_source_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result_2=source_lp_result_2,
    source_pix_result_1_source_0=source_pix_result_1_source_0,
    mesh_init=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
    regularization_init=al.reg.Adapt,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result_2=source_lp_result_2,
    source_pix_result_1_source_0=source_pix_result_1_source_0,
    source_pix_result_1_source_1=source_pix_result_1_source_1,
    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
    regularization=al.reg.Adapt,
)

"""
Finish.
"""
