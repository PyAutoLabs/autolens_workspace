"""
SLaM (Source, Light and Mass): No Lens Light (Multi Galaxy)
===========================================================

The SLaM pipeline for a multi-galaxy lens whose co-dominant deflectors have no visible light.

This script documents only how it differs from `multi_galaxy/slam.py`, the multi-galaxy SLaM baseline, which in
turn documents only how *it* differs from `guides/modeling/slam_start_here`. Read those first if the stage
structure is unfamiliar.

__Contents__

- **What Changes:** The three differences from `multi_galaxy/slam.py`.
- **Source LP Pipeline:** Mass and source only — no light to fit.
- **Source Pix Pipeline 1:** A pixelized source used to build a high-quality adapt image.
- **Source Pix Pipeline 2:** The final pixelized source, on the improved adapt image.
- **Mass Total Pipeline:** Each deflector's mass promoted to a `PowerLaw`.
- **Dataset, Centres, Mask:** Set up, and why the centres are the whole problem.
- **SLaM Pipeline:** Run the four stages in order.

__Prerequisites__

Read `guides/modeling/slam_start_here` first: it describes what the five SLaM stages are and why they are chained
in this order. This script documents only what differs.

__What Changes__

1. **The LIGHT LP pipeline is gone.** There is no lens light to fit, so the stage that fits it has nothing to do.
   Four stages instead of five, and `mass_total` chains its light from nowhere — each `lens_i` simply has no
   `bulge`.

2. **Mass centres are anchored on external information, and never fixed to light.** `multi_galaxy/slam.py` fixes
   each mass centre to its galaxy's light in `source_lp[1]` and releases it in `source_pix[1]`. Here there is no
   light to fix to. Each centre instead carries a `GaussianPrior` from the start, centred on the value in
   `main_lens_centres.json` — which, as `modeling.py` in this folder explains at length, came from another band, a
   catalogue, or the subtraction that produced this image. That prior's width is a model assumption; set it to
   reflect how good your external astrometry actually is.

3. **The adapt images come from the source alone.** In the baseline pipeline the adapt image is a
   lens-light-subtracted image. With no lens light there is nothing to subtract — the data *is* the lensed source.
   This is the one respect in which removing the lens light makes a SLaM pipeline more robust rather than less:
   the usual failure mode where an imperfect lens-light subtraction corrupts the adapt image cannot occur.

__What This Pipeline Cannot Do__

`multi_galaxy/slam.py` describes `light[1]` as carrying the measurement, because the ratio of the two deflectors'
luminosities is frequently what the science wants. That measurement is simply unavailable here. If you need it,
you need the band in which the galaxies are detected — and at that point you are running the baseline pipeline on
that band, not this one.

What remains available, and is the reason to run this at all, is the mass model: each deflector's Einstein radius
and density slope, and the split between them.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


def n_main_from(result) -> int:
    """
    The number of co-dominant deflectors in a result's model, recovered from the `lens_` key prefix so the
    pipeline runs unchanged on any number of them. Identical to the helper in `multi_galaxy/slam.py`.
    """
    return sum(1 for key in vars(result.instance.galaxies) if key.startswith("lens_"))


"""
__SOURCE LP PIPELINE__

The baseline's `source_lp[1]` fits each deflector's light, each deflector's mass, and the source together. Here it
fits mass and source only.

That makes this stage both smaller and harder. Smaller, because the two MGE light models are gone. Harder, because
the light was doing more than adding parameters: it told the search where each deflector was and how bright it was,
which is a strong constraint on which galaxy is which. Without it, the arcs alone have to separate the deflectors,
and the anchored centre priors are the only thing preventing both mass profiles from drifting onto the same place.

The capped `einstein_radius` prior from the baseline matters more here for the same reason.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    main_lens_centres,
    redshift_lens: float,
    redshift_source: float,
    centre_sigma: float = 0.1,
    upper_einstein_radius: float = 3.0,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_dict = {}

    for i, centre in enumerate(main_lens_centres):

        mass = af.Model(al.mp.Isothermal)
        mass.centre.centre_0 = af.GaussianPrior(mean=centre[0], sigma=centre_sigma)
        mass.centre.centre_1 = af.GaussianPrior(mean=centre[1], sigma=centre_sigma)
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0, upper_limit=upper_einstein_radius
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
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

Identical to `multi_galaxy/slam.py` apart from the absent `bulge`, and one thing that is *not* absent: the mass
centres are already free, so they are chained forward as models rather than being unfixed.

`unfix_mass_centre=True` is still passed. It is a no-op on a centre that was never fixed, and leaving it in keeps
this stage a line-for-line match with the baseline, which is easier to diff than a cleverer version would be.

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

Identical to `multi_galaxy/slam.py` apart from the absent `bulge`: the final pixelized source, fitted on the
improved adapt image from search 1 with every deflector's mass fixed.
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

Not present. There is no lens light to fit.

This is worth stating rather than silently omitting, because the stage numbering of the SLaM output folders will
look wrong to anyone comparing against the baseline: `light[1]` never appears, and `mass_total[1]` follows
`source_pix[2]` directly.

__MASS TOTAL PIPELINE__

Identical to `multi_galaxy/slam.py` apart from the absent light instance: each deflector's mass is promoted from
`Isothermal` to `PowerLaw`, with priors initialized from `source_pix[1]` and the source fixed from
`source_pix[2]`.

The baseline's warning applies with more force here. Promoting to a `PowerLaw` adds one slope per deflector; each
slope is degenerate with that galaxy's Einstein radius, which is degenerate with the other galaxy's — and in this
pipeline the centres are free too, with no light anchoring them. That is three coupled degeneracies feeding each
other. Inspect the posterior as a whole rather than reading any parameter in isolation, and be prepared for the
slopes to be poorly constrained.
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
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

    for i in range(n_main_from(source_result_for_lens)):

        lens_model = getattr(source_result_for_lens.model.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.PowerLaw),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_model.redshift,
            mass=mass,
        )

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.model.galaxies.shear_galaxy,
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150 + 100 * (n_main_from(source_result_for_lens) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

The `simple__no_lens_light` dataset: the `simple` lens with the deflectors' light removed and nothing else changed.

There is no `__Extra Galaxies Noise Scaling__` step here, unlike `multi_galaxy/slam.py`. This dataset contains no
contaminating galaxy — see the simulator's `__No Extra Galaxy__` note for why.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/features/no_lens_light/simulator.py"],
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

The deflector centres, which drive the loop in every stage and set the `GaussianPrior` means in `source_lp[1]`.

In `multi_galaxy/slam.py` this file is a convenience — the galaxies are visible and the pipeline could find them.
Here it is the pipeline's only information about where the deflectors are, and every mass measurement downstream
inherits whatever error it carries. `modeling.py` in this folder covers where to get it for real data.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, sized by the combined Einstein radius (~1.8").

Note the absent step: `multi_galaxy/slam.py` applies adaptive over-sampling centred on every deflector, because
each has a steep central light profile that must be evaluated accurately. No deflector has a light profile here, so
that is unnecessary and the default over-sampling is used.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("multi_galaxy") / "features" / "no_lens_light" / "slam",
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

mesh_init = af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape)
regularization_init = al.reg.Adapt

mesh = af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape)
regularization = al.reg.Adapt

"""
__SLaM Pipeline__

Four stages, not five.
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

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=mesh,
    regularization=regularization,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
)

"""
__Result__

The checks from `multi_galaxy/slam.py` apply, plus one specific to this pipeline: compare each inferred mass centre
against the input value from `main_lens_centres.json`.

If the centres have moved to the edges of their priors, the arcs disagree with your external astrometry. That is
information, not an error — but do not quote a mass model built on a prior the data is fighting. Widen it and
refit.
"""
print(mass_result.info)

aplt.subplot_fit_imaging(fit=mass_result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/slam.py` — the baseline this pipeline is a subset of.
 - `multi_galaxy/features/no_lens_light/modeling.py` — the single-search fit, and the full discussion of where the
   deflector centres come from.
 - `imaging/features/no_lens_light/slam.py` — the galaxy-scale version.
"""
