"""
Scaling Relation: SLaM (Interferometer)
=======================================

Runs the SLaM pipelines on an interferometer dataset whose foreground companions are tied to the main lens's own
Einstein radius by a Faber-Jackson relation.

**Unlike its CCD-imaging counterpart, this pipeline does not measure the luminosities — it cannot.** That is the point
of this script, and the reason it reads differently from
`imaging/features/scaling_relation/slam.py`, whose whole purpose is that measurement.

__Why This Pipeline Cannot Measure Luminosities__

Two facts combine:

 1. Foreground galaxy light is rarely detected at the long wavelengths an interferometer probes. There is no
    companion emission in the data to fit, so there is nothing to integrate a luminosity from.
 2. Consequently an interferometer SLaM pipeline has **no `light_lp` stage at all** — see
    `interferometer/features/extra_galaxies/slam.py`, whose stages are `source_lp` -> `source_pix_1` ->
    `source_pix_2` -> `mass_total`. The imaging pipeline's luminosity measurement lives in exactly the stage this
    pipeline does not have.

So the luminosities are **inputs** here, measured from ancillary optical or near-infrared imaging of the same field —
the same imaging that gave you the companions' centres, since visibilities give you neither. This script threads those
external numbers through the pipeline and re-applies the tie as the anchor's mass profile changes.

If that sounds like a weakness, note what it buys: because the luminosities never depend on the fit, they cannot be
biased by it. The imaging pipeline has to be careful that its measured luminosities come from a single light fit (a
mixed measurement corrupts the ratio); here there is nothing to mix.

__Transformer Choice__

This script uses `TransformerDFT`, which is simple and exact for a modest number of visibilities. For large visibility
sets the sibling `interferometer/features/extra_galaxies/slam.py` shows the `TransformerNUFFT` + sparse-operator
treatment, which is a performance concern orthogonal to the scaling relation and is documented there rather than
duplicated here.

__Prerequisites__

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
- **Interferometer Scaling Relation** (`interferometer/features/scaling_relation/modeling`)
- **Interferometer Extra Galaxies SLaM** (`interferometer/features/extra_galaxies/slam`) — the same stage structure
  with companions modelled individually, plus the NUFFT / sparse-operator machinery.

__The Relation Across Stages__

The anchor's mass profile changes as the pipeline proceeds and the tie follows it:

 - `source_lp[1]` — anchor mass is `Isothermal`; the tier ties to its `einstein_radius`.
 - `mass_total[1]` — anchor mass is `PowerLaw`; the tier ties to *that* `einstein_radius`.

In both cases the tie multiplies a free model parameter, so the tier stays free of parameters throughout.

__This Script__

Using SOURCE LP, SOURCE PIX (two stages) and MASS TOTAL pipelines this script fits `Interferometer` data where in the
final model:

 - The lens galaxy's total mass is a `PowerLaw` plus `ExternalShear` (no lens light — none is detected).
 - Each scaling companion has an `IsothermalSph` mass tied to the anchor, and no light.
 - The source galaxy's light is a `Pixelization`.

All companion mass profiles are **untruncated**: truncation encodes tidal stripping by a host halo's potential, which
a galaxy-scale lens lacks. Truncated `dPIEMass` members belong to the group and cluster workflows.

__Contents__

- **The Tie:** One helper, applied in two stages.
- **SOURCE LP PIPELINE:** Mass and source introduced; the tie applied for the first time.
- **SOURCE PIX PIPELINE 1 / 2:** Pixelized source, tier carried forward then fixed.
- **MASS TOTAL PIPELINE:** `PowerLaw` anchor, with the tie re-applied to it.
- **Dataset / Centres / External Luminosities / Settings / Mesh Shape.**
- **SLaM Pipeline:** Run the stages in order.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__The Tie__

One helper used by both mass stages, so the relation cannot be written two subtly different ways.
"""


def scaling_galaxies_from(
    centres,
    luminosities,
    luminosity_anchor,
    anchor_mass,
    redshift_lens,
    scaling_exponent=0.5,
):
    """
    A collection of mass-only scaling galaxies whose Einstein radii are tied to `anchor_mass.einstein_radius`.

    `anchor_mass` must be the *model* mass of the anchor in the stage being composed, so the tie multiplies that
    stage's free parameter. Passing an already-fitted instance would silently freeze the tier.
    """
    scaling_galaxies_list = []

    for centre, luminosity in zip(centres, luminosities):
        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = tuple(centre)
        mass.einstein_radius = (
            anchor_mass.einstein_radius
            * (luminosity / luminosity_anchor) ** scaling_exponent
        )

        scaling_galaxies_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)
        )

    return af.Collection(scaling_galaxies_list)


"""
__SOURCE LP PIPELINE__

Equivalent to `source_lp` in `slam_start_here.py`, with no lens light (none is detected) and the tier introduced here
with its mass tied to the anchor.
"""


def source_lp(
    settings_search,
    dataset,
    mask_radius,
    scaling_galaxies_centres,
    scaling_galaxies_luminosities,
    luminosity_anchor,
    redshift_lens,
    redshift_source,
    n_batch=50,
):
    analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=5, centre_prior_is_uniform=False
    )

    lens = af.Model(
        al.Galaxy,
        redshift=redshift_lens,
        bulge=None,
        disk=None,
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
    )

    scaling_galaxies = scaling_galaxies_from(
        centres=scaling_galaxies_centres,
        luminosities=scaling_galaxies_luminosities,
        luminosity_anchor=luminosity_anchor,
        anchor_mass=lens.mass,
        redshift_lens=redshift_lens,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=lens,
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
        ),
        scaling_galaxies=scaling_galaxies,
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

Identical to `slam_start_here.py`, except the tier is carried forward from `source_lp[1]` as a free model. The tie
travels with the model, so it stays anchored without being re-declared.
"""


def source_pix_1(
    settings_search,
    dataset,
    source_lp_result,
    mesh_shape,
    n_batch=20,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(dataset=dataset, adapt_images=adapt_images)

    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.Isothermal),
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=mass,
                shear=source_lp_result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularRTUAdaptDensity, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        scaling_galaxies=source_lp_result.model.scaling_galaxies,
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

Identical to `slam_start_here.py`, except the tier is fixed as an instance from `source_pix[1]`.
"""


def source_pix_2(
    settings_search,
    dataset,
    source_lp_result,
    source_pix_result_1,
    mesh_shape,
    n_batch=20,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(
        dataset=dataset, adapt_images=adapt_images, use_jax=True
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularRTUAdaptImage, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        scaling_galaxies=source_pix_result_1.instance.scaling_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Identical to `slam_start_here.py`, except the anchor's mass becomes a `PowerLaw` and the tier is **re-tied** to that
profile's `einstein_radius`.

Re-tying matters: the tier was tied to an `Isothermal` radius in `source_lp[1]`, and an `Isothermal` and a `PowerLaw`
do not share a parameter object. Carrying the old collection forward would leave the tier anchored to a profile no
longer in the model.

The luminosities are the same external numbers used in `source_lp[1]` — there is nothing to re-measure.
"""


def mass_total(
    settings_search,
    dataset,
    source_result_for_lens,
    source_result_for_source,
    scaling_galaxies_centres,
    scaling_galaxies_luminosities,
    luminosity_anchor,
    redshift_lens,
    n_batch=20,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(dataset=dataset, adapt_images=adapt_images)

    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    lens = af.Model(
        al.Galaxy,
        redshift=source_result_for_lens.instance.galaxies.lens.redshift,
        bulge=None,
        disk=None,
        mass=mass,
        shear=source_result_for_lens.model.galaxies.lens.shear,
    )

    scaling_galaxies = scaling_galaxies_from(
        centres=scaling_galaxies_centres,
        luminosities=scaling_galaxies_luminosities,
        luminosity_anchor=luminosity_anchor,
        anchor_mass=lens.mass,
        redshift_lens=redshift_lens,
    )

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source=source),
        scaling_galaxies=scaling_galaxies,
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
"""
dataset_name = "scaling_relation"
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256), pixel_scales=0.1, radius=mask_radius
)

dataset_path = Path("dataset") / "interferometer" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/interferometer/features/scaling_relation/simulator.py",
        ],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Centres__

From ancillary imaging — the interferometer data contains no information about where a faint companion sits.
"""
scaling_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "scaling_galaxies_centres.json")
)

"""
__External Luminosities__

The inputs this pipeline cannot produce for itself. Explicit lists first, matching the register of the other scripts
in this folder; the CSV form below is the more realistic one here, because these numbers really do come from a
separate photometric catalogue.
"""
luminosity_anchor = 31.0962

scaling_galaxies_luminosities = [1.4939, 1.0865, 0.7696, 0.4980, 0.2716]

# Equivalent, and the natural form once you have a real catalogue:
#
#     main_lens_table = al.galaxy_table_from_csv(file_path=dataset_path / "main_lens_galaxies.csv")
#     luminosity_anchor = main_lens_table.luminosities[0]
#
#     scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
#     scaling_galaxies_centres = scaling_table.centres
#     scaling_galaxies_luminosities = scaling_table.luminosities

assert len(scaling_galaxies_luminosities) == len(list(scaling_galaxies_centres))

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("interferometer") / "slam",
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

As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before modeling.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__SLaM Pipeline__
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    scaling_galaxies_centres=scaling_galaxies_centres,
    scaling_galaxies_luminosities=scaling_galaxies_luminosities,
    luminosity_anchor=luminosity_anchor,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    mesh_shape=mesh_shape,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh_shape=mesh_shape,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    scaling_galaxies_centres=scaling_galaxies_centres,
    scaling_galaxies_luminosities=scaling_galaxies_luminosities,
    luminosity_anchor=luminosity_anchor,
    redshift_lens=redshift_lens,
)

print(mass_result.info)
