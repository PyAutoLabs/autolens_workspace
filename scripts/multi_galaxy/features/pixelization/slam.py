"""
SLaM (Source, Light and Mass): Pixelization (Multi Galaxy)
==========================================================

The SLaM pipeline for a multi-galaxy lens, with every pixelization choice its SOURCE PIX stages make written out.

`multi_galaxy/slam.py` **already uses a pixelized source** — the SOURCE PIX stages are where it happens, and this
pipeline does not introduce it. What this script adds is the mesh and regularization section: which classes the
baseline picks, why the two SOURCE PIX searches use different ones, and which pairings are not available.

Read `multi_galaxy/slam.py` first, and `guides/modeling/slam_start_here` before it. This script documents only
what differs.

__What Changes__

Nothing in the five stages. The stage functions below are the baseline's, copied rather than imported —
`multi_galaxy/slam.py` is a script, so importing it would execute its whole pipeline as a side effect.

The difference is the `__Pixelization Choices__` and `__Mesh Shape__` sections, which unpack the four lines the
baseline sets in passing.

__Pixelization Choices__

 - **Two meshes, not one.** `source_pix[1]` uses `RectangularBilinearAdaptDensity`, which places pixels by the traced
   grid's own density and so needs nothing from an earlier fit. `source_pix[2]` uses `RectangularBilinearAdaptImage`,
   which places them by the source's adapt image — which `source_pix[1]` has by then produced. The progression is
   the reason there are two SOURCE PIX searches rather than one.

 - **`Adapt` regularization in both.** It varies the smoothing across the source using the same adapt image,
   regularizing bright regions less and faint regions more. `source_pix[1]` gets its adapt image from the SOURCE
   LP result's parametric source, `source_pix[2]` from `source_pix[1]`'s reconstruction.

 - **`AdaptSplit` is not available here.** It regularizes with a cross of four points around each pixel centre
   and needs the mesh to supply split-cross mappings, which the rectangular meshes do not. Pairing them raises a
   `PixelizationException`. Use `multi_galaxy/features/pixelization/delaunay.py`'s mesh if you want the split
   schemes.

 - **The mesh shape is fixed, not fitted.** The number of source pixels sets the size of every matrix in the
   inversion, and JAX needs those shapes static across samples.

__What This Means With Two Deflectors__

The adapt images are per-galaxy, and the source's is the data with every deflector's light model subtracted. With
two co-dominant deflectors that is two light models, so both SOURCE PIX searches depend on the SOURCE LP stage
having separated them. This is the same dependency `multi_galaxy/slam.py` describes for its LIGHT LP stage,
reaching one stage earlier.

__Contents__

- **Source LP Pipeline:** Light and mass per deflector, and the source.
- **Source Pix Pipeline 1 & 2:** The pixelized source, and the mesh progression between them.
- **Light LP Pipeline:** A fresh MGE per deflector.
- **Mass Total Pipeline:** Each deflector's mass promoted to a `PowerLaw`.
- **Dataset, Centres, Mask:** Set up.
- **Mesh Shape:** The pixelization classes, written out.
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

Identical to `multi_galaxy/slam.py`. One search initializing the source's light, the deflectors' light and their
mass, with each deflector's mass centre fixed to its light centre and its Einstein radius capped.

This is the stage the pixelization depends on: its parametric source is what `source_pix[1]`'s adapt image is
built from.
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

Identical to `multi_galaxy/slam.py`. The first pixelized source, whose purpose is to produce a better adapt image
than the parametric SOURCE LP source can.

__Where The Mesh Comes In__

`mesh_init` is passed in rather than hard-coded, and is a `RectangularBilinearAdaptDensity` — a mesh that needs no adapt
image of its own, because it places pixels by the density of the traced grid. That is what makes it usable at
this point in the pipeline, before a reconstruction exists.

The `regularization_init` is `Adapt`, which does need an adapt image; it gets one from the SOURCE LP result via
`galaxy_name_image_dict_via_result_from` below.

__Mass Centres Released__

`unfix_mass_centre=True` converts each mass centre from the value fixed in `source_lp[1]` into a free parameter,
as in the baseline.

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

Identical to `multi_galaxy/slam.py`. The final pixelized source, with every deflector's light and mass fixed as
instances.

__Where The Mesh Comes In__

This is the search that uses `RectangularBilinearAdaptImage`. It can, because `source_pix[1]` has now produced a
reconstruction of the source, and `galaxy_name_image_dict_via_result_from` below reads its adapt images from that
result rather than from the parametric SOURCE LP source.

Both the mesh and the regularization adapt to the same image. Everything except the source is fixed, so this
stage's cost does not grow with the number of deflectors.
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

Identical to `multi_galaxy/slam.py`: a fresh, free MGE for each deflector, with mass and source fixed from the
SOURCE PIX pipeline, reusing its adapt images.
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
__MASS TOTAL PIPELINE__

Identical to `multi_galaxy/slam.py`: each deflector's mass is promoted from `Isothermal` to `PowerLaw`, with the
light fixed from `light[1]` and the source fixed from `source_pix[2]`.

The source passed through here is the pixelization, carried as an instance by `source_from`, so the mesh and
regularization chosen at the top of this script are what the final mass measurement is made against.
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
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

    for i in range(n_main_from(source_result_for_lens)):

        lens_model = getattr(source_result_for_lens.model.galaxies, f"lens_{i}")
        light_instance = getattr(light_result.instance.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.PowerLaw),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_model.redshift,
            bulge=light_instance.bulge,
            disk=light_instance.disk,
            point=light_instance.point,
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

The `simple` multi-galaxy dataset, the same one the baseline pipeline fits.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
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
__Extra Galaxies Noise Scaling__

Scale the faint contaminant out of the fit before masking, as the baseline does. For a pixelized source this also
keeps the contaminant's flux out of the adapt images the SOURCE PIX searches build.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__

The centres of the co-dominant deflectors, which drive the loop in every stage.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

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

The settings applied to every search, including the output path.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("multi_galaxy") / "features" / "pixelization" / "slam",
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

The four lines the baseline sets in passing, written out.

`mesh_pixels_yx` is the number of source pixels per side. It is fixed rather than sampled, because it sets the
size of every matrix in the inversion and JAX requires those shapes to be static across samples. Raising it gives
a finer source reconstruction and a slower fit.

`mesh_init` and `regularization_init` go to `source_pix[1]`; `mesh` and `regularization` go to `source_pix[2]`.
The pair differ only in the mesh, for the reason given at the top of this script: `RectangularBilinearAdaptDensity` needs
no reconstruction to exist, `RectangularBilinearAdaptImage` does.

To use the Delaunay meshes and their split regularization schemes instead, see
`multi_galaxy/features/pixelization/delaunay.py`, which builds the image-plane mesh grid those meshes require.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mesh_init = af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape)
regularization_init = al.reg.Adapt

mesh = af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape)
regularization = al.reg.Adapt

"""
__SLaM Pipeline__

The five stages, in order. Each consumes the results of the ones before it.
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

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)

"""
__Result__

`mass_result` holds the final model. The checks worth making on the mass split and the mass centres are in
`multi_galaxy/slam.py`; the pixelization-specific one is the source reconstruction itself, which
`multi_galaxy/features/pixelization/source_science.py` shows how to inspect.
"""
print(mass_result.info)

aplt.subplot_fit_imaging(fit=mass_result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/slam.py` — the baseline pipeline these stages are copied from.
 - `multi_galaxy/features/pixelization/adaptive.py` — the same mesh progression, run by hand as four searches.
 - `multi_galaxy/features/pixelization/source_science.py` — measuring the reconstructed source this pipeline
   produces.
 - `guides/modeling/slam_start_here` — what each stage is for, in full.
"""
