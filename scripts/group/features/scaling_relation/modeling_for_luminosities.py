"""
Modeling Features (Group): Fit Galaxy Luminosities for a Scaling Relation
=========================================================================

The scaling-relation modeling examples (`modeling.py` in this directory and
`scripts/imaging/features/scaling_relation/modeling.py`) need a measured **luminosity** for every galaxy that sits on
the relation:

    sigma = sigma_ref * (luminosity / reference_luminosity) ** 0.25
    r_cut = r_cut_ref * (luminosity / reference_luminosity) ** 0.7

Those tutorials hardcode the luminosity list for readability. In a production fit the luminosities have to be measured
from the data itself. This example shows the standard standalone way to do that:

 1. Load the dataset and all three centre JSON files (main, extras, scaling).
 2. Build a model with **light only** — MGE bulges for every galaxy, no mass profiles, no source.
 3. Fit that model with a single non-linear search.
 4. Compute the total luminosity per galaxy from the fitted MGE bulge gaussians.
 5. Print the luminosities so they can be pasted into the scaling-relation modeling script (or write them to JSON for
    re-use).

Why a separate light-only fit? Because the lensing model is dominated by the **mass** of the galaxies, but the scaling
relation needs the **light** of each galaxy as an *input*. Fitting the light first decouples the two, gives the relation
a stable per-galaxy measurement, and avoids degeneracies between the relation parameters and individual luminosities.

The same idea is implemented inside the production SLaM pipelines as the `source_lp_0` stage. See
`scripts/group/slam.py` and `scripts/group/features/pixelization/slam.py` for the chained-pipeline equivalent. This
script is the standalone, single-stage version of that step — useful for users who don't want to commit to a full SLaM
pipeline just to obtain luminosities.

__Contents__

- **Dataset & Mask:** Standard set up of the dataset and mask.
- **Centres:** Load all three centre JSON files.
- **Light-only Model:** MGE bulge per galaxy, no mass, no source.
- **Search and Analysis:** Configure the non-linear search and run the model-fit.
- **Compute Luminosities:** Extract `total_luminosity` per galaxy from the fitted MGE bulge gaussians.
- **Output:** Print and save the luminosities for the scaling-relation modeling script.
- **Wrap Up:** What to do next.

__Light-only Model Sizing__

The MGE bulge for the main lens uses 30 gaussians across 2 bases (matching the production SLaM `source_lp_0` stage).
The extras and scaling galaxies use 10 gaussians each in 1 basis — the smaller companions don't need the extra
flexibility and shrinking the basis keeps the model dimensionality low.

__Output Galaxy Order__

After the fit, the tracer's galaxies appear in the order:

    main_lenses[0..n_main-1], extras[0..n_extra-1], scaling[0..n_scaling-1]

Use this ordering when indexing `tracer.galaxies` to compute per-galaxy luminosities.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset", "group", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/features/scaling_relation/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

pixel_scale = float(dataset.pixel_scales[0])

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__
"""
mask_radius = 8.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

n_main = len(list(main_lens_centres))
n_extra = len(list(extra_galaxies_centres))
n_scaling = len(list(scaling_galaxies_centres))

print(f"Main lens galaxies: {n_main}")
print(f"Extra galaxies: {n_extra}")
print(f"Scaling galaxies: {n_scaling}")

"""
__Light-only Model__

Each galaxy gets an MGE bulge and nothing else — no mass, no source. The main lens uses a richer 30-gaussian / 2-basis
MGE because it dominates the light; companions use 10 gaussians in 1 basis.
"""
lens_dict = {}

for i, centre in enumerate(main_lens_centres):
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=30,
        gaussian_per_basis=2,
        centre_prior_is_uniform=False,
        centre=tuple(centre),
        centre_sigma=0.1,
    )
    lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, redshift=0.5, bulge=bulge)

extra_galaxies_list = []

for centre in extra_galaxies_centres:
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre_prior_is_uniform=True,
        centre=tuple(centre),
        ell_comps_prior_is_uniform=True,
    )
    extra_galaxies_list.append(af.Model(al.Galaxy, redshift=0.5, bulge=bulge))

extra_galaxies = af.Collection(extra_galaxies_list)

scaling_galaxies_list = []

for centre in scaling_galaxies_centres:
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre_prior_is_uniform=True,
        centre=tuple(centre),
        ell_comps_prior_is_uniform=True,
    )
    scaling_galaxies_list.append(af.Model(al.Galaxy, redshift=0.5, bulge=bulge))

scaling_galaxies = af.Collection(scaling_galaxies_list)

"""
__Model__

No source galaxy: this is purely a light-only fit. Keep the three populations in their own collections so the post-fit
tracer ordering is predictable: main lenses first, then extras, then scaling galaxies.
"""
model = af.Collection(
    galaxies=af.Collection(**lens_dict),
    extra_galaxies=extra_galaxies,
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
__Over Sampling__
"""
all_centres = (
    list(main_lens_centres)
    + list(extra_galaxies_centres)
    + list(scaling_galaxies_centres)
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Search__

A single Nautilus search. With no mass and no source, the parameter space is much smaller than a full lens fit so we
can use a smaller `n_live` than `modeling.py` and converge faster.
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features" / "scaling_relation",
    name="modeling_for_luminosities",
    unique_tag=dataset_name,
    n_live=100 + 30 * (n_main + n_extra + n_scaling),
    n_batch=50,
    iterations_per_quick_update=10000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Compute Luminosities__

The fit's `tracer_linear_light_profiles_to_light_profiles` property converts the linear-inversion-solved intensities
back to standard `LightProfile`s with concrete `intensity` values. This is the canonical way to read MGE intensities
out of a result.

For an MGE basis composed of 2D Gaussians, the per-gaussian luminosity is:

    L_g = 2 * pi * sigma^2 / axis_ratio * intensity

The total luminosity of the galaxy is the sum of `L_g` over all gaussians in its `bulge.profile_list`, divided by
`pixel_scale**2` to convert from per-pixel to per-arcsec^2.

This is the same formula used by the SLaM pipelines in `source_lp_1` (see `scripts/group/slam.py`).
"""
fit_tracer = (
    result.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles
)


def total_luminosity_from(galaxy):
    return (
        sum(
            2.0 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
            for g in galaxy.bulge.profile_list
        )
        / pixel_scale**2
    )


main_luminosities = [
    total_luminosity_from(fit_tracer.galaxies[i]) for i in range(n_main)
]
extra_luminosities = [
    total_luminosity_from(fit_tracer.galaxies[n_main + i]) for i in range(n_extra)
]
scaling_luminosities = [
    total_luminosity_from(fit_tracer.galaxies[n_main + n_extra + i])
    for i in range(n_scaling)
]

print("Main lens luminosities:", main_luminosities)
print("Extra galaxy luminosities:", extra_luminosities)
print("Scaling galaxy luminosities:", scaling_luminosities)

"""
__Output__

Write the centres + luminosities to a `scaling_galaxies.csv` next to the centre JSONs. This is the same file the
scaling-relation modeling script consumes via `al.galaxy_table_from_csv`, so the result of this fit can be chained
into the next step with no manual copy/paste.
"""
csv_path = dataset_path / "scaling_galaxies.csv"

al.galaxy_table_to_csv(
    centres=list(scaling_galaxies_centres),
    luminosities=[float(l) for l in scaling_luminosities],
    file_path=csv_path,
)

print(f"Wrote scaling-galaxy centres + luminosities to: {csv_path}")

"""
__Wrap Up__

The CSV at `dataset_path / "scaling_galaxies.csv"` is the canonical chain-point. The downstream modeling script
loads it directly:

    table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres        = table.centres
    scaling_galaxies_luminosity_list = table.luminosities

Re-running this script overwrites the CSV in place, so iterating on the light fit and re-running the lens fit is
just two `python ...` invocations.

For the chained-pipeline alternative — where the light fit is the `source_lp[0]` stage of a SLaM run instead of a
standalone search — see `scripts/group/slam.py` (`__SOURCE LP PIPELINE — stage 0__`) and
`scripts/group/features/pixelization/slam.py`.
"""
