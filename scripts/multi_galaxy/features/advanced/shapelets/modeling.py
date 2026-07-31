"""
Modeling Features: Shapelets (Multi Galaxy)
===========================================

This script fits a multi-galaxy strong lens where the **source** is reconstructed with a basis of shapelets — a
set of orthogonal basis functions well suited to describing irregular, asymmetric morphology with relatively few
non-linear parameters.

Shapelets are described in full in Refregier (2003), MNRAS 338, 35 (arXiv:astro-ph/0105178).

__Contents__

- **Advantages:** Why a shapelet basis rather than an analytic profile.
- **Disadvantages:** What it costs.
- **Positive Negative Solver:** Why shapelets need negative intensities allowed.
- **Why The Deflectors Stay MGE:** The one composition choice worth explaining up front.
- **Model:** Compose the lens model fitted to the data.
- **Dataset, Mask & Over Sampling:** Standard set up.
- **Positions:** The multiple images used to constrain the mass model.
- **Model Composition:** One `lens_i` per deflector, plus a shapelet source.
- **Model Cookbook:** Where the full model-composition API is documented.
- **Search & Analysis:** Configure the fit.
- **Run Time:** Profiling the expected run time of the model-fit.
- **VRAM:** GPU memory used by the model-fit.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Advantages__

A source that is not a smooth ellipse cannot be described by a Sersic, and the residuals a Sersic leaves are
absorbed by the mass model. A shapelet basis can describe clumps, asymmetry and tidal structure, and every
shapelet's `intensity` is solved by linear algebra — so the extra flexibility costs no non-linear parameters.

The whole basis shares one centre, one ellipticity and one `beta` (its characteristic size), which is why ~60
shapelets cost only four sampled parameters between them.

__Disadvantages__

The basis is still centred and sized by those shared parameters, so it describes structure around one location.
A source split into widely separated components is better served by a pixelization
(`multi_galaxy/features/pixelization`), which has no such constraint.

Shapelets also oscillate. Truncating the basis too low leaves structure unfitted; taking it too high lets the
higher orders fit noise.

__Positive Negative Solver__

Shapelets of order `n > 0` are negative over part of their extent — that is how a sum of them describes anything
other than a bump. The linear solve must therefore be allowed to return negative intensities, so the analysis is
created with `al.Settings(use_positive_only_solver=False)`.

This is the opposite of the setting used for an MGE or a pixelization, where negative components would be
unphysical.

__Why The Deflectors Stay MGE__

The multi-galaxy regime is populated by interacting systems — the package's reference pair, SDSS J1011+0143, is a
merger, and tidally disturbed light is the norm rather than the exception. That is a natural argument for putting
a flexible basis on the *deflectors* too, and the API supports it.

It is not what this script does, because it is not what the library recommends. As
`imaging/features/advanced/shapelets/modeling.py` says under its own `__Lens Shapelets__` section, the model is
not established in the literature and an MGE is faster and gives better results for massive early-type galaxies —
which is what both deflectors here are. The disturbed-morphology case for the deflectors is answered by
`multi_galaxy/features/multi_gaussian_expansion`, whose dataset has exactly that twisted, two-component light.

So the deflectors keep the package-default MGE and the shapelets go where they earn their keep: the source.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light is an MGE, its mass an `Isothermal` with its centre fixed.
 - The system has a single overall `ExternalShear` at the system centre.
 - The source's light is a `Basis` of linear `ShapeletPolar` profiles.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/modeling.py` for the multi-galaxy composition and
`imaging/features/advanced/shapelets/modeling.py` for the full shapelet API, including the Cartesian basis and
the basis-regularization options this script does not use.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `simple` multi-galaxy dataset, the same co-dominant pair fitted by `multi_galaxy/modeling.py`. This folder
needs no dataset of its own — the shapelets describe the source, and the `simple` source is the one every other
script in the package fits.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Extra Galaxies Noise Scaling__

Scale the faint contaminant out of the fit, as `multi_galaxy/modeling.py` explains.

This matters for a shapelet source in the same way it matters for a pixelized one: unmodelled flux inside the
mask is exactly what extra flexibility will reach for.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.
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
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Positions__

The source's multiple images, applied as a `PositionsLH` likelihood penalty on the mass model.

A flexible source model makes this more useful than it is for a Sersic source: the more freedom the source has,
the more readily an incorrect mass model can be made to fit, and the positions constrain the mass model directly.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "positions.json")
)

"""
__Model Composition__

The standard multi-galaxy composition from `multi_galaxy/modeling.py` — one `lens_i` per deflector in a loop, the
shear in its own `shear_galaxy` — with the source's MGE replaced by a shapelet basis.

The basis is built by choosing a maximum order `total_n` and enumerating the `(n, m)` pairs up to it. Every
shapelet after the first takes its `centre`, `ell_comps` and `beta` from the first, so the whole basis moves,
rotates and scales together as four sampled parameters.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

# External Shear:

shear_galaxy = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

# Source (Shapelet Basis):

total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = af.Collection(
    af.Model(al.lp_linear.ShapeletPolar) for _ in range(total_n + total_m + 1)
)

n_count = 1
m_count = -1

for i, shapelet in enumerate(shapelets_bulge_list):
    if i == 0:
        shapelet.n = 0
        shapelet.m = 0

    else:
        shapelet.n = n_count
        shapelet.m = m_count

        m_count += 2

        if m_count > n_count:
            n_count += 1
            m_count = -n_count

    shapelet.centre = shapelets_bulge_list[0].centre
    shapelet.ell_comps = shapelets_bulge_list[0].ell_comps
    shapelet.beta = shapelets_bulge_list[0].beta

source_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=shapelets_bulge_list,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, shear_galaxy=shear_galaxy, source=source)
)

"""
The `info` attribute shows the model in a readable format.

Note how few parameters the source contributes despite the size of the basis — its centre, its two ellipticity
components and its `beta`. Every shapelet's intensity is solved.
"""
print(model.info)

"""
__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Search__

The lens model is fitted using the nested sampling algorithm Nautilus, with the live-point count of
`multi_galaxy/modeling.py`.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features" / "advanced",
    name="shapelets",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

`use_positive_only_solver=False` is what lets the shapelet intensities go negative, as described above.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[
        al.PositionsLH(positions=positions, threshold=0.3)
    ],
    settings=al.Settings(use_positive_only_solver=False),
    use_jax=True,
)

"""
__Run Time__

Run times are discussed in full in `multi_galaxy/modeling.py`.

A shapelet basis is a linear object per shapelet, so the intensity solve is larger than for a Sersic source and
comparable to an MGE of similar size. `total_n` above is the main run-time dial in this script.

__VRAM__

The `multi_galaxy/modeling.py` example explains how VRAM is used during GPU-based fitting and how to print the
estimated VRAM required by a model.

The method below prints the VRAM estimate for this analysis and model. It takes 20-30 seconds, so comment it out
once you are familiar with your GPU's limits.
"""
# analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result contains one entry per deflector plus the shapelet source, whose solved intensities are available on
the maximum log likelihood fit's linear objects.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/shapelets/fit.py` — the same composition without a search, where the basis
   itself can be inspected.
 - `multi_galaxy/features/pixelization` — a free-form source, for structure a centred basis cannot describe.
 - `multi_galaxy/features/multi_gaussian_expansion` — the answer to disturbed *deflector* morphology.
 - `imaging/features/advanced/shapelets` — the full shapelet API, including the Cartesian basis.
"""
