"""
Modeling Features: Operated Light Profiles (Multi Galaxy)
=========================================================

This script fits a multi-galaxy strong lens where **each co-dominant deflector hosts a compact nuclear point
source** — an AGN — modelled with an operated light profile.

An operated light profile is one assumed to have already been convolved with the PSF, so it is not convolved
again during the fit. That is what a point source is in the data: the recorded image of an unresolved nucleus is
the PSF itself, scaled by the source's brightness.

__Contents__

- **Advantages:** Why an operated profile rather than an ordinary one.
- **Disadvantages:** What it costs.
- **Positive Only Solver:** Why the solved intensities are constrained to be positive.
- **Model:** Compose the lens model fitted to the data.
- **Dataset, Mask & Over Sampling:** Standard set up.
- **Positions:** The multiple images used to constrain the mass model.
- **Model Composition:** One `lens_i` per deflector, each with an MGE bulge and an operated point source.
- **Model Cookbook:** Where the full model-composition API is documented.
- **Search & Analysis:** Configure the fit.
- **Run Time:** Profiling the expected run time of the model-fit.
- **VRAM:** GPU memory used by the model-fit.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Advantages__

Convolving a point source with the PSF twice smears it to roughly twice the PSF width, which no combination of
the other model parameters can undo. Modelling it as operated skips the second convolution, so the profile the
model puts in the image is the shape the data actually contains.

__Disadvantages__

The profile has to genuinely be pre-convolved for this to be right. Using an operated profile for extended
emission that the telescope really did blur leaves the model under-convolved, which is the same error in the
opposite direction.

It also adds a component per deflector, and each one is another thing the fit can trade against that galaxy's
extended light.

__Positive Only Solver__

The point source's `intensity` is solved by linear algebra rather than sampled, like every other linear profile
in this package. A positive-only solver is used, so the nucleus cannot reconstruct negative flux to compensate
for an over-bright bulge.

__What Changes For Multiple Deflectors__

Each deflector gets its own point source, and the two are independent — a pair of interacting galaxies is not
required to host equally active nuclei.

That independence is the reason to model them at all. Light that is not in the model does not vanish; it is
absorbed by whichever component can absorb it. With two deflectors sitting on top of each other, an unmodelled
nucleus in one of them is absorbed asymmetrically, and what distorts is the ratio between the two galaxies'
luminosities — the quantity `multi_galaxy/slam.py`'s `light[1]` stage exists to protect.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light is an MGE plus an operated linear `Gaussian` point source, its mass an
   `Isothermal` with its centre fixed.
 - The system has a single overall `ExternalShear` at the system centre.
 - The source's light is an MGE.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/modeling.py` for the multi-galaxy composition and
`imaging/features/advanced/operated_light_profile/modeling.py` for the galaxy-scale walkthrough of the operated
profiles themselves.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `operated` multi-galaxy dataset — the `simple` pair with a nuclear point source added to each deflector.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "operated"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/operated_light_profile/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.

The over-sampling is doing more work here than in the other examples: a point source is the steepest thing in the
image, and there is one at each of the centres the scheme is centred on.
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

The source's multiple images, solved by the simulator and saved alongside the dataset, applied as a `PositionsLH`
likelihood penalty on the mass model.
"""
positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))

print(f"Multiple image positions used to constrain the mass model:\n{positions}")

"""
__Model Composition__

The standard multi-galaxy composition from `multi_galaxy/modeling.py` — one `lens_i` per deflector in a loop, the
shear in its own `shear_galaxy` — with one component added per deflector.

Each deflector's `point` is an `lp_linear_operated.Gaussian`:

 - **`lp_linear_`** means its `intensity` is solved by linear algebra rather than sampled, so the nucleus costs no
   non-linear parameters for its brightness.
 - **`_operated`** means it is not convolved with the PSF during the fit.

Its centre is fixed to the deflector's known centre, like the mass centre is. With two nearby galaxies a free
point-source centre has the same failure mode as a free mass centre — nothing in the model says which galaxy it
belongs to, and it can drift onto the brighter one.

Its `sigma` is left free. Fixing it to the PSF width would assume the nucleus is exactly unresolved; leaving it
free lets the fit say whether it is marginally resolved instead.
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

    point = af.Model(al.lp_linear_operated.Gaussian)
    point.centre = (centre[0], centre[1])

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        point=point,
        mass=mass,
    )

# External Shear:

shear_galaxy = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

# Source:

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, shear_galaxy=shear_galaxy, source=source)
)

"""
The `info` attribute shows the model in a readable format.

Each deflector's `point` contributes one sampled parameter — its `sigma`. Its intensity is solved, and its centre
is fixed.
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
    name="operated_light_profile",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
    use_jax=True,
)

"""
__Run Time__

Run times are discussed in full in `multi_galaxy/modeling.py`.

An operated profile is cheaper per evaluation than an ordinary one, because the PSF convolution step is skipped
for it. Adding one per deflector still adds a linear object per deflector to the intensity solve, so the net
effect on run time is small in either direction.

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

The result contains one entry per deflector, each carrying its MGE `bulge`, its operated `point` and its `mass`.

Two checks are worth making here that the other examples do not need:

 - **The two point sources' solved intensities.** They were simulated as different, and a fit that returns them
   as near-equal is usually a sign the extended light models have absorbed the difference.
 - **The fitted `sigma` against the PSF width.** A nucleus that is genuinely unresolved should return a `sigma`
   close to the PSF's; a much wider one is describing extended light, not a point source.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/modeling.py` — the same lens without the nuclear point sources.
 - `multi_galaxy/features/advanced/shapelets` — a different way of giving a deflector's light more freedom.
 - `multi_galaxy/slam.py` — the SLaM pipeline, whose stages already carry a `point` slot per deflector.
 - `imaging/features/advanced/operated_light_profile` — the galaxy-scale walkthrough of the operated profiles.
"""
