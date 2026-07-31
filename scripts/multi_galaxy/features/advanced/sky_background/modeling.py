"""
Modeling Features: Sky Background (Multi Galaxy)
================================================

This script fits a multi-galaxy strong lens whose data still contains the **sky background** — the diffuse
emission from the atmosphere, zodiacal light and unresolved sources in the field.

The sky is usually subtracted during data reduction. If that subtraction were perfect there would be nothing to
model here, but it rarely is, and what is left over is a constant offset across the image that every component of
the lens model can absorb.

__Contents__

- **Advantages:** Why fit the sky rather than assume it away.
- **Disadvantages:** What it costs.
- **Model:** Compose the lens model fitted to the data.
- **Dataset, Mask & Over Sampling:** Standard set up.
- **Positions:** The multiple images used to constrain the mass model.
- **Model Composition:** One `lens_i` per deflector, plus the sky as a `DatasetModel`.
- **Model Cookbook:** Where the full model-composition API is documented.
- **Search & Analysis:** Configure the fit.
- **Run Time:** Profiling the expected run time of the model-fit.
- **VRAM:** GPU memory used by the model-fit.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Advantages__

A residual sky is flat, and the faint outskirts of a galaxy's light profile are nearly flat too. If the sky is
not in the model, the light profiles will take it — extending their outskirts to cover a level that has nothing
to do with the galaxies.

Fitting it costs one parameter and removes that degeneracy at the source.

__Disadvantages__

That one parameter is degenerate with the light profiles' outer slopes by construction, so it widens their
posteriors. It also needs a prior: `background_sky_level` has no default that suits every dataset, and the range
has to come from what you know about the reduction.

__What Changes For Multiple Deflectors__

The sky is a single number for the whole image, and there are two bright galaxies in that image whose faint
outskirts overlap.

That makes it a **shared** systematic. A mis-estimated sky does not perturb one galaxy's light model and leave
the other alone — both absorb it, in proportion to how much of the image each one's outskirts cover. What moves
is the ratio between the two galaxies' luminosities, which is frequently the measurement.

At galaxy scale a badly handled sky costs you one galaxy's outer profile. Here it costs you the comparison
between two.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light is an MGE, its mass an `Isothermal` with its centre fixed.
 - The system has a single overall `ExternalShear` at the system centre.
 - The source's light is an MGE.
 - The sky background is a `DatasetModel` with a free `background_sky_level`.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/modeling.py` for the multi-galaxy composition and
`imaging/features/advanced/sky_background/modeling.py` for the galaxy-scale walkthrough.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `sky_background` multi-galaxy dataset — the `simple` pair, simulated with the sky left in rather than
subtracted.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "sky_background"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/sky_background/simulator.py",
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

The mask matters more than usual here. The sky is constrained by the pixels where nothing else contributes, and a
tight mask leaves few of them — so the sky and the light profiles' outskirts become harder to tell apart.
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
        sub_size_list=[8, 4, 1],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Positions__

The source's multiple images, applied as a `PositionsLH` likelihood penalty on the mass model.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "positions.json")
)

"""
__Model Composition__

The standard multi-galaxy composition from `multi_galaxy/modeling.py` — one `lens_i` per deflector in a loop, the
shear in its own `shear_galaxy` — plus one component that is not a galaxy at all.

The sky is a property of the **dataset**, not of anything in the lens, so it is composed as a `DatasetModel`
rather than added to a galaxy. It is passed to the analysis alongside the model.

`background_sky_level` has no sensible default prior, because what counts as a plausible sky depends entirely on
the instrument and the reduction. The `UniformPrior` below spans the range this dataset's simulator could
plausibly have produced; for real data, set it from what your reduction says the residual sky could be.
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

# Source:

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Sky Background:

dataset_model = af.Model(al.DatasetModel)
dataset_model.background_sky_level = af.UniformPrior(
    lower_limit=0.0, upper_limit=10.0
)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, shear_galaxy=shear_galaxy, source=source),
    dataset_model=dataset_model,
)

"""
The `info` attribute shows the model in a readable format.

Note that `dataset_model` sits alongside `galaxies` rather than inside it — the sky belongs to the data, not to
the lens.
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
    name="sky_background",
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
    positions_likelihood_list=[
        al.PositionsLH(positions=positions, threshold=0.3)
    ],
    use_jax=True,
)

"""
__Run Time__

Run times are discussed in full in `multi_galaxy/modeling.py`.

Adding the sky costs one sampled parameter and an addition per likelihood evaluation, so the per-evaluation cost
is essentially unchanged. What it can cost is convergence, because the new parameter is degenerate with the light
profiles' outskirts.

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

The result carries the fitted `dataset_model` alongside the galaxies.

Two checks are worth making:

 - **The sky against its prior bounds.** A posterior pressed against either limit means the prior, not the data,
   is setting the sky — and the light models are absorbing the difference.
 - **The two deflectors' luminosities.** They are what a mis-estimated sky moves, and they move together rather
   than independently, so compare them against a fit with the sky fixed rather than reading each in isolation.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/sky_background/fit.py` — the same model without a search, with the sky fixed
   to its true value.
 - `multi_galaxy/modeling.py` — the same lens on sky-subtracted data.
 - `imaging/features/advanced/sky_background` — the galaxy-scale walkthrough.
"""
