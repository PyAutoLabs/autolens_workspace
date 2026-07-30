"""
Modeling Features: Multi Gaussian Expansion (Multi Galaxy)
==========================================================

A Multi Gaussian Expansion (MGE) decomposes a galaxy's light into ~10-30 Gaussians whose `intensity` values are
solved by linear algebra. This script fits a multi-galaxy strong lens giving **each co-dominant deflector its own
MGE**.

__Contents__

- **Why This Folder Exists:** `multi_galaxy/modeling.py` already uses an MGE — what is left to show.
- **Advantages:** What the MGE buys, measured on this dataset.
- **Disadvantages:** The cost, and the multi-galaxy-specific risk.
- **Model:** Compose the lens model fitted to the data.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Centres:** Load the deflector centres that drive the model composition loop.
- **Over Sampling:** Adaptive over-sampling at every deflector centre.
- **Model Composition:** One MGE basis per deflector, via the concise API.
- **How Many Gaussians:** Choosing `total_gaussians`, measured rather than guessed.
- **Search:** Configure the non-linear search used to fit the model.
- **Analysis:** Create the Analysis object that defines how the model is fitted to the data.
- **Result:** Overview of the results of the model-fit.
- **Wrap Up:** Where to go next.

__Why This Folder Exists__

`multi_galaxy/modeling.py` already fits an MGE — it is the package default, composed through the
`al.model_util.mge_model_from` helper which deliberately hides the API. So this folder is not introducing the
feature. It does three things that script does not:

 1. Fits a dataset whose light an MGE is *needed* for, rather than one a Sersic could fit.
 2. Shows the basis-composition API the helper hides, so you can vary it.
 3. Documents what having **two** MGEs in one model does to the linear system — which is the regime-specific part,
    and is not a small effect.

__The Dataset__

`simulator.py` in this folder writes a dataset the `simple` one cannot substitute for. Each deflector's light is
**two offset, differently-rotated Sersic components**, producing isophotal twists and a radially varying
ellipticity. A single elliptical Sersic has one centre, one axis ratio and one position angle at all radii, so it
provably cannot fit this.

That matters because the multi-galaxy regime is full of interacting systems — this package's reference pair,
SDSS J1011+0143, is a merger — and disturbed morphology is the norm.

__Advantages__

Measured on this dataset, at fixed light centres and without a non-linear search (so these are floors, not what a
fitted model achieves):

    single linear Sersic per deflector, at each galaxy's true bulge shape : log likelihood ~ -289,000
    MGE, 10 Gaussians per deflector                                       : log likelihood ~   -4,600
    MGE, 20 Gaussians per deflector                                       : log likelihood ~   -4,490
    MGE, 30 Gaussians per deflector                                       : log likelihood ~   -4,480
    the simulator's truth tracer                                          : log likelihood ~  +28,000

These are quoted to two significant figures on purpose. `simulator.py` adds Poisson noise without a fixed seed, so
re-simulating the dataset shifts every one of them by a percent or two. The *differences* between them are what is
stable, and they are enormous compared to that scatter.

The Sersic is given every advantage — its centre, axis ratio, position angle, effective radius and Sersic index
are all set to the true values of the galaxy's *bulge* component — and it is still roughly 284,000 in log
likelihood worse than a 10-Gaussian MGE. That is what "a Sersic cannot represent a twisted galaxy" means
quantitatively.

The MGE also costs *fewer* non-linear parameters than the Sersics it replaces, because neither the Gaussians'
`intensity` values nor their `sigma` values are sampled: the intensities are solved linearly and the sigmas are
fixed on a log10 grid spanning the mask.

__Disadvantages__

The ordinary cost is speed: 20 Gaussians per deflector is 40 light profiles to evaluate and convolve rather than
2, which is slower per likelihood evaluation (though the simpler parameter space usually wins overall).

The multi-galaxy-specific cost is subtler, and follows directly from
`multi_galaxy/features/linear_light_profiles/likelihood_function.py`. That script measured the curvature matrix
`F` for a single linear Sersic per galaxy and found the two deflectors coupled at **0.296** — the strongest
off-diagonal in the system. Repeat that measurement for this model, with 20 Gaussians per deflector, and `F`
becomes 41 x 41 with:

    deflector-deflector block |C| : mean 0.119, max 0.9877
    within-one-deflector block    : mean 0.459, max 1.0000
    deflector-source              : mean 0.098, max 0.384
    condition number of F         : ~1e24

Unlike the log likelihoods above, these coupling values are stable to four decimal places across re-simulations,
because `F` depends on the model geometry and the noise map rather than on the particular noise draw.

The number to look at is **max 0.9877**. Some Gaussian in `lens_0`'s basis is 99% degenerate with some Gaussian in
`lens_1`'s: at that separation and those widths, the solver genuinely cannot tell whose light it is. Giving each
galaxy more freedom to describe its own light necessarily gives the pair more freedom to trade light between
them.

This is not an argument against the MGE — the Sersic comparison above settles that. It is the reason the fixed
`sigma` grid, the fixed-ish centres and the **positive-only** solver are load-bearing rather than stylistic in
this regime. A positive-negative solver on a system with a condition number of ~1e24 and near-degenerate columns
will happily produce one galaxy with large positive Gaussians and its neighbour with compensating negative ones.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light is an MGE of 20 Gaussians with a free centre and shared ellipticity.
 - Each deflector's total mass distribution is an `Isothermal` with its centre fixed to its known position.
 - The system has a single overall `ExternalShear` at the system centre.
 - The source galaxy's light is an MGE.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/modeling.py` and
`imaging/features/multi_gaussian_expansion/modeling.py`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the `mge` multi-galaxy dataset — the co-dominant pair with twisted, two-component light.
"""
dataset_name = "mge"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

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
            "scripts/multi_galaxy/features/multi_gaussian_expansion/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
Look closely at the two foreground galaxies. Their isophotes are not concentric ellipses — the outer light is
rotated relative to the inner light and offset from it. That twist is what the MGE is here for.

__Centres__

Load the centres of the co-dominant deflectors, which drive the model composition loop.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask__

The standard 3.0" circular mask used throughout the multi-galaxy package, sized by the *combined* Einstein radius
(~1.8") rather than either galaxy's individually.

The mask radius does double duty for an MGE: it also sets the largest Gaussian `sigma`, since the basis spans
0.01" to the mask radius on a log10 grid. A mask that clips the galaxies' outskirts therefore also removes the
basis's ability to describe them.
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
__Over Sampling__

Adaptive over-sampling centred on **every** deflector. The smallest Gaussians in the basis have `sigma` = 0.01",
which is a fifth of a pixel — they are unresolved and must be evaluated on a finer grid at each galaxy's centre or
they are simply wrong.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__How Many Gaussians__

`total_gaussians` is the one MGE knob worth thinking about, and the measurement above answers it for this dataset:

    10 Gaussians : ~ -4,600
    20 Gaussians : ~ -4,490     (~ +130 over 10)
    30 Gaussians : ~ -4,480     (~   +8 over 20)

Ten is already almost all of the way there; twenty captures the rest; thirty adds nothing measurable while costing
50% more light-profile evaluations per likelihood call. **20 is the right choice here**, and the way to find that
out for your own data is to run exactly this comparison rather than to copy a number.

Be aware of what the extra Gaussians buy at multi-galaxy scale specifically. Every Gaussian you add to `lens_0`'s
basis is another column in the linear system that `lens_1`'s Gaussians can be degenerate with — the max
cross-deflector coupling of 0.9877 quoted above is with 20 each. Adding Gaussians past the point where they
improve the fit does not leave the model unchanged; it makes the two galaxies harder to separate.

__Model Composition__

One `lens_i` per deflector in a loop over the centres, the shear in its own `shear_galaxy`, and the source
separate — the standard multi-galaxy composition of `multi_galaxy/modeling.py`.

`al.model_util.mge_model_from` builds each basis. Its arguments:

 - `mask_radius` — sets the largest Gaussian `sigma`.
 - `total_gaussians` — how many Gaussians, chosen above.
 - `gaussian_per_basis` — how many *groups* of Gaussians, each with its own free ellipticity. Two groups let the
   ellipticity vary with radius, which is exactly the twist this dataset contains, so we use 2 for the deflectors
   and 1 for the source.
 - `centre` / `centre_sigma` — the basis centre and how far it may move.
"""
total_gaussians = 20

# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=total_gaussians,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
        centre_sigma=0.1,
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
    total_gaussians=total_gaussians,
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

Note how few free parameters each deflector's 20-Gaussian basis contributes — no `intensity`, no `sigma`, just the
centre and the ellipticity of each group.
"""
print(model.info)

"""
__Search__

The lens model is fitted using the nested sampling algorithm Nautilus, with 200 live points as in
`multi_galaxy/modeling.py` — a multi-galaxy parameter space is multi-modal, and too few live points settle into a
local maximum with the two Einstein radii mis-apportioned.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features",
    name="multi_gaussian_expansion",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, described in `multi_galaxy/modeling.py` and in full in
`autolens_workspace/*/guides/results`.

Two checks specific to a two-MGE model:

 - **Look at each deflector's reconstructed light separately**, not just the summed model image. The summed image
   can look excellent while the flux has been mis-apportioned between the galaxies, because that is precisely what
   a 0.9877 cross-coupling permits.
 - **Check for a galaxy whose basis has gone faint and whose neighbour has gone bright.** That is the
   redistribution failure mode, and it biases the flux ratio without hurting the residuals.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/multi_gaussian_expansion/fit.py` — the same composition without a search, where the
   Sersic-versus-MGE comparison above is reproduced directly.
 - `multi_galaxy/features/multi_gaussian_expansion/likelihood_function.py` — where the basis enters the
   likelihood, and the 41 x 41 curvature matrix.
 - `multi_galaxy/features/multi_gaussian_expansion/source_science.py` — integrating an MGE to get a luminosity,
   which is what the scaling-relation tier needs.
 - `multi_galaxy/features/multi_gaussian_expansion/slam.py` — the SLaM pipeline on this dataset.
 - `multi_galaxy/features/linear_light_profiles` — the single-profile case, and where the 0.296 comes from.
 - `imaging/features/multi_gaussian_expansion` — the galaxy-scale walkthrough, with the fuller API tour.
"""
