"""
Modeling Features: Group Mass Stellar Dark
==========================================

A group-scale strong lens where each main lens galaxy carries a decomposed mass model: a stellar component
tied to the galaxy's own light via a mass-to-light ratio, plus a separately-parameterized dark matter halo.
The total lens-plane deflection is the sum, over every main lens galaxy, of stellar + dark contributions,
plus a single external shear attached to `lens_0` representing the group-wide shear field.

This script fits a group lens model where each main lens galaxy is decomposed into stellar + dark components.
Per-galaxy decomposition is the standard tool for studying mass-to-light variation across a group environment.

__Practical Use: Read This First__

This script is a tutorial. It produces a working fit by "cheating" — every prior is initialised at the true
simulator value, narrowed by a small Gaussian. On real data this is impossible, and a single Nautilus search
on a group-scale decomposed mass model would almost certainly converge to a local maximum. Two effects compound:

 - the per-galaxy `mass_to_light_ratio` couples each bulge's light and stellar mass, creating tight parameter
   degeneracies between the light parameters and the mass deflection;
 - every main lens galaxy adds its own stellar + dark contribution, so the deflection field is the sum of
   many independent components which a single search struggles to disentangle without good starting points.

The script you will actually use to fit a group decomposed-mass model on real data is
`autolens_workspace/scripts/group/features/advanced/mass_stellar_dark/chaining.py`, which runs two chained
non-linear searches: the first fits each lens galaxy's bulge as a pure light profile (no stellar mass
coupling, no dark NFW), the second reintroduces the stellar-mass coupling and adds the dark NFW per galaxy
with priors carried over from search 1.

For production-quality modeling, see `slam.py` in the same directory, which uses the `MASS_LIGHT_DARK` SLaM
pipeline.

Read this script to understand the model composition API, then jump to `chaining.py`.

__Contents__

- **Model:** Compose the lens model fitted to the data.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Main Lens Centres:** Load the centres of the two main lens galaxies from JSON.
- **Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
- **Model Cookbook:** A full description of model composition is provided by the model cookbook.
- **Search:** Configure the non-linear search used to fit the model.
- **Analysis:** Create the Analysis object that defines how the model is fitted to the data.
- **VRAM:** The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the.
- **Run Time:** Profiling the expected run time of the model-fit.
- **Result:** Overview of the results of the model-fit.
- **Wrap Up:** Summary of the script and next steps.

__Model__

This script fits an `Imaging` dataset of a 'group-scale' strong lens with a model where:

 - Each main lens galaxy's light and stellar mass is a linear `Sersic` [7 parameters per galaxy].
 - Each main lens galaxy's dark matter mass distribution is a `NFWSph` aligned with that galaxy's bulge centre.
 - The first main lens galaxy additionally carries an `ExternalShear` [2 parameters].
 - The source galaxy's light is a Multi Gaussian Expansion.

For two main lens galaxies, the lens-plane carries (7 + 4) * 2 = 22 free parameters, plus 2 for the shear,
plus the source MGE parameters.

Note that for each main lens galaxy's stellar light and mass, we use a "light and mass profile" via the `.lmp`
package. This profile simultaneously acts like a light and mass profile.

__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Start Here Notebook__

If any code in this script is unclear, refer to:

 - `autolens_workspace/scripts/group/start_here.py` — the canonical group modeling walkthrough.
 - `autolens_workspace/scripts/imaging/features/advanced/mass_stellar_dark/modeling.py` — the single-galaxy
   decomposed-mass walkthrough.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the strong lens dataset `mass_stellar_dark` via .fits files.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/mass_stellar_dark/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
__Main Lens Centres__

Load the centres of the two main lens galaxies from JSON. These centres are fixed on each galaxy's bulge and
dark profile in the model below.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask__

Define a 3.7" circular mask, which includes both main lens galaxies and the lensed source emission.
"""
mask_radius = 3.7

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Over Sampling__

Apply adaptive over sampling at each main lens galaxy centre, so the stellar mass-to-light coupling is
evaluated accurately at the peak of each bulge.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

We compose a lens model where:

 - Each main lens galaxy's light and stellar mass is a linear `Sersic` [7 parameters per galaxy].
 - Each main lens galaxy's dark matter mass distribution is a `NFWSph` whose centre is fixed to the bulge
   centre (i.e. that galaxy's `main_lens_centres` entry) [3 parameters per galaxy].
 - The first main lens galaxy additionally carries an `ExternalShear` [2 parameters].
 - The source galaxy's light is a Multi Gaussian Expansion.

The bulge and dark centres are fixed (no priors) because the centres are determined externally (e.g. by the
GUI used in `group/start_here.py`).

__List-Based Model Composition__

For group-scale lenses, we compose the lens-plane model via a `for i, centre in enumerate(main_lens_centres)`
loop. Each main lens galaxy is created in a loop and stored in a dictionary as `lens_0`, `lens_1`, etc. This
API scales naturally to groups with any number of main lens galaxies.

Only the first lens galaxy (`lens_0`) carries an `ExternalShear`, as the group system has one overall external
shear.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):
    bulge = af.Model(al.lmp.Sersic)
    bulge.centre = (centre[0], centre[1])

    dark = af.Model(al.mp.NFWSph)
    dark.centre = (centre[0], centre[1])

    galaxy_kwargs = dict(redshift=0.5, bulge=bulge, dark=dark)

    if i == 0:
        galaxy_kwargs["shear"] = af.Model(al.mp.ExternalShear)

    lens_dict[f"lens_{i}"] = af.Model(al.Galaxy, **galaxy_kwargs)

# Source:

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Overall Model:

model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen
refer to `start_here.ipynb` for a description of how to fix this).
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start_here.py` for a full
description).
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="mass_stellar_dark",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=2000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM
required by a model.

Deflection angle calculations of stellar mass models and dark matter mass models can use techniques which
store more data in VRAM than other methods.

Given VRAM use is an important consideration for group-scale lenses (which carry many mass components), we
print out the estimated VRAM required for this model-fit.
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Time__

The likelihood evaluation time for analysing decomposed stellar and dark matter mass models is longer than for
total mass models like the isothermal or power-law. This is because the deflection angles of these mass
profiles are more expensive to compute, requiring a Gaussian expansion or numerical calculation.

For a group lens with multiple main galaxies, each galaxy adds its own pair of (stellar, dark) deflection
evaluations, scaling the per-likelihood cost roughly linearly with the number of main lens galaxies.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output
folder for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen
refer to `start_here.ipynb` for a description of how to fix this).
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

These plots show that the per-galaxy decomposed stars + dark matter model recovers the ray-traced source.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

These examples include a results API with specific tools for visualizing and analysing decomposed mass models,
for example 1D plots which separately show the density of stars and dark matter as a function of radius for
each main lens galaxy.

__Wrap Up__

Group-scale decomposed mass models have advantages and disadvantages compared to total mass models.

The model which is best suited to your needs depends on the science you are hoping to undertake and the
quality of the data you are fitting.

In general, it is recommended that you first get fits going using total mass models, because they are simpler
and make fewer assumptions regarding how light is tied to mass. Once you have robust results, decomposed mass
models can then be fitted and compared in order to gain deeper insight into the per-galaxy stellar and dark
contributions across the group.
"""
