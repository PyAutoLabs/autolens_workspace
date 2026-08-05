"""
Start Here: Multi Galaxy
========================

Multi-galaxy strong lenses have **two or more galaxies of comparable mass which both contribute significantly to
the lensing** of a single background source. Neither galaxy is a minor perturber — they are co-dominant
deflectors, and every one of them gets its own free light and mass model.

This script shows you how to model a multi-galaxy lens system using **PyAutoLens** with as little setup as
possible. In about 15 minutes you'll be able to point the code at your own FITS files and fit your first
multi-galaxy lens.

__Which Regime Is My Lens?__

PyAutoLens organises lenses above the single-galaxy scale into a ladder of three regimes. Every group and cluster
is a multi-galaxy system, but not vice versa — what changes as you climb is first the mass model, then the entire
analysis strategy:

 - **Multi galaxy** (this package): 2+ co-dominant galaxies (individual halos ~10^11-10^13 M_sun), NO shared
   dark-matter halo. One free mass model per deflector + external shear. The source is a single extended galaxy,
   reconstructed at pixel level from CCD imaging — the **standard extended-source workflow**, unchanged
   from `imaging/`.

 - **Group** (`group/start_here.ipynb`): a dominant group-scale halo (~10^13-10^14 M_sun) enters as an *explicit
   modelling choice*, and the galaxies split into tiers (main / extra / scaling galaxies, the latter tied to a
   luminosity scaling relation — with tidally truncated dPIE members in the Lenstool-convention workflow of
   `group/features/group_halo`). Still one extended source, still the same
   pixel-level `AnalysisImaging` workflow — the sophistication moves into the mass model.

 - **Cluster** (`cluster/start_here.ipynb`): the mass framework is the same as a group's (host halo(s) + many
   truncated members on scaling relations), but the **analysis itself changes**: dozens of sources at different
   redshifts are fitted as point-source multiple-image positions (`AnalysisPoint` + a factor graph, multi-plane
   ray tracing), and the lens galaxies' light is not modeled at all.

If your system has one dominant lens galaxy, start with `imaging/start_here.ipynb` instead.

__The Example System__

We fit a simulated merging pair of lens galaxies modeled on **SDSS J1011+0143** (Shu et al. 2016, ApJ 820, 43,
arXiv:1602.02927) — one of the cleanest co-dominant pairs known: two early-type galaxies separated by ~4.2 kpc
(~0.9") at z=0.331, lensing a z=2.701 Lyman-alpha emitter into a wide Einstein cross of radius ~1.8". The
published model of the real system is exactly the model this script fits: two isothermal mass profiles plus
external shear, with an extended source. Its headline science — kiloparsec-scale offsets between each galaxy's
mass and light, a potential probe of dark-matter self-interaction — is a measurement only a multi-deflector
model can make.

__Contents__

- **JAX:** JAX acceleration for fast GPU/CPU model-fitting.
- **Google Colab Setup:** Run this example in a web browser without local installation.
- **Imports:** Import the required Python libraries.
- **Dataset:** Load (auto-simulating if absent) and plot the strong lens dataset.
- **Extra Galaxy Removal:** Remove the light of a nearby galaxy that is not part of the strong lens.
- **Main Lens Galaxies:** Every deflector in a multi-galaxy lens is a main lens galaxy.
- **Masking:** Mask the region of the image the model is fitted to.
- **Model:** Compose the lens model — one free light + mass model per deflector.
- **Model Fit:** Perform the model-fit using the search and analysis.
- **Iterations Per Update:** How often the search writes the maximum likelihood model to hard-disk.
- **Live Visual Update:** Opt-in live matplotlib window (scripts) or Jupyter cell refresh (notebooks) during the fit.
- **Result:** Overview of the results of the model-fit.
- **Extra Galaxy Removal GUI:** A GUI for creating the extra-galaxies mask for your own data.
- **Model Your Own Lens:** Adapting this script to your own imaging data.
- **Simulator:** Simulate your own multi-galaxy strong lens imaging.
- **Sample:** Pointer to simulating many multi-galaxy lenses at once.
- **Wrap Up:** Summary and the ladder up to groups and clusters.

__JAX__

PyAutoLens runs multi-galaxy model-fits on JAX by default — `al.AnalysisImaging` auto-enables `use_jax=True` if
you installed `autolens[jax]`. The multi-galaxy deflection sum vectorises cleanly, so fits benefit substantially
from GPU acceleration. Expect ~10-20 minutes on a GPU for this example.

For the broader JAX principles, see the top-level `autolens_workspace/start_here.py` `__JAX__` section.

__Google Colab Setup__

The `start_here` examples are runnable on Google Colab without local PyAutoLens installation. The block below
installs the dependencies and downloads the example files if you're on Colab; running it locally is a no-op.
"""

try:
    import google.colab
except ImportError:
    from autolens import setup_colab as _setup_colab
else:
    import importlib
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autonerves", "--no-deps"]
    )
    _setup_colab = importlib.import_module("autonerves.setup_colab")

_setup_colab.for_autolens(
    raise_error_if_not_gpu=False  # Switch to True to require GPU on Colab.
)

"""
__Imports__
"""
from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the multi-galaxy dataset `simple`: HST-resolution (0.05"/pixel) CCD imaging of the simulated merging-pair
lens. If the dataset is not found on disk it is simulated automatically by `multi_galaxy/simulator.py`, so this
script runs with no manual setup.
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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Extra Galaxy Removal__

There may be regions of an image that have signal near the lens and source that is from other galaxies not
associated with the strong lens we are studying. The emission from these images will impact our model fitting and
needs to be removed from the analysis.

In a multi-galaxy field this step carries a judgement the single-galaxy case does not: some of the other galaxies
in the image *are* part of the lens. A co-dominant deflector belongs in the model as a main lens galaxy; a
contaminant belongs here, removed from the analysis. The test is whether it contributes significantly to the
lensing of the source — not simply whether it is bright or nearby.

This `mask_extra_galaxies` is used to prevent them from impacting a fit by scaling the RMS noise map values to
large values. This mask may also include emission from objects which are not technically galaxies, but blend with
the galaxies we are studying in a similar way. Common examples of such objects are foreground stars or emission
due to the data reduction process.

After performing lens modeling to this strong lens, the script further down provides a GUI to create such a mask
for your own data, if necessary.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Main Lens Galaxies__

For a multi-galaxy lens, two or more galaxies' light and mass all contribute significantly to the lensing of the
source. We call these the "main lens galaxies" and model each one individually — and unlike the group and cluster
regimes, there are no other tiers: no extra galaxies, no scaling relations, no host halo. Every deflector is a
main lens galaxy.

We load the centres of the main lens galaxies from a `.json` file in the dataset folder. These centres are used
to initialize the model for each lens galaxy. For your own data, the centre-input GUI shown in
`group/start_here.ipynb` writes this file from mouse clicks on the image.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Masking__

Lens modeling does not need to fit the entire image, only the region containing the lens and source light. We
define a circular mask around the system — for a multi-galaxy lens make sure it encloses the *combined* Einstein
ring (the lensed arcs wrap around the pair as a whole), not just one galaxy's light.

We also oversample the central pixels of each galaxy, which improves modeling accuracy without adding
unnecessary cost far from the lens.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

To perform lens modeling we must define a lens model, describing the light profiles of every lens galaxy and of
the source galaxy, and the mass profile of every lens galaxy.

A brilliant lens model to start with is one which uses a Multi Gaussian Expansion (MGE) to model the lens and
source light, and a Singular Isothermal Ellipsoid (SIE) plus shear to model the lens mass. The only thing the
multi-galaxy regime changes is *how many* of them there are: **one free light and mass model per co-dominant
deflector**, plus a single overall external shear.

That shear is given its own entry in the model, at the system centre (0.0", 0.0"), rather than being attached to one
of the deflectors as it is in the galaxy-scale examples. The shear describes the tidal field of structure outside the
system, so it belongs to the system as a whole and there is no principled reason to hang it off a particular galaxy.

Full details of why this model is so good are provided in the main workspace docs, but in a nutshell it provides
an excellent balance of being fast to fit, flexible enough to capture complex galaxy morphologies and providing
accurate fits to the vast majority of strong lenses. It is also the parameterization the real SDSS J1011+0143
pair was published with.

The MGE model composition API is quite long and technical, so we simply load the MGE models for the lens and
source galaxies below via a utility function `mge_model_from` which hides the API to make the code in this
introduction example easy to read. We then use the PyAutoLens Model API to compose the overall lens model.

`multi_galaxy/modeling.py` explains all of this in detail — why MGEs and linear light profiles are used, why the
mass profiles are untruncated in this regime, and how to upgrade each galaxy to a free density slope.

__List-Based Model Composition__

Each main lens galaxy is created in a loop over the main lens galaxy centres and stored as `lens_0`, `lens_1`,
etc. This list-based API scales to any number of co-dominant deflectors, and is the same API the group package
uses — so when you climb the ladder there is nothing to re-learn.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
        sigma_min=dataset.pixel_scales[0] / 10.0,
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

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, shear_galaxy=shear_galaxy, source=source)
)

"""
Print the model to see its free parameters — note `lens_0` and `lens_1` each carry their own free mass model,
the signature of the multi-galaxy regime, and that the shear sits in its own `shear_galaxy` entry rather than
inside either of them.
"""
print(model.info)

"""
__Model Fit__

We fit the data using `MultiStartProdigy`, a multi-start gradient optimizer, and an `AnalysisImaging` object,
which defines the `log_likelihood_function` fitted to the imaging data.

This is the point worth pausing on: `AnalysisImaging` is the **extended-source, pixel-level analysis** — the same
object used for single-galaxy (`imaging/`) and group-scale fits. The multi-galaxy regime changes the mass model,
not the analysis. Only at cluster scale does the analysis itself switch, to `AnalysisPoint` fits of
multiple-image positions.

__Multi Start Gradient Optimization__

`MultiStartProdigy` launches `n_starts` independent optimizations from broad starting points spread across the
parameter space, all of which descend the likelihood in parallel via `jax.vmap`, and returns the best one. A
single starting point would frequently get stuck in a local maximum — and a multi-galaxy mass model, with several
co-dominant deflectors, has a particularly multi-modal parameter space. Running a wide population of starts is
what makes a gradient optimizer reliable here (the GIGA-Lens approach, Gu, Huang et al. 2022, arXiv:2202.07663).
Prodigy is *learning-rate free* (Mishchenko & Defazio 2024, arXiv:2306.06101), so there is nothing to tune.

__Posterior__

`MultiStartProdigy` is a maximum a posteriori (MAP) optimizer: it returns the **single best-fit lens model** and
nothing else — no posterior, no error bars, no covariances between parameters.

To get uncertainties, run `autolens_workspace/scripts/multi_galaxy/modeling.py`, which fits this same model with
the nested sampling algorithm `Nautilus` and returns the **full posterior**. Use the fast optimizer here to check
your model and data are sensible, then `Nautilus` when you need results you can quote.

__JAX__

`AnalysisImaging` defaults to `use_jax=True` when JAX is installed (set explicitly below for clarity). The search
driver wraps the likelihood in `jax.vmap(jax.jit(...))` internally — every one of the `n_starts` parameter vectors
evaluates in parallel in a single GPU call. Watch for `JAX: Applying vmap and jit to likelihood function -- may
take a few seconds.` in the log; that's the JIT compile starting, after which evaluations re-use the compiled
trace.

This is also what makes the multi-galaxy regime cheap to scale. Each extra deflector adds another deflection field
to the sum, which vectorises cleanly, so a two-galaxy fit costs little more per likelihood evaluation than a
single-galaxy one — the cost of extra galaxies is in the *dimensionality* of parameter space, not the arithmetic.

Force NumPy with `use_jax=False` (or `PYAUTO_DISABLE_JAX=1`) when debugging — NumPy stack traces are easier to
read than JAX traces.

__Iterations Per Update__

Every `iterations_per_quick_update`, the search outputs the maximum likelihood model and its best fit image to
hard-disk (as `fit.png` in the output folder).

This process takes around ~10 seconds, so we don't want it to happen too often so as to slow down the overall fit,
but we also want it to happen frequently enough that we can track the progress.

For this search the unit is a gradient step, so the value of 50 below gives us an update every 50 steps. The fit
usually converges well before the `n_steps` ceiling, so expect a handful of updates rather than `n_steps / 50`.

__Live Visual Update__

By default the quick-update image is only written to disk. Set `live_visual_update=True` to also push it to a
live display surface:

- **Python script** — a matplotlib window opens automatically and refreshes with each quick update, so you can
  watch the fit converge without leaving your terminal.
- **Jupyter / Colab notebook** — the cell that ran `search.fit(...)` shows a single self-updating image that
  refreshes in place every `iterations_per_quick_update`.

Watching a multi-galaxy fit converge is genuinely informative: the deflectors' Einstein radii trade off against
one another early on, so the arcs snap into place as the pair's mass ratio settles.

The disk write (`fit.png`) always happens regardless of this flag. Set it to `False` (the default) if you just
want the on-disk output, or if you are running in a headless environment (e.g. an HPC cluster).
"""
search = af.MultiStartProdigy(
    path_prefix=Path("multi_galaxy"),  # The path where results are stored.
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_starts=48,  # The number of independent optimizations run in parallel, increase for more complex models.
    batch_size=None,  # Starts evaluated at once: `None` vmaps all 48 together, which is fastest but allocates the whole batched gradient; set an integer (e.g. 4) if you hit an out-of-memory error.
    n_steps=300,  # The maximum gradient steps per start; the search stops early once the best fit stops improving.
    iterations_per_quick_update=50,  # Every N steps the max likelihood model is visualized and output to hard-disk.
    live_visual_update=True,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,  # JAX uses GPUs for acceleration if available, else multithreaded CPUs.
)

print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

Now this is running you should checkout the `autolens_workspace/output` folder, where many results of the fit
are written in a human readable format (e.g. .json files) and .fits and .png images of the fit are stored.

When the fit is complete, we can print the results by printing `result.info`.
"""
print(result.info)

"""
The result also contains the maximum likelihood lens model which can be used to plot the best-fit lensing
information and fit to the data.

For a multi-galaxy lens the tracer subplot is worth a close look: the critical curve is that of the *combined*
mass distribution, so it wraps around the pair as a whole rather than encircling either galaxy.
"""
aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
The result object contains pretty much everything you need to do science with your own strong lens, but details
of all the information it contains are beyond the scope of this introductory script. The `guides` and `result`
packages of the workspace contain all the information you need to analyze your results yourself.

__Extra Galaxy Removal GUI__

The model-fit above removed a region of the image to the north-east of the lens pair, which contains light from
another galaxy not associated with the strong lens system.

This GUI below provides the tool you need to produce such a mask for your own data, if necessary, with which you
can then use the `apply_noise_scaling` function.

Remember the multi-galaxy judgement when you use it: scribble over contaminants only. A galaxy that is deflecting
the source belongs in `main_lens_centres.json` and gets modeled, not masked.

Note that this **overwrites** `mask_extra_galaxies.fits` in the dataset folder — the same file loaded by the
`__Extra Galaxy Removal__` step above. That is intentional: what you draw here becomes the mask every multi_galaxy
example uses. If you would rather keep the shipped mask, change the `file_path` below or re-run
`multi_galaxy/simulator.py` to regenerate it.
"""
cmap = "jet"

try:
    scribbler = al.Scribbler(
        image=dataset.data.native,
        cmap=cmap,
        brush_width=0.04,
        mask_overlay=mask,
    )
    mask_gui = scribbler.show_mask()
    mask_gui = al.Mask2D(mask=mask_gui, pixel_scales=dataset.pixel_scales)

    aplt.fits_array(
        array=mask_gui,
        file_path=dataset_path / "mask_extra_galaxies.fits",
        overwrite=True,
    )
except Exception as e:
    print(
        """
        Problem loading GUI, probably an issue with TKinter or your matplotlib TKAgg backend.

        You will likely need to try and fix or reinstall various GUI / visualization libraries, or try
        running this example not via a Jupyter notebook.

        There are also manual tools for performing this task in the workspace.
        """
    )
    print()
    print(e)

"""
__Model Your Own Lens__

If you have your own multi-galaxy lens imaging data, you are now ready to model it yourself by adapting the code
above and simply inputting the path to your own .fits files into the `Imaging.from_fits()` function.

A few things to note, with full details on data preparation provided in the main workspace documentation:

- Supply your own CCD image, PSF, and RMS noise-map.
- Double-check `pixel_scales` for your telescope/detector.
- Adjust the mask radius to enclose the combined Einstein ring and all lens galaxies — this is the step most
  often got wrong for a multi-galaxy lens, because the arcs wrap around the pair and extend further than either
  galaxy's own light.
- Decide which galaxies are co-dominant deflectors and which are contaminants. Provide the centres of the
  deflectors in a `main_lens_centres.json` file (the GUI in `group/start_here.ipynb` writes this from mouse
  clicks), and remove the contaminants with the extra galaxies mask GUI above.
- Start with the default model — one MGE + SIE per deflector works very well for pretty much all multi-galaxy
  lenses!

__Simulator__

Let's now switch gears and simulate our own multi-galaxy strong lens imaging. This is a great way to:

- Practice multi-galaxy lens modeling before using real data.
- Build large training sets (e.g. for machine learning).
- Test how well a pair of deflectors can actually be disentangled at a given resolution and signal-to-noise.

To do this we need to define a 2D grid of (y,x) coordinates in the image-plane. This grid is where we'll evaluate
the light from the lens and source galaxies. We oversample the centre of *each* lens galaxy, not just one.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

simulator_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=simulator_lens_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
We now define a `Tracer` — this is the key object that combines all galaxies in the system and computes how light
rays are deflected.

- Each lens galaxy has both light (a Sersic bulge) and mass (an isothermal profile).
- The system's single external shear is held separately, at the system centre.
- The source galaxy has its own light (a SersicCore profile).

The two lens galaxies' Einstein radii (1.0" and 0.8") are deliberately comparable — that is what makes this a
multi-galaxy lens rather than a single lens with a minor perturber. Note also the small offsets between each
galaxy's light centre and its mass centre, which is the kind of structure an interacting pair really shows.

Together they define a strong lens system. The tracer will "ray-trace" our grid through this mass distribution,
summing both deflectors' deflection fields, and generate a lensed image.
"""
lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

shear_galaxy_simulated = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

tracer = al.Tracer(
    galaxies=[lens_galaxy_0, lens_galaxy_1, shear_galaxy_simulated, source_galaxy]
)

"""
Plotting the tracer's image gives us a "perfect" view of the multi-galaxy lens system, before adding telescope
effects.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Simulator__

The image above does not represent real CCD imaging data, as it does not include the blurring due to the telescope
optics or sources of noise.

The `SimulatorImaging` class simulates these two key properties of real imaging data, which we use below to create
realistic imaging of the multi-galaxy lens system.

The units of the image are arbitrary, with the workspace providing guides on how to convert to physical units for
lens simulations.

The code below performs the simulation, plots the simulated imaging data and outputs it to .fits files.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11),  # The 2D shape of the PSF array.
    sigma=0.08,  # The size of the Gaussian PSF, where FWHM = 2.35 * sigma.
    pixel_scales=grid.pixel_scales,  # The pixel scale of the PSF, matches the image's pixel scale.
)

simulator = al.SimulatorImaging(
    exposure_time=900.0,  # The exposure time of the observation, increases the S/N of the image.
    psf=psf,  # The PSF which blurs the image.
    background_sky_level=0.1,  # The background sky level of the image, increases the noise.
    add_poisson_noise_to_data=True,  # Whether Poisson noise is added to the image or not.
)

dataset_simulated = simulator.via_tracer_from(tracer=tracer, grid=grid)

simulated_dataset_path = Path("dataset") / "multi_galaxy" / "simulated_lens"

aplt.fits_imaging(
    dataset=dataset_simulated,
    data_path=simulated_dataset_path / "data.fits",
    psf_path=simulated_dataset_path / "psf.fits",
    noise_map_path=simulated_dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
We can now inspect the simulated dataset: image, noise-map and PSF. These have been written to FITS files in
exactly the same format as real data, so you can immediately try fitting the simulated dataset with the modeling
workflow above — including writing your own `main_lens_centres.json` for the two galaxies you just simulated.
"""
aplt.subplot_imaging_dataset(dataset=dataset_simulated)
aplt.plot_array(array=dataset_simulated.data, title="Data")

"""
__Sample__

Often we want to simulate *many* strong lenses — for example, to train a neural network or to explore
population-level statistics. For multi-galaxy lenses this is especially useful, because the pair's mass ratio and
separation are exactly the population properties you want to vary.

`autolens_workspace/*/multi_galaxy/simulator_sample.py` does this: it draws random co-dominant pairs and simulates
a sample of datasets from them.

__Wrap Up__

This script has shown how to model a multi-galaxy strong lens: the standard extended-source imaging workflow,
with one free light and mass model per co-dominant deflector — and how to simulate your own.

Where to go next:

- `autolens_workspace/*/multi_galaxy/modeling`: the full modeling API and how to customize the fit, and the fit
  that returns a posterior rather than a single best-fit model.
- `autolens_workspace/*/multi_galaxy/simulator`: how the example dataset was simulated.
- `autolens_workspace/*/multi_galaxy/simulator_sample`: simulating many multi-galaxy lenses at once.
- `autolens_workspace/*/multi_galaxy/fit`: the anatomy of a multi-galaxy fit — residuals, chi-squared, likelihood,
  and each deflector's share of the summed deflection field.
- `autolens_workspace/*/multi_galaxy/source_science`: the source's flux and magnification behind two deflectors.
- `autolens_workspace/*/multi_galaxy/likelihood_function`: a step-by-step guide to the likelihood function.
- `autolens_workspace/*/multi_galaxy/features`: extensions — extra galaxies, scaling galaxies (untruncated),
  pixelized source reconstructions.
- `autolens_workspace/*/group`: the next rung of the ladder — tiered galaxy populations and the group halo as an
  explicit modelling choice.
- `autolens_workspace/*/cluster`: the top rung — point-source constraints, many sources, multi-plane ray tracing.
- `autolens_workspace/guides/results`: loading and analyzing the results of your fits.
"""
