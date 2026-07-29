"""
Start Here: Group
=================

Group scale lenses typically have multiple lens galaxies whose light and mass all contribute significantly to the
lensing of the source galaxy. Groups typically also have just one lensed source.

This script shows you how to model a group lens system using **PyAutoLens** with as little setup
as possible. In about 15 minutes you'll be able to point the code at your own FITS files and
fit your first group-scale lens.

We focus on a *group-scale* lens (multiple lens galaxies nearby). If you have a single
lens galaxy, see the `imaging/start_here.ipynb` example. If your lens has 2+ co-dominant galaxies but no
dominant group dark-matter halo, see the `multi_galaxy/start_here.ipynb` example. If your system has many lens
and source galaxies, see the `cluster/start_here.ipynb` example.

This example uses Euclid CCD imaging data. The model uses the group regime's **three-tier galaxy API** — the
signature composition of group- and cluster-scale modeling:

 - **Main lens galaxies** (here: 2 — the central galaxy and a bright companion just 0.4" away): the dominant
   lenses, each with a free MGE light model, `Isothermal` mass and (on `lens_0`) an `ExternalShear`.
 - **Extra galaxies** (here: 1): a nearby companion inside the mask, with its own MGE light model and a
   bounded free `IsothermalSph` mass at its observed centre.
 - **Scaling galaxies** (here: 5): further-out galaxies whose light sits outside the mask; mass-only
   `IsothermalSph` profiles tied to their catalogue luminosities through one shared free normalization —
   the whole tier adds a single parameter, however many galaxies it holds.

This runs in roughly 10-20 minutes on a good GPU; more complex groups with more galaxies fit with the identical
workflow. A fourth, optional mass component — the group-scale dark-matter halo — is deliberately NOT in this
default model: whether a group needs one is an explicit modelling choice, and `group/features/group_halo`
is the tutorial that teaches how to make it (fitting the same data with and without the halo).

__Contents__

- **JAX:** JAX acceleration for fast GPU/CPU model-fitting.
- **Google Colab Setup:** The introduction `start_here` examples are available on Google Colab, which allows you to run them.
- **Imports:** Import the required Python libraries.
- **Dataset:** Load and plot the strong lens dataset.
- **Main Lens Galaxies:** For a group-scale lens, we have multiple lens galaxies whose light and mass all contribute.
- **Masking:** Lens modeling does not need to fit the entire image, only the region containing lens and source.
- **Model:** Compose the lens model fitted to the data.
- **Model Fit:** Perform the model-fit using the search and analysis.
- **Live Visual Update:** Push the quick-update image to a live display surface.
- **Result:** Overview of the results of the model-fit.
- **Centre Input GUI:** The centres of the main lens galaxies above were loaded from a .json file, which was created using.
- **Model Your Own Lens:** If you have your own strong lens imaging data, you are now ready to model it yourself by adapting.
- **Wrap Up:** Summary of the script and next steps.

__JAX__

PyAutoLens runs group-scale model-fits on JAX by default — `al.AnalysisImaging`
auto-enables `use_jax=True` if you installed `autolens[jax]`. Group fits
benefit substantially from GPU acceleration (the multi-galaxy deflection
sum is the dominant cost). Expect 5-30 minutes on GPU vs hours on pure
NumPy.

For the broader JAX principles, see the top-level
`autolens_workspace/start_here.py` `__JAX__` section. For per-dataset
simulator / fit / likelihood patterns shared with imaging, see
`scripts/imaging/start_here.py` and its companions.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoLens installation.

The code below sets up your environment if you are using Google Colab, including installing autolens and downloading
files required to run the notebook. If you are running this script not in Colab (e.g. locally on your own computer),
running the code will still check correctly that your environment is set up and ready to go.
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
    raise_error_if_not_gpu=False  # Switch to False for CPU Google Colab
)

"""
__Imports__

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
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

We begin by loading the dataset. Three ingredients are needed for lens modeling:

1. The image itself (CCD counts).
2. A noise-map (per-pixel RMS noise).
3. The PSF (Point Spread Function).

Here we use HST imaging of a Euclid group-scale strong lens. Replace these FITS paths with your own to
immediately try modeling your data.

The `pixel_scales` value converts pixel units into arcseconds. It is critical you set this
correctly for your data.
"""
dataset_name = "102021990_NEG650312660474055399"
dataset_path = Path("dataset") / "group" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Main Lens Galaxies__

For a group-scale lens, we have multiple lens galaxies whose light and mass all contribute significantly to the
lensing of the source galaxy. We call these the "main lens galaxies" and model each one individually.

We load the centres of the main lens galaxies from a `.json` file contained in the dataset folder. These centres
are used to initialize the model for each lens galaxy.

After modeling the data, this example will provide a GUI for you to determine the centres of the lens galaxies in
your own data, if they are not already known.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Extra Galaxies + Scaling Galaxies__

The other two tiers load from the dataset folder alongside the main centres:

 - `extra_galaxies_centres.json` — the companion ~3" from the main pair, clicked with the same GUI as the
   main centres. (The brighter companion just 0.4" from the central galaxy is in the *main* tier: it is a
   distinct, bright deflector, so it gets full model freedom rather than an extra galaxy's restricted one.)
 - `scaling_galaxies.csv` — the `y, x, luminosity` catalogue of the five further-out galaxies (the same
   three-column schema used at cluster scale). For this example the catalogue was derived from this image
   with simple peak detection and 1"-aperture photometry; for your own group, use your survey's photometry
   catalogue — only luminosity *ratios* relative to the fixed reference below enter the model, so any
   consistent units work.
"""
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

scaling_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

scaling_centres = scaling_table.centres.in_list
scaling_luminosities = scaling_table.luminosities

"""
__Masking__

Lens modeling does not need to fit the entire image, only the region containing lens and
source light and all lens galaxies in the group. We therefore define a circular mask around all galaxies.

- Make sure the mask fully encloses the lensed arcs and all lens galaxies.
- Avoid masking too much empty sky, as this slows fitting without adding information.

We'll also oversample the central pixels of each galaxy, which improves modeling accuracy without adding
unnecessary cost far from the lens.
"""
mask_radius = 3.7

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

# Over sampling is important for accurate lens modeling, but details are omitted
# for simplicity here, so don't worry about what this code is doing yet!

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres) + list(extra_galaxies_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

To perform lens modeling we must define a lens model, describing the light profiles of the lens and source galaxies,
and the mass profile of each lens galaxy.

A brilliant lens model to start with is one which uses a Multi Gaussian Expansion (MGE) to model the lens and source
light, and a Singular Isothermal Ellipsoid (SIE) plus shear to model the lens mass.

Full details of why this model is so good are provided in the main workspace docs, but in a nutshell it
provides an excellent balance of being fast to fit, flexible enough to capture complex galaxy morphologies and
providing accurate fits to the vast majority of strong lenses. For group scale lenses, the MGE allows us to fit
the light of all lens galaxies without increasing the number of free parameters in the model.

__List-Based Model Composition__

For group-scale lenses, we compose the model using a list-based API. Each main lens galaxy is created in a loop
over the main lens galaxy centres, and stored in a dictionary as `lens_0`, `lens_1`, etc. This API scales naturally
to groups with any number of main lens galaxies.

Only the first lens galaxy (`lens_0`) carries an `ExternalShear`, as the group system has one overall external shear.

The extra galaxies are composed the same way (`extra_0`, `extra_1`, ...), each with a small MGE for its light and a
bounded free `IsothermalSph` mass fixed at its observed centre. The scaling galaxies sit outside the mask so their
light is not modeled; each carries a mass-only `IsothermalSph` whose einstein radius is tied to its catalogue
luminosity via `einstein_radius_ref * (L / L_ref) ** 0.5` — one shared free parameter for the whole tier.
(These are **untruncated** profiles, the PyAutoLens-native parameterization; the tidally truncated dPIE variant —
physically motivated once a group halo enters the model — is taught in `group/features/group_halo`.)

The MGE model composition API is quite long and technical, so we simply load the MGE models for the lens and source
below via a utility function `mge_model_from` which hides the API to make the code in this introduction example ready
to read. We then use the PyAutoLens Model API to compose the overall lens model.
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
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

# Extra Galaxies:

for i, centre in enumerate(extra_galaxies_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=1.0,
        total_gaussians=10,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = (centre[0], centre[1])
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    lens_dict[f"extra_{i}"] = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Scaling Galaxies (mass-only; one shared free normalization for the whole tier):

einstein_radius_ref = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

# A fixed fiducial reference luminosity (the analogue of Lenstool's mag0) — an explicit constant, NOT the
# sample maximum, so the normalization is invariant to which galaxies are placed in the tier.
reference_luminosity = 1.0

for i, (centre, luminosity) in enumerate(zip(scaling_centres, scaling_luminosities)):
    luminosity_ratio = float(luminosity) / reference_luminosity

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = einstein_radius_ref * luminosity_ratio**0.5

    lens_dict[f"scaling_{i}"] = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

"""
We can print the model to show the parameters that the model is composed of, which shows many of the MGE's fixed
parameter values the API above hided the composition of.

Note how the model lists all three tiers — `lens_0` (main), `extra_0`/`extra_1` (bounded companions) and
`scaling_0`...`scaling_4` (every einstein radius tied to the single shared `einstein_radius_ref`).
"""
print(model.info)

"""
__Model Fit__

We now fit the data with the lens model using `MultiStartProdigy`, a multi-start gradient optimizer which finds
the best-fit lens model quickly.

This requires an `AnalysisImaging` object, which defines the `log_likelihood_function` used by the search to fit
the model to the imaging data.

__Multi Start Gradient Optimization__

`MultiStartProdigy` launches `n_starts` independent optimizations from broad starting points spread across the
parameter space, all of which descend the likelihood in parallel via `jax.vmap`, and returns the best one. A
single starting point would frequently get stuck in a local maximum, which a group-scale model — several lens
galaxies plus an optional group halo — makes especially likely. Running a wide population of starts is what makes
a gradient optimizer reliable here (the GIGA-Lens approach, Gu, Huang et al. 2022, arXiv:2202.07663). Prodigy is
*learning-rate free* (Mishchenko & Defazio 2024, arXiv:2306.06101), so there is nothing to tune.

__Posterior__

`MultiStartProdigy` is a maximum a posteriori (MAP) optimizer: it returns the **single best-fit lens model** and
nothing else — no posterior, no error bars, no covariances between parameters.

To get uncertainties, run `autolens_workspace/scripts/group/modeling.py`, which fits this same model with the
nested sampling algorithm `Nautilus` and returns the **full posterior**. Use the fast optimizer here to check
your model and data are sensible, then `Nautilus` when you need results you can quote.

__JAX__

`AnalysisImaging` defaults to `use_jax=True`. Search driver wraps the
likelihood in `jax.vmap(jax.jit(...))`. Force NumPy with `use_jax=False`
(or `PYAUTO_DISABLE_JAX=1`) when debugging.

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce
an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.

__Live Visual Update__

By default the quick-update image is only written to disk. Set `live_visual_update=True` to also push it to a
live display surface:

- **Python script** — a matplotlib window opens automatically and refreshes with each quick update, so you can
  watch the fit converge without leaving your terminal.
- **Jupyter / Colab notebook** — the cell that ran `search.fit(...)` shows a single self-updating image that
  refreshes in place every `iterations_per_quick_update`.

The disk write (`fit.png`) always happens regardless of this flag. Set it to `False` (the default) if you just
want the on-disk output, or if you are running in a headless environment (e.g. an HPC cluster).
"""
search = af.MultiStartProdigy(
    path_prefix=Path("group"),  # The path where results and output are stored.
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_starts=48,  # The number of independent optimizations run in parallel, increase for more complex models.
    n_steps=300,  # The maximum gradient steps per start; the search stops early once the best fit stops improving.
    iterations_per_quick_update=50,  # Every N steps the max likelihood model is visualized and output to hard-disk.
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
The code below begins the model-fit. This will take around 10 minutes with a GPU, or 20-30 minutes with a CPU.

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce
an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

Now this is running you should checkout the `autolens_workspace/output` folder, where many results of the fit
are written in a human readable format (e.g. .json files) and .fits and .png images of the fit are stored.

When the fit is complex, we can print the results by printing `result.info`.
"""
print(result.info)

"""
The result also contains the maximum likelihood lens model which can be used to plot the best-fit lensing information
and fit to the data.
"""
aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
The result object contains pretty much everything you need to do science with your own strong lens, but details
of all the information it contains are beyond the scope of this introductory script. The `guides` and `result`
packages of the workspace contains all the information you need to analyze your results yourself.

__Centre Input GUI__

The centres of the main lens galaxies above were loaded from a .json file, which was created using a GUI where one
simply clicks the centres of the galaxies on the image.

For your own group lens, if you do not know the centres of the galaxies already, you can use the GUI below
to do this yourself. It will output a .json file in the dataset folder you can then load and use in the model above.
"""
search_box_size = (
    3  # Size of the search box to find the brightest pixel around your click
)

try:
    clicker = al.Clicker(
        image=dataset.data,
        pixel_scales=dataset.pixel_scales,
        search_box_size=search_box_size,
    )

    main_lens_centres = clicker.start(
        data=dataset.data,
        pixel_scales=dataset.pixel_scales,
    )

    al.output_to_json(
        file_path=dataset_path / "main_lens_centres.json",
        obj=main_lens_centres,
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

If you have your own strong lens imaging data, you are now ready to model it yourself by adapting the code above
and simply inputting the path to your own .fits files into the `Imaging.from_fits()` function.

A few things to note, with full details on data preparation provided in the main workspace documentation:

- Supply your own CCD image, PSF, and RMS noise-map.
- Ensure the primary lens galaxy is roughly centered in the image.
- Double-check `pixel_scales` for your telescope/detector.
- Adjust the mask radius to include all relevant light.
- Provide the centres of all main lens galaxies in a `main_lens_centres.json` file and of nearby
  companions in `extra_galaxies_centres.json` (the GUI below writes `main_lens_centres.json`; re-run it
  with the output `file_path` changed to write the extras file).
- Provide further-out galaxies in a `scaling_galaxies.csv` catalogue (`y, x, luminosity` — any consistent
  luminosity units; only ratios enter the model).
- Start with the default model -- it works very well for pretty much all groups!

__Wrap Up__

This script has shown how to model CCD imaging data of group-scale strong lenses with the three-tier
galaxy API — free main galaxies, bounded extra galaxies, and a scaling tier whose parameter count does
not grow with the number of galaxies.

Details of the **PyAutoLens** API and how lens modeling works were omitted for simplicity, but everything you need to
know is described throughout the main workspace documentation. You should check it out, but maybe you want to try and
model your own lens first!

The following locations of the workspace are good places to checkout next:

- `autolens_workspace/*/group/modeling`: A full description of the lens modeling API and how to customize your model-fits.
- `autolens_workspace/*/group/simulators`: A full description of the lens simulation API and how to customize your simulations.
- `autolens_workspace/*/group/data_preparation`: How to load and prepare your own imaging data for lens modeling.
- `autolens_workspace/guides/results`: How to load and analyze the results of your lens model fits, including tools for large samples.
- `autolens_workspace/guides`: A complete description of the API and information on lensing calculations and units.
- `autolens_workspace/*/group/features/group_halo`: THE group-regime tutorial — whether to add a group-scale
  dark-matter halo to this model is an explicit choice; learn to make it by fitting with and without one.
- `autolens_workspace/group/features`: A description of advanced features for lens modeling, for example pixelized source reconstructions, read this once you're confident with the basics!

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

start_here loads real full-resolution FITS data; SMALL_DATASETS would break
the committed data/mask shapes.

ENV: full_datasets
"""
