"""
Multi Galaxy: Modeling
======================

This script models an example strong lens on the 'multi galaxy' scale, where two (or more) galaxies of comparable
mass both contribute significantly to the lensing of a single background source.

The multi-galaxy regime keeps the **standard extended-source analysis workflow**: the data is CCD imaging, the
source is reconstructed at pixel level (parametric Sersic / MGE, or a pixelized mesh) via `AnalysisImaging`, and
the lens galaxies' light is modeled. What changes relative to `imaging/` is only the mass model: **one free light
and mass model per deflector**, because every deflector is co-dominant. This is a deliberate contrast with the
regimes above it on the ladder:

 - `group/`: still an extended-source `AnalysisImaging` fit, but the galaxies split into tiers (main / extra /
   scaling galaxies) and a group-scale dark-matter halo enters as an explicit modelling choice.
 - `cluster/`: the analysis itself changes — many sources at many redshifts are fitted as **point-source multiple
   image positions** via `AnalysisPoint` and a factor graph, with no lens light in the model at all.

All groups and clusters are multi-galaxy systems, but not vice versa: this package is the base rung, where the
only new concept is "more than one main lens galaxy".

This example uses a list-based model composition API, where main lens galaxies are built in a loop over centres
loaded from a JSON file and stored in the model as `lens_0`, `lens_1`, etc. The system's one overall external shear
is held separately, at the system centre, rather than on any single deflector. This API scales naturally to any
number of co-dominant deflectors.

__Contents__

- **Example:** The dataset fitted in this example and the model fitted to it.
- **Plotters:** Overview of plotting tools used for visualization.
- **Simulation:** Overview of how the simulated dataset was generated.
- **Data Preparation:** Data standards required for fitting with PyAutoLens.
- **Dataset:** Load and plot the multi-galaxy strong lens dataset.
- **Extra Galaxies Noise Scaling:** Scale the noise of nearby contaminating galaxies so they do not impact the fit.
- **Main Lens Galaxies:** In the multi-galaxy regime every deflector is a main lens galaxy.
- **Mask:** Standard set up of the mask that is fitted.
- **Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
- **Redshifts:** All deflectors are at the same redshift, so ray tracing is single-plane.
- **Model:** Compose the lens model fitted to the data.
- **External Shear:** Why the shear is held at the system centre, not on one deflector.
- **Model Composition:** Compose the lens model using the Model and Collection API.
- **Coordinates:** Coordinate system assumptions for the model-fit.
- **Improved Lens Model:** Why the model uses MGE and linear light profiles rather than plain Sersics.
- **Linear Light Profiles:** Solving for each component's intensity via a linear inversion.
- **Concise API:** The utility function which hides the long MGE composition API.
- **Search:** Configure the non-linear search used to fit the model.
- **Unique Identifier:** How the output folder's unique identifier is generated.
- **Iterations Per Update:** How often the search writes the maximum likelihood model to hard-disk.
- **Live Visual Update:** Push the quick-update image to a live display surface.
- **Analysis:** Create the Analysis object that defines how the model is fitted to the data.
- **JAX:** JAX acceleration for fast GPU/CPU model-fitting.
- **VRAM Use:** When running with JAX on a GPU, the analysis must fit within the GPU's available VRAM.
- **Run Times:** Profiling the expected run time of the model-fit.
- **Output Folder Layout:** Description of the structure of the `output` folder where results are written.
- **Result:** Overview of the results of the model-fit.
- **Loading From Output Folder:** Loading results back from hard-disk as Python objects.
- **Source Science (Magnification, Flux and More):** Computing the source's magnification, flux and size.
- **Features:** Extensions of the multi-galaxy model (extra galaxies, scaling galaxies, pixelized sources).
- **HowToLens:** The lecture series which teaches how lens modeling actually works.
- **Modeling Customization:** Alternative searches and how to customize the model-fit.

__Example__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - The lens is a pair of co-dominant galaxies, each with its own MGE light model and `Isothermal` mass model.
 - The system has a single overall `ExternalShear`, held at the system centre rather than on either galaxy.
 - The source galaxy's light is an MGE.

The pair configuration is modeled on **SDSS J1011+0143** (Shu et al. 2016, ApJ 820, 43, arXiv:1602.02927), a
merging pair of early-type galaxies at z=0.331 (projected separation ~4.2 kpc) lensing a z=2.701 Lyman-alpha
emitter into a wide Einstein cross / arc. The published model of that system is exactly the model composed here:
two isothermal mass profiles plus shear, with an extended source.

This lens model is the natural starting point for a multi-galaxy lens, and is computationally fast to fit, so it
acts as a good starting point for new users.

__Start Here vs Modeling__

The folder's `start_here.py` fits this same lens with `af.MultiStartProdigy`, a multi-start gradient optimizer. It
is much faster, but it is a maximum a posteriori optimizer: it returns a single best-fit model with no error bars
at all.

This script instead uses the nested sampling algorithm `Nautilus`, which returns the **full posterior** — every
parameter's probability density, its errors, and the correlations between parameters. That matters especially for
a multi-galaxy lens, because the two deflectors' masses are *correlated*: the data constrains the total deflection
well and the split between the two galaxies less well. A single best-fit model hides that degeneracy; a posterior
shows it to you directly. Use the fast optimizer to check a model, and this script when you need results you can
quote.

__Plotters__

To produce images of the data, plotting function objects are used, which are high-level wrappers of matplotlib
code which produce high quality visualization of strong lenses.

The plotting function API is described in the script `autolens_workspace/*/guides/plot`.

__Simulation__

This script fits a simulated `Imaging` dataset produced by `autolens_workspace/*/multi_galaxy/simulator.py`. If
the dataset is not found on disk it is simulated automatically before the fit.

__Data Preparation__

The `Imaging` dataset fitted in this example conforms to a number of standards that make it suitable to be fitted
in **PyAutoLens**. If you are intending to fit your own strong lens data, ensure it conforms to these standards,
which are described in `autolens_workspace/*/imaging/data_preparation/start_here.ipynb`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the multi-galaxy strong lens dataset `simple`, which is the dataset we will use to perform lens modeling.

This is loaded via .fits files, which is a data format used by astronomers to store images.

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are
using is 0.05" / pixel (Hubble Space Telescope ACS resolution, matching the data the real SDSS J1011+0143 pair
was modeled with).
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
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
Use an `aplt.subplot_imaging_dataset` to plot the data, including:

 - `data`: The image of the strong lens.
 - `noise_map`: The noise-map of the image, which quantifies the noise in every pixel as their RMS values.
 - `psf`: The point spread function of the image, which describes the blurring of the image by the telescope optics.
 - `signal_to_noise_map`: Quantifies the signal-to-noise in every pixel.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Extra Galaxies Noise Scaling__

Before masking, we must deal with any extra galaxies in the data: nearby galaxies (or foreground stars, or
data-reduction artefacts) whose emission is not associated with the strong lens but blends into the field. If
their light is left in the data it will contaminate the model-fit and bias the inferred lens model. It is too
easy to skip straight to modeling without checking for these, so we make this step an explicit part of the
workflow.

For a multi-galaxy lens this step comes with a judgement that the single-galaxy case does not force on you. The
image contains several galaxies, and you must decide which are which:

 - A galaxy that contributes significantly to the lensing of the source is a **main lens galaxy**. It belongs in
   the model, with its own free light and mass profiles, and its centre goes in `main_lens_centres.json`.
 - A galaxy whose light merely blends into the field is an **extra galaxy** — a contaminant. It belongs here,
   removed from the analysis by noise scaling, and never enters the model at all.

Getting this wrong in either direction is costly: model a contaminant and you waste parameters and invite a local
maximum; mask a co-dominant deflector and your mass model is simply wrong. The test is the lensing contribution,
not brightness or proximity.

To prevent extra galaxies from impacting the model-fit, we do not mask them entirely from the fit, which
would be analogous to making the circular mask smaller or using a more refined mask. When pixels are masked and
removed entirely from the fit, their coordinates are not used when performing ray-tracing and the light of the
lens and source galaxies in these pixels not evaluated.

Instead, the pixels are kept in the fit, but their data values are scaled to zero and their noise-map values
are increased to very large values. This means that during the model-fit, these pixels contribute negligibly to
the likelihood of the fit, and therefore do not impact the lens model.

This approach is used because for certain types of modeling approaches, like a pixelized source reconstruction,
masking regions of the image in a way that removes their image pixels entirely from the fit can produce
discontinuities in the pixelization. This can lead to unexpected systematics and unsatisfactory results.

In this case, applying the mask in a way where the image pixels are not removed from the fit, but their data and
noise-map values are scaled such that they contribute negligibly to the fit, is a better approach.

The `simple` dataset includes a faint extra galaxy, and a `mask_extra_galaxies.fits` covering it is shipped with
the dataset (created by the simulator). If you are modeling your own data with an extra galaxy, you must either:

 - Create a `mask_extra_galaxies.fits` for it using the data-preparation tools (the GUI
   `autolens_workspace/*/imaging/data_preparation/gui/mask_extra_galaxies.py`, the GUI at the end of
   `multi_galaxy/start_here.py`, or the manual
   `autolens_workspace/*/imaging/data_preparation/examples/optional/mask_extra_galaxies.py`), then load it
   as below; or
 - Shrink the circular mask below so the extra galaxy lies outside it and is removed from the fit entirely.

We then plot the dataset, where the extra galaxy's pixels now have their data scaled to zero and noise-map
values increased, making their signal-to-noise effectively zero.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,  # `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Main Lens Galaxies__

In the group and cluster packages, galaxies split into tiers — main galaxies modeled freely, extra galaxies with
restricted freedom, scaling galaxies tied to a luminosity relation — because there are too many deflectors to free
them all. The multi-galaxy regime needs no tiers: there are only a handful of deflectors and each contributes
comparably to the lensing, so **every galaxy is a main lens galaxy** with its own free light and mass model.

We load the centres of the main lens galaxies from a `.json` file in the dataset folder. These centres initialize
the centre priors of each galaxy's light and mass profiles. For your own data, the `group/start_here.ipynb`
example provides a GUI for clicking the galaxy centres on the image.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the lens model to the data.

We create a 3.0 arcsecond circular mask and apply it to the `Imaging` object that the lens model fits.

For a multi-galaxy lens, sizing this mask needs more care than at galaxy scale. The Einstein radius that matters is
that of the *combined* mass distribution (~1.8" here), not either galaxy's individually, and the lensed arcs wrap
around the pair as a whole. A mask sized by eye from one galaxy's light will clip the ring.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
If we plot the masked data, the mask removes the exterior regions of the image where there is no emission from the
lens and lensed source galaxies.

The mask used to fit the data can be customized, as described in
the script `autolens_workspace/*/guides/modeling/customize`
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

For lensing calculations, the high magnification regions of a lensed source galaxy require especially high levels
of over sampling to ensure the lensed images are evaluated accurately.

For a new user, the details of over-sampling are not important, therefore just be aware that calculations either:

 (i) use adaptive over sampling for the foreground lenses' light, which ensures high accuracy in their centres.
 (ii) use cored light profiles for the background source galaxy, where the core ensures low levels of over-sampling
 produce numerically accurate but fast to compute results.

The one multi-galaxy specific point is that the adaptive scheme is centred on **every** main lens galaxy rather
than on a single centre. Each deflector has its own steep central light profile needing accurate evaluation, and
`centre_list` takes as many centres as you give it — so this scales with no change as you add deflectors.

Once you are more experienced, you should read up on over-sampling in more detail via
the `autolens_workspace/*/guides/advanced/over_sampling.ipynb` notebook.
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
__Redshifts__

Both deflectors are at the same redshift (as in a physically associated pair like SDSS J1011+0143), so ray
tracing is single-plane and the two galaxies' deflection fields simply add.

If your two deflectors are at *different* redshifts, the system is a compound, multi-plane lens (e.g. the
"Einstein zigzag" J1721+8842) — PyAutoLens supports this natively by simply assigning each galaxy its redshift,
and the cluster package documents multi-plane ray tracing in detail.

__Model__

We compose the lens model, with one entry per co-dominant deflector:

 - Each main lens galaxy's light is an MGE (Multi Gaussian Expansion), which captures complex morphologies with
   few free parameters — important here, where the two galaxies' light blends together.
 - Each main lens galaxy's mass is an `Isothermal` (SIE), the standard galaxy-scale profile. Note these are
   **untruncated** profiles: truncation of a galaxy's mass encodes tidal stripping by a host halo's potential,
   and the multi-galaxy regime by definition has no host halo. Truncated (dPIE) profiles enter with the
   group regime's Lenstool-style workflow (`group/features/group_halo`) and are the default at cluster
   scale.
 - The system's single `ExternalShear` is held in its own model entry at the system centre (0.0", 0.0"), rather
   than being attached to one of the deflectors.
 - The source galaxy's light is an MGE.

An upgrade path used by published multi-deflector analyses is to swap each `Isothermal` for a `PowerLaw` (EPL),
freeing each galaxy's density slope — a one-line change per galaxy (`al.mp.PowerLaw`), best made after this
simpler model has converged.

The dimensionality of parameter space is what grows in this regime. Each additional co-dominant deflector adds its
own light and mass parameters, so a pair is a substantially larger fit than a single-galaxy lens. This is exactly
why the tiered group and cluster APIs exist: beyond a handful of deflectors, freeing them all becomes intractable
and members must be tied to a scaling relation instead.

__Model Composition__

The API below for composing a lens model uses the `Model` and `Collection` objects, which are imported from
**PyAutoLens**'s parent project **PyAutoFit**.

The API is fairly self explanatory and is straight forward to extend, for example adding more light profiles
to the lens galaxies and source or using a different mass profile.

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Coordinates__

At galaxy scale, the model fitting default settings assume the lens galaxy centre is near (0.0", 0.0"). A
multi-galaxy lens has no single such centre, so this example takes a different approach: the centre of every
deflector is read from `main_lens_centres.json` and used to set that galaxy's light and mass centre priors
explicitly, in the loop below.

This means the relevant data-preparation question is not "is my lens centred at the origin?" but "do I have
accurate centres for every deflector?". If you do not:

 - Use the centre-input GUI shown in `group/start_here.ipynb` to click them off the image.
 - Or manually override the lens model priors (`autolens_workspace/*/guides/modeling/customize`).

The centres are used as *priors*, not fixed values for the light profiles, so modest errors are recovered by the
fit — but a centre placed on the wrong galaxy will not be.

__Improved Lens Model__

A model could use plain Sérsic light profiles for the lens and source galaxies. This makes the model API concise,
readable, and easy to follow.

However, single Sérsic profiles perform poorly for most strong lenses. Symmetric profiles (e.g. elliptical
Sérsics) typically leave significant residuals because they cannot capture the irregular and asymmetric morphology
of real galaxies (e.g. isophotal twists, radially varying ellipticity).

This is worse in the multi-galaxy regime than at galaxy scale, and for a specific reason: the deflectors are close
together, so their light *blends*. Residuals left by an inadequate light model for one galaxy sit right on top of
the other galaxy's light and the lensed arcs, where they can be absorbed into the mass model. An interacting pair
like SDSS J1011+0143 is also genuinely disturbed, which is precisely what a symmetric profile cannot represent.

This example therefore uses a lens model that combines two features, described in detail elsewhere (but a brief
overview is provided below):

- **Linear light profiles** (see ``autolens_workspace/*/imaging/features/linear_light_profiles``)
- **Multi-Gaussian Expansion (MGE) light profiles** (see ``autolens_workspace/*/imaging/features/multi_gaussian_expansion``)

These features avoid wasted effort trying to fit Sérsic profiles to complex data, which is likely to fail unless
the lens is extremely simple. This does mean the model composition is more complex and as a user it is a steeper
learning curve to understand the API, but it is worth it for the improved accuracy and speed of lens modeling.

__Multi-Gaussian Expansion (MGE)__

A Multi-Gaussian Expansion (MGE) decomposes a galaxy's light into ~50-100 Gaussians with varying ellipticities and
sizes. An MGE captures irregular features far more effectively than Sérsic profiles, leading to more accurate lens
models.

Remarkably, modeling with MGEs is also significantly faster than using Sérsics: they remain efficient in JAX (on
CPU or GPU), require fewer non-linear parameters despite their flexibility, and yield simpler parameter spaces that
sample in far fewer iterations.

That last property is what makes MGEs close to essential here. A multi-galaxy model already carries a larger,
more multi-modal parameter space than a single-galaxy fit; a light model that added dozens of free parameters per
deflector on top of that would be impractical.

__Linear Light Profiles__

The MGE model below uses a **linear light profile** for each bulge via the ``lp_linear`` API, instead of the
standard ``lp`` light profiles.

A linear light profile solves for the *intensity* of each component via a linear inversion, rather than treating it
as a free parameter. This reduces the dimensionality of the non-linear parameter space: a model with ~80 Gaussians
does not introduce ~80 additional free parameters.

Linear light profiles therefore improve speed and accuracy, and they are used by default in all modeling examples.

__Concise API__

The MGE model composition API is quite long and technical, so we simply load the MGE models for the lens and source
galaxies below via a utility function `mge_model_from` which hides the API to make the code readable. We then use
the PyAutoLens Model API to compose the overall lens model.

The full MGE composition API is given in the `imaging/features/multi_gaussian_expansion` package.

__List-Based Model Composition__

Each main lens galaxy is created in a loop over the main lens galaxy centres and stored as `lens_0`, `lens_1`,
etc. — the same list-based API the group package uses, so moving up the ladder later requires no re-learning.

__External Shear__

Note that no lens galaxy is given an `ExternalShear`. The shear describes the tidal field of everything *outside*
the system being modeled, so it is a property of the system as a whole, not of an individual galaxy. The galaxy-scale
examples attach it to their single lens galaxy, but a multi-galaxy lens has no single galaxy to attach it to, and
choosing one arbitrarily would misrepresent what it is.

We therefore give it its own model entry, `shear_galaxy`, at the system centre (0.0", 0.0"). `ExternalShear` takes no
`centre` argument because it is a uniform field defined about the coordinate origin, which for this dataset is the
centre of the lens pair.

This is a presentational choice, not a physical one: because the tracer sums every deflection field, a shear in its
own galaxy is numerically identical to the same shear attached to a deflector. What it buys you is a model whose
`info` and posterior label the shear as a property of the system, so you are never tempted to read it as a
measurement of `lens_0`.

Note also that giving a shear to *every* deflector would be a redundant parameterization: the shears would be
degenerate with one another and the fit would wander along that degeneracy.
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
The `info` attribute shows the model in a readable format. Note how each lens galaxy is listed as `lens_0`,
`lens_1`, etc., each with its own free mass model — the signature of the multi-galaxy regime — and how only
`lens_0` carries a `shear`.

The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The `info_whitespace_length` parameter in the file `config/general.yaml` in the [output] section can be changed to
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to
appear in a notebook).
"""
print(model.info)

"""
__Search__

The lens model is fitted to the data using a non-linear search.

This example uses the nested sampling algorithm Nautilus
(https://nautilus-sampler.readthedocs.io/en/latest/), which extensive testing has revealed gives the most accurate
and efficient modeling results for posterior estimation.

Nautilus has one main setting that trades-off accuracy and computational run-time, the number of `live_points`.
A higher number of live points gives a more accurate result, but increases the run-time. A lower value gives
less reliable lens modeling (e.g. the fit may infer a local maxima), but is faster.

The suitable value depends on the model complexity, whereby models with more parameters require more live points.
Because this model carries a full light and mass model for *every* deflector, it has more free parameters than the
galaxy-scale examples, which use 100 live points. We therefore use 200 here — a multi-modal parameter space with
several co-dominant deflectors is exactly the situation in which too few live points settles into a local maximum,
typically one where the two galaxies' Einstein radii have been mis-apportioned.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is
generated based on the model, search and dataset that are used in the fit.

An identical combination of model and search generates the same identifier, meaning that rerunning the script will
use the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique
identifier will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different
datasets with the same model and search, results are output to a different folder. We achieve this below by passing
the `dataset_name` to the search's `unique_tag`.

__Iterations Per Update__

Every `iterations_per_quick_update`, the non-linear search outputs the maximum likelihood model and its best fit
image to the Jupyter Notebook display and to hard-disk.

This process takes around ~10 seconds, so we don't want it to happen too often so as to slow down the overall
fit, but we also want it to happen frequently enough that we can track the progress.

The value of 10000 below means this output happens every few minutes on GPU and every ~10 minutes on CPU, a good
balance. Note that the unit here is a Nautilus likelihood evaluation, unlike `start_here.py` where the search is a
gradient optimizer and the unit is a gradient step — which is why the two scripts use very different values.

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
search = af.Nautilus(
    path_prefix=Path("multi_galaxy"),  # The path where results and output are stored.
    name="modeling",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=200,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=10000,  # Every N iterations the max likelihood model is visualized and output to hard-disk.
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

We next create an `AnalysisImaging` object, which can be given many inputs customizing how the lens model is
fitted to the data (in this example they are omitted for simplicity).

Internally, this object defines the `log_likelihood_function` used by the non-linear search to fit the model to
the `Imaging` dataset.

This is the point worth pausing on for the multi-galaxy regime: `AnalysisImaging` is the **extended-source,
pixel-level analysis** — the exact same object used for single-galaxy (`imaging/`) and group-scale fits. The
multi-galaxy regime changes the mass model, not the analysis. Only at cluster scale does the analysis itself
switch, to `AnalysisPoint` fits of multiple-image positions.

It is not vital that you as a user understand the details of how the `log_likelihood_function` fits a lens model to
data, but interested readers can find a step-by-step guide of the likelihood function at
`autolens_workspace/*/multi_galaxy/likelihood_function.py`, which walks through the multi-deflector case
specifically — including the step where the two galaxies' deflection fields are summed.

__JAX__

Analysis uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU support, your fits
will run much faster. If only a CPU is available, JAX will still provide a speed up via multithreading.

The multi-galaxy deflection sum vectorises cleanly — each deflector contributes another deflection field which is
simply added — so a two-galaxy likelihood costs little more per evaluation than a single-galaxy one. The extra cost
of this regime is in the *dimensionality* of parameter space (more iterations to converge), not the arithmetic per
iteration.

If you don't have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much
faster.

Force NumPy with `use_jax=False` (or `PYAUTO_DISABLE_JAX=1`) when debugging — NumPy stack traces are easier to read
than JAX traces.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
__VRAM Use__

When running with JAX on a GPU, the analysis must fit within the GPU's available VRAM. If insufficient VRAM is
available, the analysis will fail with an out-of-memory error, typically during JIT compilation or the first
likelihood call.

Two factors dictate the VRAM usage of an analysis:

- The number of arrays and other data structures JAX must store in VRAM to fit the model
  to the data in the likelihood function. This is dictated by the model complexity and dataset size.

- The `batch_size` sets how many likelihood evaluations are performed simultaneously.
  Increasing the batch size increases VRAM usage but can reduce overall run time,
  while decreasing it lowers VRAM usage at the cost of slower execution.

Before running an analysis, users should check that the estimated VRAM usage for the
chosen batch size is comfortably below their GPU's total VRAM.

The method below prints the VRAM usage estimate for the analysis and model with the specified batch size. It takes
about 20-30 seconds to run so you may want to comment it out once you are familiar with your GPU's VRAM limits.

Note that this dataset is higher resolution than the galaxy-scale examples (0.05"/pixel over a 3.0" mask), and each
deflector adds arrays of its own, so VRAM use here is higher than for a single-galaxy MGE fit. If you hit an
out-of-memory error, reduce `n_batch` on the search above before reaching for a smaller model.
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Times__

Lens modeling can be a computationally expensive process. When fitting complex models to high resolution datasets
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single `instance` of the lens model to be fitted to
   the dataset such that a log likelihood is returned.

 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex lens
   models require more iterations to converge to a solution.

For a multi-galaxy lens it is the second factor that dominates. The likelihood evaluation time is barely different
from a single-galaxy fit, because the deflection sum vectorises, but with two MGE light models, two free mass
profiles, a shear and an MGE source there are substantially more free parameters, so Nautilus needs considerably
more iterations to converge.

Expect ~10-20 minutes on a GPU and an hour or more on CPU. Each additional co-dominant deflector adds its own light
and mass parameters and pushes this up further, which is why the tiered group/cluster APIs exist for systems with
many galaxies.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs the
Nautilus non-linear search in order to find which models fit the data with the highest likelihood.

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may
produce an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
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
__Output Folder Layout__

Now the fit is running you should checkout the `autolens_workspace/output` folder. This is where results are
written to hard-disk in human-readable formats — `.json`, `.csv`, `.fits`, `.png` and plain text.

As the fit progresses, results are written on the fly using the highest likelihood model found by the
non-linear search so far. This means you can inspect the model-fit as it runs, without waiting for the
non-linear search to terminate.

Each completed fit lives at a path like::

    output/multi_galaxy/<dataset_name>/modeling/<unique_hash>/
        files/                         <- JSON + CSV: loadable Python objects
            tracer.json                <- max log likelihood Tracer
            model.json                 <- fitted af.Collection model
            samples.csv                <- full Nautilus samples
            samples_summary.json       <- max log likelihood parameter values + errors
            samples_info.json          <- metadata about the samples
            search.json                <- non-linear search configuration
            settings.json              <- search settings
            cosmology.json             <- cosmology used for the fit
            covariance.csv             <- parameter covariance matrix
        image/                         <- FITS + PNG: imaging products
            dataset.fits               <- data, noise-map and PSF
            fit.fits                   <- model image, residuals, chi-squared map
            tracer.fits                <- tracer image-plane images per galaxy
            source_plane_images.fits   <- source plane reconstructions
            model_galaxy_images.fits   <- per-galaxy model images
            galaxy_images.fits         <- per-galaxy images
            dataset.png, fit.png, tracer.png   <- visualisations
        model.info                     <- human-readable model summary
        model.results                  <- human-readable fit summary
        search.summary                 <- search run summary
        search_internal/               <- internal files used to resume / visualise the search
        metadata                       <- run metadata

The per-galaxy products are more useful here than at galaxy scale: `model_galaxy_images.fits` and
`galaxy_images.fits` contain one image per galaxy, so you can inspect each deflector's fitted light separately even
though their light blends together in the data.

The `<unique_hash>` is a 32-character identifier derived from the model, search and dataset, so re-running the
same configuration resumes from the existing fit automatically.

__Result__

The search returns a result object, whose `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could be changed to make
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
The `Result` object also contains:

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` and `FitImaging` objects.

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""
print(result.max_log_likelihood_instance)

"""
For a multi-galaxy lens the tracer subplot is worth a close look: the critical curve is that of the *combined* mass
distribution, so it wraps around the pair as a whole rather than encircling either galaxy individually.
"""
aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
It also contains information on the posterior as estimated by the non-linear search (in this example `Nautilus`).

Below, we make a corner plot of the "Probability Density Function" of every parameter in the model-fit.

The plot is labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand
parameter `n`). These mappings are specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass of the first galaxy, the names `lens_0` and `mass` defined when making the `Model` above are used).

This is the plot to study for a multi-galaxy lens, and the reason this script uses Nautilus rather than the fast
optimizer in `start_here.py`. Look for correlation between the two galaxies' `einstein_radius` parameters: the data
constrains the *total* deflection well, so if the pair's separation is small relative to the Einstein radius the
split between them is much less well constrained, and the corner plot shows that as an elongated, tilted degeneracy.
A single best-fit model would report one point along that ridge and tell you nothing about its length.
"""
aplt.corner_anesthetic(samples=result.samples)

"""
__Loading From Output Folder__

Everything the `Result` object contains has also been written to hard-disk, inside the fit's output folder. Each
file loads back into a full Python object with a single line — much faster and simpler than re-running the fit.

For example, the maximum log likelihood `Tracer` is saved as a `.json` file and the tracer image-plane images as
a `.fits` file:
"""
from autolens import from_json

result_path = search.paths.output_path  # Points at the fit's unique output folder.

if (result_path / "files" / "tracer.json").exists():
    tracer = from_json(file_path=result_path / "files" / "tracer.json")

    tracer_fits = al.Array2D.from_fits(
        file_path=result_path / "image" / "tracer.fits", hdu=0, pixel_scales=0.05
    )

"""
The output folder also contains `model.json`, `samples.csv`, `dataset.fits`, `fit.fits` and more. A full walkthrough
of loading results from the output folder — covering both single-fit (`from_json`) and multi-fit (aggregator)
workflows — is given in:

  `autolens_workspace/*/guides/results/start_here.py`

__Source Science (Magnification, Flux and More)__

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or
galaxies).

Using the reconstructed source model, we can compute key quantities such as the magnification, total flux, and
intrinsic size of the source.

The example `autolens_workspace/*/multi_galaxy/source_science.py` gives a complete overview of how to calculate
these quantities behind two co-dominant deflectors, where the summed deflection field produces a larger and more
structured magnification than a single galaxy would.

If you want to study the source galaxy after modeling has reconstructed it, then check out that example.

__Features__

The examples in `autolens_workspace/*/multi_galaxy/features` extend this model:

 - **Extra galaxies**: more distant perturbers added with restricted freedom (fixed centres) — the tier below
   co-dominance.
 - **Scaling galaxies** (`features/scaling_relation`): a population of faint galaxies far from the lens whose
   Einstein radii are tied to the brightest deflector's by a Faber-Jackson relation, so the tier costs **zero** free
   parameters however many galaxies it holds. They use **untruncated** isothermal profiles (no host halo means no
   tidal truncation; the truncated dPIE variant belongs to the group/cluster Lenstool-style workflows).
 - **Pixelized sources**: swap the MGE source for a Delaunay / adaptive mesh reconstruction, exactly as at
   galaxy scale.

We also recommend you checkout the following galaxy-scale features, because they make lens modeling in general more
reliable and efficient — you benefit from them irrespective of how many deflectors your lens has:

- ``linear_light_profiles``: The model light profiles use linear algebra to solve for their intensity, reducing
  model complexity.
- ``multi_gaussian_expansion``: A galaxy's light is modeled as ~25-100 Gaussian basis functions.
- ``pixelization``: The source is reconstructed using an adaptive rectangular or Delaunay mesh.
- ``no_lens_light``: The foreground lenses' light is not present in the data and thus omitted from the model.

The files `autolens_workspace/*/guides/modeling/searches` and `autolens_workspace/*/guides/modeling/customize`
provide guides on how to customize many other aspects of the model-fit. Check them out to see if anything
sounds useful, but for most users you can get by without using these forms of customization!

__Data Preparation__

If you are looking to fit your own CCD imaging data of a multi-galaxy strong lens, checkout
the `autolens_workspace/*/imaging/data_preparation/start_here.ipynb` script for an overview of how data should be
prepared before being modeled. The one additional step this regime needs is a `main_lens_centres.json` giving the
centre of every co-dominant deflector — the centre-input GUI in `group/start_here.ipynb` writes it from mouse
clicks.

__HowToLens__

This script, and the features examples above, do not explain many details of how lens modeling is performed, for
example:

 - How does PyAutoLens perform ray-tracing and lensing calculations in order to fit a lens model?
 - How is a lens model fitted to data? What quantifies the goodness of fit (e.g. how is a log likelihood computed?).
 - How does Nautilus find the highest likelihood lens models? What exactly is a "non-linear search"?

You do not need to be able to answer these questions in order to fit lens models with PyAutoLens and do science.
However, having a deeper understanding of how it all works is both interesting and will benefit you as a scientist.

This deeper insight is offered by the **HowToLens** Jupyter notebook lectures, which live in their own repository:
https://github.com/PyAutoLabs/HowToLens.

I recommend that you check them out if you are interested in more details!

__Modeling Customization__

The folder `autolens_workspace/*/guides/modeling/searches` gives an overview of alternative non-linear searches,
other than Nautilus, that can be used to fit lens models.

They also provide details on how to customize the model-fit, for example the priors.

__Where To Go Next__

- `autolens_workspace/*/multi_galaxy/fit`: the anatomy of a multi-galaxy fit — residuals, chi-squared, likelihood,
  and each deflector's share of the summed deflection field.
- `autolens_workspace/*/multi_galaxy/likelihood_function`: a step-by-step guide to the likelihood function.
- `autolens_workspace/*/multi_galaxy/simulator_sample`: simulating many multi-galaxy lenses at once.
- `autolens_workspace/*/group`: the next rung of the ladder — tiered galaxies and the optional group halo.
- `autolens_workspace/*/cluster`: the top rung — point-source constraints, scaling relations, multi-plane.
- `autolens_workspace/guides/results`: analyzing the results of your fits.
"""
