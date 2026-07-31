"""
Modeling Features: No Lens Light (Multi Galaxy)
===============================================

CCD imaging data of a multi-galaxy strong lens may not have lens galaxy light emission present — for example if
the deflectors' light has already been subtracted from the image, or if they are undetected at the observed
wavelength.

This example illustrates how to fit a multi-galaxy lens model to data where **no** deflector's light is present.

__Contents__

- **Advantages:** What removing the lens light buys you, and the count of parameters it actually saves.
- **Disadvantages:** What a multi-galaxy lens loses that a galaxy-scale lens does not.
- **Model:** Compose the lens model fitted to the data.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Centres:** Where the deflector centres come from when there is no light to read them off.
- **Over Sampling:** Why the adaptive scheme is not needed for the deflectors here.
- **Model Composition:** Compose the lens model using the Model and Collection API.
- **Search:** Configure the non-linear search used to fit the model.
- **Analysis:** Create the Analysis object that defines how the model is fitted to the data.
- **Result:** Overview of the results of the model-fit.
- **Wrap Up:** Summary of the script and where to go next.

__Advantages__

Removing the lens light removes light parameters. For the model of `multi_galaxy/modeling.py` the arithmetic is:

 - **20** free parameters with lens light: each deflector contributes a 4-parameter MGE `bulge` (its centre and
   ellipticity; the Gaussian intensities are solved linearly) and a 3-parameter `Isothermal` mass with its centre
   fixed, so 2 x 7 = 14, plus 2 for the shear and 4 for the source MGE.
 - **12** free parameters if the lens light is removed *and* the mass centres stay fixed — 8 fewer.

That is the number usually quoted, and for a galaxy-scale lens it is the whole story. Read the next section before
believing it here.

__Disadvantages__

Two of them, and the second is specific to this regime.

**The lens light subtraction may not be clean.** The lens and source light are blended in the data (e.g. by PSF
convolution), so a subtraction performed before modeling may over-subtract source light in some regions and
under-subtract lens light in others, distorting the lensed emission that the mass model is then fitted to. Fitting
the lens and source simultaneously instead lets the model find the optimal deblending and propagates the errors
forward. This applies at every scale, and is worse when two galaxies' light overlaps.

**You no longer know where the deflectors are.** At galaxy scale this barely registers: there is one lens, it is
conventionally near (0.0", 0.0"), and its position is not seriously in question. A multi-galaxy lens has two or
more deflectors, neither at the origin, and with the light gone there is nothing in the image that marks them.
So the mass centres cannot simply be fixed the way `multi_galaxy/modeling.py` fixes them to the light. They have
to be freed, anchored on whatever external information you have:

 - **16** free parameters for the model composed below — the light's 8 parameters are gone, but 4 mass-centre
   parameters have come back.

The real saving is therefore 4 parameters, not 8, and it is not a clean win: 8 well-constrained light parameters
have been traded for 4 poorly-constrained mass ones. Those four are also the *worst* ones to be uncertain about,
because a deflector's centre is degenerate with its Einstein radius, and the two deflectors' Einstein radii are
already degenerate with each other (`multi_galaxy/modeling.py` — the data constrains the total deflection well and
the split between the galaxies less well). Freeing the centres feeds that degeneracy.

None of this is an argument against fitting data that has no lens light — if the light is not there, it is not
there. It is an argument for getting the centres from somewhere and saying where they came from.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light is omitted (and is not present in the simulated data).
 - Each co-dominant deflector's total mass distribution is an `Isothermal` with a free, anchored centre
   [5 parameters each].
 - The system has a single overall `ExternalShear` at the system centre [2 parameters].
 - The source galaxy's light is a Multi Gaussian Expansion [4 parameters].

__Start Here Notebook__

If any code in this script is unclear, refer to `multi_galaxy/modeling.py`, which fits the same pair of deflectors
with their light included, and `imaging/features/no_lens_light/modeling.py` for the galaxy-scale version of this
feature.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the multi-galaxy strong lens dataset `simple__no_lens_light` via .fits files.

This is the same lens as `multi_galaxy/simple` — same two deflectors, same Einstein radii, same shear, same
source — with the lens light removed and nothing else changed.
"""
dataset_name = "simple__no_lens_light"
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
        [sys.executable, "scripts/multi_galaxy/features/no_lens_light/simulator.py"],
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
Note what the image does *not* contain. There is no foreground emission at all — only the lensed arcs. Compare
this to the `simple` dataset plotted in `multi_galaxy/modeling.py`, where two bright, blended galaxies sit inside
the ring.

__Centres__

We load the centres of the deflectors from a `.json` file in the dataset folder, exactly as
`multi_galaxy/modeling.py` does. What changes is what they mean and how much you should trust them.

For `simple` these centres are read off the light, and the centre-clicking GUI in `multi_galaxy/start_here.py`
produces them directly. Here there is no light to click. For real data the equivalent information has to come from
outside this image:

 - **Another band.** The deflectors are often detected at redder wavelengths even when absent from the band being
   modeled. This is the usual and best answer.
 - **The subtraction itself.** If the light was subtracted before modeling, whatever model performed the
   subtraction knows where the galaxies were — carry those centres forward.
 - **A catalogue position**, with the caveat that its astrometry may not be aligned with this image's.

If you have none of these, you are not modeling a multi-galaxy lens so much as asking the arcs how many deflectors
there are and where. That is a legitimate but much harder problem, and it needs wider centre priors than the ones
set below and far more live points.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask__

We create a 3.0 arcsecond circular mask and apply it to the `Imaging` object that the lens model fits, the same
mask used throughout the multi-galaxy package.

The sizing logic is the one from `multi_galaxy/modeling.py`: the radius that matters is that of the *combined*
mass distribution (~1.8" here), not either galaxy's individually, because the arcs wrap around the pair as a
whole. Note that with no lens light you cannot fall back on sizing the mask by eye from the foreground galaxies —
the arcs are all you have.
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

Over sampling evaluates light profiles on a higher resolution grid than the image data to ensure the calculation
is accurate.

`multi_galaxy/modeling.py` centres an adaptive over-sampling scheme on **every** deflector, because each has a
steep central light profile needing accurate evaluation. Here no deflector has a light profile at all, so that step
is simply unnecessary — this is the one respect in which removing the lens light makes the numerics easier as well
as the model smaller.

The source galaxy uses a cored light profile (`SersicCore` in the simulation, an MGE in the model) which varies
gradually in its centre, so it does not require heavy over-sampling either.

__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Model Composition__

We compose the lens model with one entry per co-dominant deflector, in a loop over the centres, exactly as
`multi_galaxy/modeling.py` does. Two things differ.

**No `bulge`.** Each `lens_i` carries a mass profile only.

**The mass centre is free, not fixed.** `multi_galaxy/modeling.py` writes `mass.centre = (centre[0], centre[1])`,
fixing the centre to the light. With no light to fix it to, we instead put a `GaussianPrior` on each coordinate,
centred on the input value with `sigma=0.1"`. This says "I believe this galaxy is here to within about 0.1
arcseconds, and I am prepared to be wrong". Choose that width to reflect how good your external centre information
actually is — it is doing real work in this model, and quoting a result without saying what it was set to leaves
out a model assumption.

The external shear is held in its own `shear_galaxy` at the system centre (0.0", 0.0"), for the reasons given in
`multi_galaxy/modeling.py`: the shear is a property of the system as a whole, not of either deflector, and
attaching it to one of two co-dominant galaxies would misrepresent it.
"""
# Main Lens Galaxies (mass only):

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=centre[0], sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=centre[1], sigma=0.1)

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        mass=mass,
    )

# External Shear:

shear_galaxy = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

# Source (MGE light):

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
The `info` attribute shows the model in a readable format.

Confirm two things in it: that no `lens_i` has a `bulge`, and that each `mass.centre` appears as a free parameter
with the `GaussianPrior` set above rather than as a fixed value.
"""
print(model.info)

"""
__Search__

The lens model is fitted to the data using the nested sampling algorithm Nautilus
(https://nautilus-sampler.readthedocs.io/en/latest/).

`multi_galaxy/modeling.py` uses 200 live points, double the galaxy-scale examples, because a model with a full
light and mass model per deflector has a large and multi-modal parameter space. This model has 4 fewer parameters,
which argues for fewer live points — but they are traded for free mass centres, which argues for more, since a
centre that can wander is exactly what lets the sampler find a local maximum with the two Einstein radii
mis-apportioned.

We therefore keep 200. If you fix the centres instead (a defensible choice when your external astrometry is
good), 100 is sufficient.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features",
    name="no_lens_light",
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
__VRAM__

The `multi_galaxy/modeling.py` example explains how VRAM is used during GPU-based fitting and how to print the
estimated VRAM required by a model.

The method below prints the VRAM estimate for this analysis and model. It takes 20-30 seconds, so comment it out
once you are familiar with your GPU's limits.
"""
# analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Time__

Run times are discussed in full in `multi_galaxy/modeling.py`. A no-lens-light fit is faster per likelihood
evaluation than the equivalent lit fit, because the two MGE light models are no longer evaluated or convolved —
in this model that is 40 Gaussians' worth of image evaluation removed.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a
non-linear search to find which models fit the data with the highest likelihood.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, described in `multi_galaxy/modeling.py` and in full in
`autolens_workspace/*/guides/results`.

The check worth making here specifically is on the inferred mass centres. Compare each to the input value used to
set its prior:

 - If a centre has moved by much less than the prior's `sigma`, the arcs are locating that deflector themselves
   and your external information was not load-bearing.
 - If a centre has moved to the edge of the prior, the data wants it somewhere the prior will not let it go —
   widen the prior and refit before quoting anything.
 - If a centre has barely moved *and* its Einstein radius is poorly constrained, the fit has absorbed the
   uncertainty into the mass split rather than the position. This is the failure mode the __Disadvantages__
   section warns about.
"""
print(result.info)

print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

This script fitted a multi-galaxy lens with no deflector light, and the substance of it was not the API change
(deleting `bulge=` from a loop) but the consequence: with the light gone, the deflector centres become model
parameters informed by data outside this image.

Where to go next:

 - `multi_galaxy/modeling.py` — the same lens with its light, and the baseline every number here is quoted against.
 - `imaging/features/no_lens_light` — the galaxy-scale version of this feature, where the centre problem does not
   arise.
 - `multi_galaxy/features/no_lens_light/slam.py` — the SLaM pipeline for this case, which skips the lens-light
   stages entirely and is where a production no-lens-light fit actually lives.
 - `imaging/features/pixelization` — a pixelized source, which combines naturally with a mass-only model.
"""
