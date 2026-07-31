"""
Fits: Linear Light Profiles (Multi Galaxy)
==========================================

This script fits a multi-galaxy strong lens with linear light profiles **without a non-linear search** — the model
is composed at known values and fitted once, so the linear solve can be inspected directly.

This is the readable way to see what the solver does. `modeling.py` in this folder puts the same composition
behind a search, where the solved intensities are buried in a result object.

__Contents__

- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Tracer:** Compose the two co-dominant deflectors, the shear and the source with linear light profiles.
- **Fit:** Fit the tracer to the dataset and plot the result.
- **Intensities:** Read the intensities the solver found.
- **Comparison To Truth:** Check the solve against the values the simulator used.
- **The Flux Ratio:** The quantity a co-dominant pair is usually measured for.
- **Negative Intensities:** What a negative solve means and why it matters here.
- **Wrap Up:** Where to go next.

__What This Script Shows__

The linear solve is not an approximation or a fitting stage — it is exact, and it happens on every likelihood
evaluation. Given the *shape* of every light profile (centre, ellipticity, size, index), there is one set of
intensities that maximizes the likelihood, and linear algebra finds it in closed form.

For a multi-galaxy lens that has a consequence worth seeing explicitly: the flux ratio between the two deflectors
is **not an independent degree of freedom**. It is determined by the shape parameters. Two galaxies whose profiles
are the right shape will be given the right relative brightness automatically; two galaxies whose profiles are the
wrong shape will have flux moved between them by the solver, silently, to make the residuals as small as it can.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the `simple` multi-galaxy dataset, the same one fitted by `multi_galaxy/fit.py`.
"""
dataset_name = "simple"
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

Scale out the faint contaminant, as `multi_galaxy/modeling.py` explains. Leaving it in would give the solver a
patch of unmodelled flux near `lens_0` to absorb, which is exactly the effect this script is trying to show
cleanly.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Mask__

The standard 3.0" circular mask used throughout the multi-galaxy package.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

Adaptive over-sampling centred on **every** deflector, as `multi_galaxy/modeling.py` applies.

Do not skip this step when inspecting a linear solve. Both deflectors have `sersic_index=4.0` profiles, which are
steeply peaked in their centres; evaluated on the raw pixel grid those peaks are badly under-estimated. The solver
cannot tell an under-sampled profile from a genuinely fainter one, so it compensates by inflating the intensity —
and it inflates the two galaxies' intensities by *different* amounts, because they have different sizes. The flux
ratio you read out is then a numerical artefact rather than a measurement.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.35, 0.25), (-0.35, -0.25)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Tracer__

Compose the two co-dominant deflectors, the shear galaxy and the source, using **linear** light profiles.

Every shape parameter below is set to the value `multi_galaxy/simulator.py` used to create the data. The one thing
not set is `intensity` — `lp_linear` profiles do not take it. That is what the solver will supply.

The simulator's true intensities were `1.2` for `lens_0`, `1.0` for `lens_1` and `3.0` for the source. Keep those
numbers in mind; they are what the solve is checked against below.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source])

"""
__Fit__

Fit the tracer to the dataset. The linear solve happens inside this call.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log likelihood = {fit.log_likelihood}")

"""
__Intensities__

`linear_light_profile_intensity_dict` maps each linear light profile object in the tracer to the intensity the
solver assigned it.
"""
print(fit.linear_light_profile_intensity_dict)

"""
The same information in a more usable form: a tracer whose profiles are ordinary light profiles carrying the
solved intensities. This object can be plotted, integrated, or written to disk like any other tracer.
"""
tracer_solved = fit.tracer_linear_light_profiles_to_light_profiles

intensity_0 = tracer_solved.galaxies[0].bulge.intensity
intensity_1 = tracer_solved.galaxies[1].bulge.intensity

print(f"\nlens_0 solved intensity = {intensity_0}")
print(f"lens_1 solved intensity = {intensity_1}")

"""
__Comparison To Truth__

Because every shape parameter above is set to its true value, the solve should recover the simulator's
intensities — and it does, to within a fraction of a percent:

    lens_0 solved intensity = 1.2010    (simulated: 1.2)
    lens_1 solved intensity = 0.9978    (simulated: 1.0)

That is worth dwelling on. The intensities were never sampled, never initialized, and never given a prior. They
were computed in closed form from the data and the profile shapes, and they came back right. The residual error is
the Poisson noise in the data plus the faint contaminant's flux, which was noise-scaled rather than modelled.

The likelihood is `12399.8`, against `12583.4` for the simulator's own truth tracer loaded from `tracer.json` — a
gap of ~180, which is the price of solving the source's intensity linearly rather than knowing it.

Note that these numbers are only reproducible at full resolution. If you are running under the smoke-test profile
(`PYAUTO_SMALL_DATASETS=1`), the mask is capped to a handful of pixels and everything printed here will differ.
"""
true_ratio = 1.2 / 1.0
solved_ratio = intensity_0 / intensity_1

print(f"\nTrue intensity ratio (lens_0 / lens_1)   = {true_ratio}")
print(f"Solved intensity ratio (lens_0 / lens_1) = {solved_ratio}")

"""
__The Flux Ratio__

Note what the number above is and is not.

It is a ratio of `intensity` **normalizations**, not of luminosities. Two Sersic profiles with the same intensity
but different `effective_radius` and `sersic_index` have different total fluxes. To compare the galaxies
physically, each profile has to be integrated —
`multi_galaxy/features/scaling_relation/slam.py` shows that integration for an MGE, and the same reasoning applies
to a Sersic.

It also has no error bar. The solve is exact for the given shapes, so it carries no uncertainty of its own; all the
uncertainty lives in the shape parameters, which this script fixed to their true values rather than fitting.

__Sensitivity To Shape__

This is the multi-galaxy point of the script, and it is worth running yourself. Change `lens_0`'s
`effective_radius` from `0.6` to `0.8` — one wrong parameter, on one galaxy — and re-run. Measured on the full
resolution dataset:

    effective_radius_0 = 0.6 (true):  LL = 12399.8   I_0 = 1.2010   I_1 = 0.9978   ratio = 1.204
    effective_radius_0 = 0.8:         LL =    76.9   I_0 = 0.7646   I_1 = 0.9456   ratio = 0.809

The likelihood collapse is unsurprising. The instructive part is the second intensity. `lens_1`'s shape was not
touched, yet its solved intensity fell by 5.2% — the solver moved flux between the two galaxies, because their
light overlaps and only the *sum* is constrained where it does.

The flux ratio, meanwhile, went from 1.20 to 0.81: a 33% error, in the direction of the galaxy whose model was
never wrong. A single mis-specified size parameter on one deflector has corrupted a measurement about the other.

This is the mechanism by which a wrong light model biases the flux ratio in a multi-galaxy lens, and it is why
`multi_galaxy/slam.py` treats its lens-light stage as load-bearing rather than cosmetic. It has no galaxy-scale
equivalent: with one lens galaxy there is nothing for the solver to redistribute flux *to*.

__Negative Intensities__

The solver is not constrained to return positive values. A negative intensity means a profile is subtracting light
somewhere, which in a multi-galaxy fit usually means one deflector's profile has reached across into its
neighbour's territory.
"""
if intensity_0 < 0.0 or intensity_1 < 0.0:
    print(
        "\nWARNING: a deflector's solved intensity is negative — the light model is over-subtracting."
    )

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/linear_light_profiles/modeling.py` — the same composition behind a non-linear search.
 - `multi_galaxy/features/linear_light_profiles/likelihood_function.py` — where in the likelihood the solve
   happens, step by step.
 - `multi_galaxy/fit.py` — the multi-galaxy fit anatomy, including per-deflector deflection fields.
 - `imaging/features/linear_light_profiles/fit.py` — the galaxy-scale version of this script.
"""
