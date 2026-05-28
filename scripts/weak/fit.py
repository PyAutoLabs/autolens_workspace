"""
Fit: Weak Lensing
=================

This script shows how to fit a strong-lens mass model to a weak gravitational lensing shear catalogue. Where
the `imaging` and `interferometer` workflows fit a 2D image of a lensed source, the weak-lensing workflow fits
a set of (gamma_2, gamma_1) shear measurements at the (y, x) positions of background source galaxies — a
``WeakDataset`` produced by the simulator script in `scripts/weak/simulator.py`.

A weak-lensing fit is conceptually simpler than its imaging counterpart: there is no PSF convolution, no
masking, no inversion / pixelization, and no source-galaxy light profile. The model is a `Tracer` whose mass
profiles induce a shear field, and the `FitWeak` class compares that model shear against the observed shear
to compute residuals, chi-squared and the log-likelihood.

__Contents__

- **Dataset:** Load the simulated `WeakDataset` from disk and visualise it.
- **Model:** Build a lens-model `Tracer` whose mass profiles produce the model shear field.
- **Fit:** Construct a `FitWeak` and inspect its derived quantities (residuals, chi-squared, log-likelihood).
- **Visualization:** Plot the fit as a 2x2 mosaic of data, model, overlay, and chi-squared map.
- **Notes:** What a "good" fit looks like and how this script relates to the upcoming modeling tutorial.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We load the simulated `WeakDataset` produced by `scripts/weak/simulator.py`: 200 background
source-galaxy positions in a 3.0" half-extent square, each with a measured `(gamma_2, gamma_1)` shear
vector and per-galaxy noise standard deviation 0.3. The shear field carries the signature of the
foreground lens's mass distribution.

If `dataset.json` does not yet exist, run `scripts/weak/simulator.py` first.
"""
dataset_path = Path("dataset") / "weak" / "simple"

dataset = al.from_json(file_path=dataset_path / "dataset.json")

print(dataset.info)

"""
Before fitting it is worth visualising the dataset. `aplt.subplot_weak_dataset` produces a 2x2 mosaic
showing the shear field as headless quiver segments, the per-galaxy noise map, the shear magnitude
`|gamma|` and the position angle `phi`. Tangential alignment around the lens centre at `(0, 0)` is the
characteristic visual signature of a strong-lens shear field.
"""
aplt.subplot_weak_dataset(
    dataset=dataset,
    output_path=dataset_path,
    output_format="png",
)

"""
__Model__

The model `Tracer` is built from the same primitives the simulator used: a foreground Isothermal lens
galaxy and a background source galaxy with no light profile (weak-lensing measurements are sensitive to
the lens mass, not the source's appearance). In a real workflow the mass parameters would be inferred by a
non-linear search (see the modeling tutorial in the next step of the weak-lensing series). Here we
hand-pick parameters close to the simulator's truth — `einstein_radius=1.6`, `axis_ratio=0.9`,
`angle=45.0`, `centre=(0.0, 0.0)` — so the fit shows what residuals consistent with shape noise look like.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(redshift=1.0)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Fit__

`FitWeak` evaluates the model shear field at the dataset's galaxy positions via the lens's projected
mass Hessian, then derives residuals, chi-squared and the log-likelihood under the assumption that each
shear component is independently Gaussian-distributed around the model with the per-galaxy noise.

Each background galaxy contributes **two** independent measurements (`gamma_1` and `gamma_2` carry the
same per-galaxy noise but are independent draws), so the total number of degrees of freedom is
`2 * n_galaxies`. For a well-fitting model with shape-noise-dominated residuals the expected chi-squared
is approximately equal to that number.
"""
fit = al.FitWeak(dataset=dataset, tracer=tracer)

print()
print("Fit Summary")
print("-----------")
print(f"n_galaxies        : {dataset.n_galaxies}")
print(f"degrees_of_freedom: {2 * dataset.n_galaxies}")
print(f"chi_squared       : {fit.chi_squared:.3f}")
print(f"noise_normalization: {fit.noise_normalization:.3f}")
print(f"log_likelihood    : {fit.log_likelihood:.3f}")

"""
__Visualization__

`aplt.subplot_fit_weak` produces the 2x2 mosaic that summarises a weak-lensing fit:

 - **Top-left:** the observed shear field, drawn in the same headless-quiver style as the dataset plot.
 - **Top-right:** the model shear field evaluated at the galaxy positions.
 - **Bottom-left:** data and model overlaid on a single axes — data in black, model in red. Deviations
   are visible where the two segments disagree in length or orientation.
 - **Bottom-right:** the per-galaxy chi-squared map (summed across the two shear components), colour-coded
   to highlight any galaxies driving large residuals — for example, those whose true ellipticity happens
   to be poorly aligned with the lens's induced shear.
"""
aplt.subplot_fit_weak(
    fit=fit,
    output_path=dataset_path,
    output_format="png",
)

"""
__Notes__

A "good" fit produces residuals consistent with the shape-noise floor — visually, the residual quivers
in the data-vs-model overlay should look short and randomly oriented, and the chi-squared map should be
fairly uniform with no clear spatial pattern. A systematic mismatch (e.g. residuals all pointing inward
around a region) usually indicates that the model's mass profile is the wrong shape, not just the wrong
amplitude.

The next step in the weak-lensing series, `scripts/weak/modeling.py`, replaces this hand-picked model
with an `AnalysisWeak` driven by a non-linear search: the same `FitWeak` machinery, called inside a
likelihood function, with priors on the lens mass parameters.
"""
