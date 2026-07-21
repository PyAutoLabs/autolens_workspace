"""
__Log Likelihood Function: Weak Lensing__

This script provides a step-by-step guide of the **PyAutoLens** likelihood function for fitting a lens mass
model to a weak gravitational lensing shear catalogue (a `WeakDataset`). It is the weak-lensing companion of
the guides for the other dataset types (e.g. `scripts/imaging/likelihood_function.py`), following the same
style and level of detail.

Every step below is what happens inside a single call of `al.AnalysisWeak.log_likelihood_function` — the
function a non-linear search calls tens of thousands of times in `scripts/weak/modeling.py`. By the end of
the script we will have computed the log likelihood "by hand" and verified it matches `al.FitWeak` and
`al.AnalysisWeak` exactly.

A weak-lensing likelihood is the simplest in **PyAutoLens**: there is no PSF convolution, no mask, no
over-sampling and no linear inversion. The data are a catalogue of `2N` numbers — two shear components per
background galaxy — and the likelihood is a pure Gaussian comparison of these against the model shear field
evaluated at the galaxy positions. This simplicity is what makes weak-lensing constraints so cheap to add to
a strong-lensing analysis.

__Contents__

- **Dataset:** Load the weak-lensing shear catalogue that is fitted (auto-simulating it if missing).
- **Lens Galaxy:** Define the lens galaxy mass model whose shear field is compared to the data.
- **Shear Field Evaluation:** Evaluate the tracer's shear at every catalogue position via the lensing Hessian.
- **Residuals:** Compare the model shear to the observed shear, galaxy by galaxy.
- **Chi Squared:** Sum the noise-normalized squared residuals over all 2N shear components.
- **Noise Normalization Term:** The Gaussian normalization, with its factor of 2 for the two components.
- **Calculate The Log Likelihood:** Combine the two terms into the log likelihood.
- **Fit:** Verify the manual calculation against `al.FitWeak`.
- **Analysis:** Verify it against `al.AnalysisWeak.log_likelihood_function`, as called in lens modeling.
- **Wrap Up:** Summary and pointers to the rest of the weak-lensing series.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al

"""
__Dataset__

We load the simulated `WeakDataset` produced by `scripts/weak/simulator.py`: 1500 background source-galaxy
positions in a 50"-200" annulus around a cluster-scale lens, each with a measured `(gamma_2, gamma_1)` shear
vector and a per-galaxy shape noise of 0.25 (a typical weak-lensing shape-noise value).

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
dataset_path = Path("dataset") / "weak" / "simple"

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/weak/simulator.py"],
        check=True,
    )

dataset = al.from_json(file_path=dataset_path / "dataset.json")

print(dataset.info)

"""
The three ingredients of the likelihood are all on the dataset:

 - `dataset.positions`: the `(N, 2)` grid of `(y, x)` arc-second coordinates of the background galaxies.
 - `dataset.shear_yx`: the `(N, 2)` observed shear components, stored as `(gamma_2, gamma_1)` per galaxy.
 - `dataset.noise_map`: the `(N,)` per-galaxy noise, shared by both shear components of that galaxy.
"""
print(f"n_galaxies : {dataset.n_galaxies}")
print(f"positions (first galaxy) : {np.asarray(dataset.positions)[0]}")
print(f"shear (first galaxy)     : {np.asarray(dataset.shear_yx)[0]}")
print(f"noise (first galaxy)     : {np.asarray(dataset.noise_map)[0]}")

"""
__Lens Galaxy__

The model whose likelihood we are evaluating is a `Tracer` — the same object used by every other dataset
type. Only the mass profiles matter for weak lensing: background galaxies are pure probes of the shear
field, so no light profiles are needed anywhere.

We use mass parameters close to (but not exactly) the simulator's truth, so the residuals below are visibly
non-zero but the fit is good.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=25.0,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(redshift=1.0)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Shear Field Evaluation__

Step one of the likelihood: evaluate the model's shear at every catalogue position.

The shear is a second derivative of the lensing potential. **PyAutoLens** computes it by numerically
differentiating the tracer's deflection-angle field — the `LensCalc.shear_yx_2d_via_hessian_from` method
evaluates deflections on a small cross of points around each galaxy position and forms the Hessian, from
which the two shear components follow.

Two things are worth noting:

 - This is the *same primitive* the simulator (`SimulatorShearYX`) uses to generate data, so a noise-free
   dataset fitted by its own truth tracer round-trips bit-exactly to zero residuals.

 - The result is the shear `gamma`, not the reduced shear `g = gamma / (1 - kappa)` that real surveys
   measure — adequate here and at large radii where `kappa` is small, and the planned real-data example
   in the weak-lensing series will introduce the distinction.
"""
from autogalaxy.operate.lens_calc import LensCalc

model_shear = LensCalc.from_tracer(tracer).shear_yx_2d_via_hessian_from(
    grid=dataset.positions
)

print(f"model shear (first galaxy) : {np.asarray(model_shear)[0]}")

"""
__Residuals__

Step two: subtract the model from the data. Both are `(N, 2)` arrays of `(gamma_2, gamma_1)` components, so
the residual map is simply their difference — no convolution, binning or masking intervenes.
"""
residual_map = np.asarray(dataset.shear_yx) - np.asarray(model_shear)

print(f"residuals (first galaxy) : {residual_map[0]}")

"""
Each residual is then divided by that galaxy's noise. The per-galaxy noise `sigma` applies to *both* shear
components (they share the same measurement process) but the two components are *independent* Gaussian
draws — this independence is why the likelihood below counts `2N` data points, not `N`.

The `[:, None]` broadcasts the `(N,)` noise map across both components of the `(N, 2)` residual map.
"""
noise_map = np.asarray(dataset.noise_map)

normalized_residual_map = residual_map / noise_map[:, None]

"""
__Chi Squared__

Step three: the chi-squared is the sum of squared normalized residuals over all `N x 2` components:

$\\chi^2 = \\sum_{i=1}^{N} \\sum_{k=1}^{2} \\left( \\frac{\\gamma^{\\rm data}_{i,k} - \\gamma^{\\rm model}_{i,k}}{\\sigma_i} \\right)^2$

For a well-fitting model whose residuals are pure shape noise, the expected chi-squared is approximately the
number of data points, `2N = 3000` — a quick sanity check worth internalising for any weak-lensing fit.
"""
chi_squared_map = normalized_residual_map**2.0

chi_squared = float(np.sum(chi_squared_map))

print(
    f"chi_squared : {chi_squared:.4f}  (expected ~{2 * dataset.n_galaxies} for a good fit)"
)

"""
__Noise Normalization Term__

The Gaussian likelihood also carries a model-independent normalization term:

$\\text{noise normalization} = \\sum_{i=1}^{N} \\sum_{k=1}^{2} \\ln \\left( 2 \\pi \\sigma_i^2 \\right) = 2 \\sum_{i=1}^{N} \\ln \\left( 2 \\pi \\sigma_i^2 \\right)$

The leading factor of 2 is the same `2N` counting as the chi-squared: each galaxy contributes two
independent measurements with the same `sigma`. Because it does not depend on the model, this term does not
influence which model a non-linear search prefers — but it is required for the log likelihood's absolute
value to be meaningful (e.g. when comparing to other dataset types in a combined fit).
"""
noise_normalization = float(2.0 * np.sum(np.log(2.0 * np.pi * noise_map**2.0)))

print(f"noise_normalization : {noise_normalization:.4f}")

"""
__Calculate The Log Likelihood__

The log likelihood combines the two terms in the standard Gaussian form:

$\\ln \\mathcal{L} = -\\frac{1}{2} \\left( \\chi^2 + \\text{noise normalization} \\right)$

That is the entire weak-lensing likelihood function — compare with the imaging guide, where PSF convolution
and masking sit between the model and this same final expression.
"""
log_likelihood = -0.5 * (chi_squared + noise_normalization)

print(f"log_likelihood (by hand) : {log_likelihood:.8f}")

"""
__Fit__

`al.FitWeak` packages the steps above (shear evaluation, residuals, chi-squared, normalization) into a
single object. Its log likelihood must match our manual calculation exactly.
"""
fit = al.FitWeak(dataset=dataset, tracer=tracer)

print(f"log_likelihood (FitWeak) : {fit.log_likelihood:.8f}")

assert fit.log_likelihood == log_likelihood

"""
__Analysis__

Finally, `al.AnalysisWeak` is the object handed to a non-linear search in `scripts/weak/modeling.py`. Its
`log_likelihood_function` builds the `Tracer` from a model instance and returns exactly the `FitWeak` log
likelihood — so the number below is what Nautilus receives at every sampled point of parameter space.
"""
import autofit as af

model = af.Collection(galaxies=af.Collection(lens=lens_galaxy, source=source_galaxy))

analysis = al.AnalysisWeak(dataset=dataset)

instance = model.instance_from_unit_vector([])

analysis_log_likelihood = analysis.log_likelihood_function(instance=instance)

print(f"log_likelihood (AnalysisWeak) : {analysis_log_likelihood:.8f}")

assert analysis_log_likelihood == log_likelihood

"""
__Wrap Up__

We have computed the weak-lensing log likelihood step by step and verified it against `al.FitWeak` and
`al.AnalysisWeak`:

 1. Evaluate the tracer's shear field at the catalogue positions (lensing Hessian).
 2. Residuals: data minus model, per galaxy and per component.
 3. Chi-squared: noise-normalized squared residuals summed over all `2N` components.
 4. Noise normalization: `2 sum(ln(2 pi sigma^2))`, with the factor 2 counting both components.
 5. Log likelihood: `-0.5 * (chi_squared + noise_normalization)`.

The rest of the weak-lensing series builds on this: `scripts/weak/modeling.py` samples this likelihood with
a non-linear search, `scripts/weak/fit.py` visualizes fits (including the tangential shear profile, the
standard observable this likelihood constrains), and upcoming examples apply it to a real cluster shear
catalogue and combine it with strong-lensing imaging in a joint fit.
"""
