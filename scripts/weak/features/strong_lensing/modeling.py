"""
Modeling: Combined Strong + Weak Lensing
========================================

This script fits the combined dataset of `simulator.py` — an `Imaging` dataset of strongly lensed arcs and a
`WeakDataset` of the surrounding shear field — with a **single lens mass model**, using PyAutoFit's
factor-graph API to sample the joint likelihood with one non-linear search.

__Why combine them?__

The two datasets constrain the same mass distribution in complementary regimes:

 - **Strong lensing** (the arcs) pins the Einstein radius, and the mass centre exquisitely — but only
   *inside* ~1 Einstein radius, and mass-model families that agree there can diverge immediately outside it.

 - **Weak lensing** (the shear catalogue) measures the mass profile and its ellipticity out to many Einstein
   radii — noisily per galaxy, but with statistical power in the ensemble, and precisely where the strong
   lensing has none.

Fitting them jointly forces one parametric mass model to satisfy both, the approach of hybrid-Lenstool's
joint strong+weak cluster reconstructions (Niemiec et al. 2020, who showed sequential fitting biases the
profile at 2-3 sigma where a joint fit stays within ~1 sigma) and of the stacked strong+weak analysis of the
Sloan Giant Arcs Survey group-to-cluster lenses (Oguri et al. 2012).

__Contents__

- **Dataset:** Load both datasets (auto-simulating if missing) and mask the imaging data.
- **Model:** One lens model whose priors are shared by both datasets' analyses.
- **Analysis List:** An `AnalysisImaging` and an `AnalysisWeak`, one per dataset.
- **Analysis Factor & Factor Graph:** Combine them so one search samples the joint likelihood.
- **Search & Model-Fit:** Nautilus over the shared parameter space.
- **Result:** The joint constraints, and how to read the strong/weak complementarity in them.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load both halves of the combined dataset, auto-simulating them if missing (the standard pattern of all
example scripts), and apply the standard 3.0" circular mask to the imaging data. Everything outside that
mask is the weak catalogue's territory.
"""
dataset_name = "strong_lensing"
dataset_path = Path("dataset") / "weak" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/weak/features/strong_lensing/simulator.py"],
        check=True,
    )

dataset_imaging = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset_imaging.shape_native,
    pixel_scales=dataset_imaging.pixel_scales,
    radius=3.0,
)

dataset_imaging = dataset_imaging.apply_mask(mask=mask)

dataset_weak = al.from_json(file_path=dataset_path / "dataset.json")

"""
__Model__

One model serves both datasets:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters] — the component both
   datasets constrain, through the arcs inside the mask and the shear outside it.

 - The source galaxy's light is a linear `SersicCore` [6 parameters] — only the imaging dataset sees this;
   the weak catalogue's galaxies are pure shear probes with no model components.

Crucially the model is composed **once**: passing the same model to both analysis factors below means they
share the same priors and therefore the same parameters — the definition of a joint fit.
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

bulge = af.Model(al.lp_linear.SersicCore)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__Analysis List__

One analysis object per dataset, exactly as each would be built in its own modeling script.

The imaging analysis runs in NumPy mode here (`use_jax=False`): the factor graph below evaluates its factors
in a plain Python loop because the weak-lensing analysis is a NumPy calculation, and mixing an eagerly-JAX
imaging likelihood into that loop gains nothing over NumPy without JIT compilation. On this small masked
dataset the NumPy likelihood is fast, and the weak likelihood is fractions of a millisecond.
"""
analysis_imaging = al.AnalysisImaging(dataset=dataset_imaging, use_jax=False)

analysis_weak = al.AnalysisWeak(dataset=dataset_weak)

"""
__Analysis Factor__

Each analysis is wrapped in an `AnalysisFactor` paired with the model. Because both factors receive the
*same* model object, the factor graph recognises every prior as shared — a single 11-dimensional parameter
space whose likelihood is the sum of the two factors.

(In the `multi` examples each factor gets a slightly different copy of the model, e.g. per-wavelength
ellipticities; here total sharing is exactly what "one mass distribution, two datasets" means.)
"""
analysis_factor_imaging = af.AnalysisFactor(
    prior_model=model, analysis=analysis_imaging
)

analysis_factor_weak = af.AnalysisFactor(prior_model=model, analysis=analysis_weak)

"""
__Factor Graph__

The factors combine into a `FactorGraphModel`, whose `global_prior_model` is the shared parameter space and
whose `log_likelihood_function` is the sum over factors — the quantity `fit.py` computed by hand.
"""
factor_graph = af.FactorGraphModel(analysis_factor_imaging, analysis_factor_weak)

"""
__Search & Model-Fit__

Nautilus samples the joint likelihood. The parameter space is simple (N=11, unimodal), so 100 live points
suffice; expect the fit to take some minutes on an ordinary CPU (the imaging likelihood dominates the cost —
adding the weak factor is essentially free, which is much of weak lensing's practical appeal).
"""
search = af.Nautilus(
    path_prefix=Path("weak") / "features",
    name="strong_lensing_joint",
    unique_tag=dataset_name,
    n_live=100,
    iterations_per_quick_update=10000,
)

print(
    """
    The joint strong+weak non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed - this could take some minutes!
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

The search returns one result per factor (imaging first, weak second), sharing a single posterior. The
mass parameters below are constrained by *both* datasets simultaneously.

To see the complementarity in play, compare this joint posterior to a run with the weak factor removed
(comment it out of the `FactorGraphModel` above): the Einstein radius barely changes — the arcs own it —
while the constraints on the mass's elliptical components tighten visibly when the shear at large radius is
included, because ellipticity is exactly what coherent tangential shear across the field measures.
"""
result = result_list[0]

print(result.info)

print(result.max_log_likelihood_instance)

"""
The per-factor maximum-likelihood fits visualize each dataset's view of the shared model.
"""
aplt.subplot_fit_imaging(
    fit=result_list[0].max_log_likelihood_fit,
    output_path=dataset_path,
    output_format="png",
)

aplt.subplot_fit_weak(
    fit=result_list[1].max_log_likelihood_fit,
    output_path=dataset_path,
    output_format="png",
)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

This example closed the loop the weak-lensing series builds towards: one mass model, constrained inside the
Einstein radius by strong lensing and outside it by weak shear, sampled as a single joint likelihood.

The same factor-graph pattern extends directly to the realistic versions of this analysis: cluster-scale
lenses with many cluster members (see `scripts/cluster`), real shear catalogues (the upcoming real-data
example in this series), and any other dataset combination (`scripts/multi`).
"""
