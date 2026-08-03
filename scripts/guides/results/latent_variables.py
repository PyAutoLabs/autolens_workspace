"""
Results: Latent Variables
=========================

PyAutoLens ships a curated catalogue of lensing-specific latent variables — quantities derived from the lens
model that aren't sampled directly but are computed from each posterior draw to give you Bayesian uncertainties
on the science quantities you actually care about. This tutorial shows the catalogue, how to toggle individual
latents via the workspace config, how to load latent results from a completed fit, and how to extend
``al.AnalysisImaging`` with your own derived quantities.

The conceptual underpinning — what a latent variable IS in the Bayesian sense, what its 1σ/3σ errors actually
mean (empirical posterior quantiles, NOT analytic Gaussian propagation), the trade-off between every-sample
and N-draws-from-PDF output modes — lives in the autofit_workspace foundational tutorial at
``../../../autofit_workspace/scripts/cookbooks/latent_variables.py``. Read that first if any of those terms
look unfamiliar; the tutorial here focuses on the lensing-specific catalogue and the workspace ergonomics.

__Contents__

 - Lensing Latents in PyAutoLens: The eight library-shipped latents and what each one means physically.
 - Toggling Latents: The workspace ``config/latent.yaml`` override.
 - Model Fit: Reuse the shared quick fit (``_quick_fit.py``) that produces real latent output.
 - Loading Latent Results: ``analysis.compute_latent_samples`` over a subset of PDF draws, and the two
   config surfaces (``latent.yaml`` / ``output.yaml``) that control which latents and how many draws.
 - Extending with a Custom Latent: Subclass ``al.AnalysisImaging`` to add lens-mass derived quantities.
 - Contributing Upstream: When your custom latent is general enough, promote it to the library.
"""

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autolens as al

"""
__Lensing Latents in PyAutoLens__

The library ships a flat registry of named latent functions at ``autolens.analysis.latent.LATENT_FUNCTIONS``,
backed by the toggle file ``autolens/config/latent.yaml``. Each entry maps a snake-case latent name to a Python
function that takes a fit, magzero, and ``xp`` and returns a scalar value. The catalogue splits into three
groups: raw-flux latents (no instrument inputs, default-on), microjansky variants (require ``magzero``,
default-off), and the dimensionless lensing latents (``magnification``, ``effective_einstein_radius``).

Raw-flux latents — sum the relevant model image in the fit's raw image units. Same units as
``dataset.data.array`` (typically e- s^-1 for HST, MJy/sr for JWST). See
``scripts/guides/units/flux.py`` for how to convert to microjanskies or AB magnitudes in post.

 - ``total_lens_flux`` — total integrated flux of the lens galaxy. Sum of
   ``fit.galaxy_image_dict[fit.tracer.galaxies[0]].array``. Returns NaN when the lens has no light profile.

 - ``total_lensed_source_flux`` — image-plane integrated flux of the source galaxy after lensing (with
   magnification baked in). Sum of ``fit.galaxy_image_dict[fit.tracer.galaxies[-1]].array``.

 - ``total_source_flux`` — the source's intrinsic flux in the source plane (before lensing). Computed from
   the source's light profile evaluated on the workspace's light-profile grid. Critically uses
   ``fit.tracer_linear_light_profiles_to_light_profiles`` rather than ``fit.tracer`` so MGE / linear light
   profiles report the inversion-solved intensities rather than zero.

Microjansky variants — same image sources as the raw-flux trio, but with the AB-mag → µJy conversion baked
in. Each one requires ``magzero`` on the analysis; if it's missing, the latent returns NaN and the library
emits a single warning per process per latent name (your search still completes — the cost is just an empty
column).

 - ``total_lens_flux_mujy`` — ``total_lens_flux`` in microjanskies. Useful for stellar-mass-light scaling and
   photometric inference.

 - ``total_lensed_source_flux_mujy`` — ``total_lensed_source_flux`` in microjanskies.

 - ``total_source_flux_mujy`` — ``total_source_flux`` in microjanskies.

Dimensionless lensing latents — no instrument inputs, no µJy variant.

 - ``magnification`` — dimensionless ratio of ``total_lensed_source_flux_mujy / total_source_flux_mujy``.
   This is the empirical flux-amplification factor implied by the lens model and source light profile. Source-
   plane errors propagate non-linearly here because the source brightness and the lens magnification both
   contribute multiplicatively to the lensed flux. The empirical posterior on ``magnification`` (samples
   transformed through the ratio) captures that non-linearity faithfully; analytic error propagation through
   the ratio would not.

 - ``effective_einstein_radius`` — the Einstein radius in arcseconds, computed via
   ``LensCalc.einstein_radius_jit_from`` which traces the zero-contour of the tangential eigenvalue of the
   deflection field. "Effective" here means the radius of the circle with the same enclosed area as the
   tangential critical curve — the critical curve isn't circular for non-spherical mass models, but its
   enclosed area is the right physical quantity for mass-within-Einstein-radius estimates.

The raw-flux latents default to ``true`` in the library yaml — they cost essentially nothing and produce a
universally useful column. The µJy variants and the two dimensionless latents default to ``false`` so existing
fits and instrument-naive workflows stay unchanged on upgrade.

__Toggling Latents__

The library defaults the three raw-flux latents to ``true`` and everything else to ``false`` for the
regression-safety reasons described above. To opt in to the µJy variants, dimensionless lensing latents, or to
disable a default-on raw-flux latent, edit your workspace's ``config/latent.yaml`` and set the keys you want.
This workspace ships such a file at ``autolens_workspace/config/latent.yaml`` with all eight lensing latents
enabled.

Workspace ``config/`` values shadow the library defaults — PyAutoFit's ``conf.instance`` searches the workspace
``config/`` directory first, so toggling a latent in your workspace yaml is enough to enable it without modifying
the library install. To disable a specific latent for a particular fit (e.g. you're profiling and don't want
to incur the latent computation cost on every search update), flip it to ``false`` in
``autolens_workspace/config/latent.yaml`` or override locally with ``conf.instance.push(...)``.

__Model Fit__

The loading and extending sections need a completed fit to read latents from. Rather than run a bespoke
fit here, we reuse the shared quick fit that the other results guides use: ``_quick_fit.py`` writes an
Isothermal-lens + MGE-source fit of the ``simple__no_lens_light`` dataset to ``output/results_folder/``.
It is idempotent — it returns immediately if those results already exist — so across the whole guide
suite the expensive non-linear search is paid once rather than repeated in every example.

We then load that fit's samples via the aggregator, exactly as ``start_here.py`` and
``aggregator/models.py`` do.
"""
import subprocess
import sys

subprocess.run(
    [sys.executable, "scripts/guides/results/_quick_fit.py"],
    check=True,
)

from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(directory=Path("output") / "results_folder")
samples = list(agg)[0].samples

"""
The samples carry the parameter posterior; the ``analysis`` carries the machinery that turns each
posterior draw into latent values. We rebuild the dataset and an ``al.AnalysisImaging`` with
``magzero=25.0`` so the three µJy latents populate with real values (without it they'd be NaN and the
library would log a single warning per latent name per process). The raw-flux trio, Einstein-radius and
magnification latents don't need ``magzero``.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/no_lens_light/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)
dataset = dataset.apply_mask(mask=mask)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=False, magzero=25.0)

"""
__Loading Latent Results__

``analysis.compute_latent_samples(samples)`` returns a ``Samples`` object whose API matches the parameter
``Samples`` — ``median_pdf``, ``max_log_likelihood``, ``values_at_sigma_1``, etc. — but reports on the induced
latent posterior.

__Controlling the Cost via Config__

Latents are computed by reconstructing a fit for every posterior sample, so the cost scales with the number
of samples — and the two dimensionless lensing latents (``magnification``, ``effective_einstein_radius``) add
a zero-contour critical-curve solve on top. Two workspace config files control this, and both are worth
knowing:

 - ``config/latent.yaml`` — controls *which* latents are computed (the eight toggles; this workspace enables
   all of them). Disable the ones you don't need to save their per-sample cost.

 - ``config/output.yaml`` — ``latent_draw_via_pdf`` / ``latent_draw_via_pdf_size`` control *how many* posterior
   draws the latents are computed over when a live search updates. Drawing a representative subset from the PDF
   gives faithful latent errors at a fraction of the every-sample cost.

Here we mirror that draw-from-PDF behaviour explicitly with ``samples.samples_drawn_randomly_via_pdf_from``,
computing the latents over 20 PDF draws so this guide runs quickly while still producing a real, representative
latent posterior. For a publication-quality result, compute over all samples (or a larger number of draws).
"""
latent_draws = samples.samples_drawn_randomly_via_pdf_from(total_draws=20)
latent_samples = analysis.compute_latent_samples(latent_draws)

median_instance = latent_samples.median_pdf()
print(f"Median PDF magnification: {median_instance.magnification}")
print(
    f"Median PDF effective_einstein_radius: {median_instance.effective_einstein_radius}"
)
print(f"Median PDF total_source_flux_mujy: {median_instance.total_source_flux_mujy}")

"""
The 1σ / 3σ intervals on these latents are *empirical quantiles of the induced posterior*. For magnification
specifically, this is the right thing — magnification is a strongly non-linear function of the mass parameters
and the source position, so the symmetric Gaussian propagation of parameter errors would underreport the tail
risk. The empirical posterior captures the full non-linearity faithfully. See the autofit foundational tutorial
for the full treatment.

The ``effective_einstein_radius`` latent can produce noisy 1D posteriors when the zero-contour solver converges
to slightly different critical-curve estimates across samples. If you see step-like artefacts in a corner plot,
that's the discrete contour-finding cadence interacting with the continuous parameter posterior — the median
and 1σ intervals remain reliable, but use caution when reading the tails.

__Extending with a Custom Latent__

The library catalogue is intentionally narrow. If you want a different derived quantity — the lens mass's
axis ratio, the source-plane Sérsic effective radius, the time-delay between multiply-imaged source pixels —
subclass ``al.LatentLens`` (override ``keys`` and ``variables``) and declare it on your analysis through the
``Latent`` class attribute. This mirrors exactly how you customise visualisation with a ``Visualizer`` subclass:
the latent catalogue is a first-class, swappable component of the analysis rather than a pair of methods you
monkey-patch.

The example below adds ``mass_axis_ratio`` (the lens-mass axis ratio derived from the ``Isothermal`` mass's
``ell_comps``). The same pattern works for any function of the lens model instance. Calling
``al.LatentLens.keys(analysis)`` / ``al.LatentLens.variables(...)`` from your overrides keeps the library
latents alongside your custom ones — the Euclid pipeline
(``euclid_strong_lens_modeling_pipeline/util.py``, ``LatentEuclid``) uses this exact composition pattern in
production to append its FWHM aperture-flux latents.
"""

import numpy as np


class LatentMassAxisRatio(al.LatentLens):
    """
    The library lensing latents plus a custom ``mass_axis_ratio`` — the axis
    ratio of the lens galaxy's Isothermal mass profile, derived from its
    ``ell_comps``. Demonstrates adding a user-defined latent without modifying
    the library: subclass ``al.LatentLens`` and compose its ``keys`` /
    ``variables`` static methods.
    """

    @staticmethod
    def keys(analysis):
        return list(al.LatentLens.keys(analysis)) + ["mass_axis_ratio"]

    @staticmethod
    def variables(analysis, parameters, model):
        library_values = al.LatentLens.variables(analysis, parameters, model)

        instance = model.instance_from_vector(vector=parameters)
        xp = analysis._xp
        try:
            ell_y, ell_x = instance.galaxies.lens.mass.ell_comps
            axis_ratio = (1.0 - np.sqrt(ell_y**2 + ell_x**2)) / (
                1.0 + np.sqrt(ell_y**2 + ell_x**2)
            )
        except AttributeError:
            axis_ratio = xp.nan

        return library_values + (axis_ratio,)


class AnalysisImagingWithMassAxisRatio(al.AnalysisImaging):
    """
    ``AnalysisImaging`` that swaps in the custom ``LatentMassAxisRatio`` catalogue
    via the ``Latent`` class attribute — the same one-line mechanism used to
    declare a custom ``Visualizer`` or ``Result``. No library code is modified.
    """

    Latent = LatentMassAxisRatio


"""
A fit that uses ``AnalysisImagingWithMassAxisRatio`` produces a ``latent.csv`` with one extra column
(``mass_axis_ratio``) on top of the eight library defaults (raw-flux + µJy + dimensionless lensing). We don't
run a second fit here — the pattern above is the full recipe.

__Contributing Upstream__

If your custom latent is general enough that other PyAutoLens users would benefit (a SLACS-style external-
convergence proxy, a critical-curve perimeter, an enclosed-mass-at-fixed-radius), please consider promoting it
to the library:

 1. Add the function to ``autolens/analysis/latent.py``, following the signature
    ``(fit, magzero, xp=np) -> scalar``. Use NaN as the fallback when the function can't apply (no lens / no
    source / singular mass model).
 2. Register it in the module-level ``LATENT_FUNCTIONS`` dict.
 3. Add an entry to ``autolens/config/latent.yaml`` defaulting it to ``false`` (the workspace yaml opts users in).
 4. Add a unit test under ``test_autolens/analysis/test_latent.py``.
 5. Open a PR.

Pipeline-specific latents that need non-standard kwargs (PSF-relative aperture fluxes, dataset-specific
photometric quantities, multi-band colour terms) belong in your pipeline's local Analysis subclass. The Euclid
pipeline (``euclid_strong_lens_modeling_pipeline/util.py``) demonstrates this — its FWHM aperture-flux latents
stay pipeline-local and the rest inherit from the library.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape. Results guides must produce real
(reduced) samples so downstream aggregator reads (data_fitting, queries,
...) are non-empty.

ENV: full_datasets real_search
"""
