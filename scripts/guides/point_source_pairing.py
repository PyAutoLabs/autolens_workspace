"""
Guide: Point-Source Pairing, Over-Prediction and Under-Prediction
=================================================================

When fitting multiple-image positions — the bread and butter of group- and cluster-scale lens
modeling — the model tracer will not, in general, predict exactly the images you observed. A wrong
(or merely uncertain) mass model predicts *extra* images that were never detected, or fails to
produce an observed image at all. What the likelihood does in those two situations decides which
models a sampler rewards, and historically lensing codes have handled it with quiet conventions
rather than explicit choices.

This guide documents PyAutoLens's choices: the three image-plane pairing schemes, the
over/under-prediction policies, the solver settings that interact with them at cluster scale, the
source-plane vs image-plane chi-squared trade-off, and the second axis of the likelihood-option
matrix — whether the source-plane centre is a free model parameter or analytically solved
(``al.ps.PointSolved``). It is the reference the cluster examples (``scripts/cluster/``, including
the Lenstool walkthrough in ``scripts/cluster/lenstool/``) point at for likelihood choices.

__The two failure modes__

**Under-prediction** (n_model < n_observed): the model cannot produce an observed image. This is
always physically damning — you *saw* the image — so every scheme must penalize it hard. A
likelihood that quietly drops unmatched observed images actively rewards mass models that lens
less, and samplers will find and exploit that reward.

**Over-prediction** (n_model > n_observed): the model predicts images you did not detect. This is
sometimes damning (a bright predicted image where the data shows blank sky) and sometimes entirely
fine — most lens models predict a strongly *demagnified* central image that real observations
cannot detect. The observational convention (shared by Lenstool practice) is therefore: extra
images below the detection limit are tolerated; bright extra images count against the model.

__The three pairing schemes__

- ``FitPositionsImagePairAll`` (the model-fit default, in its solved-centre form
  ``FitPositionsImagePairAllSolved``): a mixture likelihood — each observed position marginalizes
  over every model position, and the 1/n_permutations normalization acts as an Occam factor that
  mildly penalizes extra images. Statistically the most principled, smooth in the pairings (and so
  gradient-friendly), and differentiable end to end; its penalties are implicit rather than
  tunable. The mixture is computed with a max-shifted log-sum-exp, so it stays finite even when
  the worst pairing is tens of sigma away.

- ``FitPositionsImagePairRepeat``: every observed position pairs to its *nearest* model position,
  repeats allowed. Under-prediction is penalized by construction (an unmatched observed image pays
  its distance to the nearest surviving image; if the solver returns no images at all, a large
  finite floor applies). Over-prediction is governed by the ``unmatched_model_policy`` described
  below. See the benchmark evidence below for why this is no longer the default.

- ``FitPositionsImagePair``: Hungarian (linear-sum-assignment) pairing without repeats. Unmatched
  observed positions (under-prediction) now contribute their distance to the nearest model
  position — this scheme previously *dropped* them, which rewarded under-predicting models and is
  why its docstring long carried a do-not-use warning. Repeats-forbidden pairing is mainly useful
  when images are well separated and you want strict one-to-one bookkeeping.

__How the default penalizes too few / too many images__

Because ``FitPositionsImagePairAll(Solved)`` is the default, it is worth being precise about what its
mixture does in each failure mode — there are no tunable knobs, the penalties fall out of the
likelihood itself:

- **Too few images (under-prediction)**: every *observed* position marginalizes over every model
  position (a max-shifted log-sum-exp of Gaussian terms). An observed image with no model image
  nearby is dominated by its distance to the *nearest* model image, so its penalty grows
  quadratically with that distance over the position noise — automatic, and damning, exactly as
  under-prediction should be. If the solver returns no images at all, every observed position
  contributes a large finite floor (``no_image_residual = 1e4``, on the same (residual/noise)^2
  scale as the other schemes) so the model is *scored* terribly rather than silently resampled.

- **Too many images (over-prediction)**: extra model positions enter through the mixture's
  ``1/n_permutations`` normalization, which costs ``n_observed x log(n_model)`` — a *mild,
  logarithmic* Occam penalty per extra image (the ``1/P^I`` term of the Lombardi 2024 mixture).
  This penalty is magnification-blind: the ubiquitous demagnified central image is tolerated by
  construction, with no threshold to tune — but so is a *bright* predicted image on blank sky,
  which only pays the same gentle factor. If your science demands a hard penalty for bright
  unobserved images, that is what ``FitPositionsImagePairRepeat``'s ``unmatched_model_policy``
  (next section) provides — and in every case the image-plane validation of the max-likelihood
  model should inspect the unmatched model images before you trust a fit.

__The over-prediction policy (FitPositionsImagePairRepeat)__

The ``unmatched_model_policy`` class attribute selects what happens to model images no observed
position paired to:

- ``"magnification_filter"`` (default): model images with absolute magnification below
  ``magnification_threshold`` (default 0.1) are exempt — the demagnified-central convention —
  and every *other* unmatched model image adds its distance to the nearest observed position as
  a residual (normalized by the mean position noise).
- ``"penalize"``: as above with no magnification exemption.
- ``"ignore"``: extra images cost nothing — the historical behaviour, now an explicit opt-in.

Switching policy uses the class-attribute pattern (no constructor plumbing):

    class FitStrict(al.FitPositionsImagePairRepeat):
        unmatched_model_policy = "penalize"

    analysis = al.AnalysisPoint(dataset=dataset, solver=solver, fit_positions_cls=FitStrict)

The ``n_unmatched_model_positions`` property reports how many extras the policy counted — worth
inspecting on any max-likelihood fit before trusting it.

__Solver settings that masquerade as physics__

The ``PointSolver`` tiles the image plane in triangles and refines toward the source position; a
too-coarse starting grid can *miss* a genuine image entirely. That looks exactly like model
under-prediction — but it is a numerical artifact, and it will bias the sampler for numerical
rather than physical reasons. Rules of thumb at cluster scale:

- The starting grid must resolve the smallest image separation you care about: member-galaxy-scale
  perturbations produce image pairs separated by ~1", so grids much coarser than that will merge
  or miss them.
- ``pixel_scale_precision`` sets the refinement floor; the cluster profiling scripts
  (``autolens_profiling/scripts/cluster/likelihood_breakdown/image_plane.py``) time the cost of tightening
  it (solve ~0.3 s/call at a 200x200 @ 0.7" grid and 0.01" precision, with a ~10 s one-off JAX
  compile per source plane).
- If a fit reports under-prediction, re-solve the max-likelihood model at double resolution
  before believing it: if the missing image appears, it was the grid.

__Source-plane vs image-plane chi-squared__

The source-plane chi-squared (``FitPositionsSource``, Lenstool's default) ray-traces observed
images backwards and never solves the lens equation — it is ~100x cheaper per evaluation (3 ms vs
0.3 s on the standard cluster model, per the profiling breakdowns) and pairing is trivial because
every observed image maps to one source. Its costs: it cannot see over-prediction *at all* (no
forward solve, so extra images never exist), magnification weighting only approximates the
image-plane noise mapping, and the magnification amplification gives it a documented precision
floor on high-magnification systems (see ``autolens_workspace_test/scripts/cluster/
likelihood_sanity.py``) — penalty terms of the image-plane policies sit far above that floor, but
sub-percent mass perturbations do not.

The pragmatic workflow at cluster scale: **search with the source-plane chi-squared, validate with
the image-plane chi-squared** — run the image-plane fit (and inspect ``n_unmatched_model_positions``
plus the per-system image counts) on the max-likelihood model before publishing, exactly as the
Lenstool-users example does. The solved-centre variants below sharpen this advice further.

__The second axis: free vs analytically-solved source centre__

Every fit above anchors its prediction to a source-plane centre — the ``centre`` of the model's
``al.ps.Point`` (or ``al.ps.PointFlux``), sampled as two non-linear parameters per point source.
That centre can instead be **solved analytically** from the observed positions and the current mass
model, dropping out of the non-linear parameter space entirely. The model component for this is
``al.ps.PointSolved``, which is parameter-free, and each fit class gains a ``*Solved`` sibling:

| Free-centre fit | Solved-centre sibling |
|---|---|
| ``FitPositionsSource`` | ``FitPositionsSourceSolved`` |
| ``FitPositionsImagePairRepeat`` | ``FitPositionsImagePairRepeatSolved`` |
| ``FitPositionsImagePairAll`` | ``FitPositionsImagePairAllSolved`` |
| ``FitPositionsImagePair`` | — (deliberately none; see below) |
| ``FitFluxes`` | ``FitFluxesSolved`` |
| ``FitTimeDelays`` | ``FitTimeDelaysSolved`` |

The Hungarian ``FitPositionsImagePair`` has no solved sibling: its linear-sum-assignment pairing is
numpy-only (not JAX-differentiable) and the repeat/all-pairs solved variants supersede it.

The payoff is dimensionality: each point source loses 2 free parameters (and ``FitFluxesSolved``
drops the ``flux`` parameter too, so ``PointSolved`` alone covers positions + fluxes + time-delay
datasets). A 10-source cluster fit loses 20 parameters — at cluster scale, where sources are many
and each adds little individual constraint on the mass model, this is where the gain is largest.

Mixing the two conventions is an error in both directions, raised loudly as
``PointProfileMismatchException``: a ``*Solved`` fit given a centre-bearing profile would sample two
parameters the analytic solve silently ignores, and a free-centre fit given ``PointSolved`` has no
centre to read.

__The solved source-plane chi-squared (FitPositionsSourceSolved)__

The solved source-plane fit follows Lombardi (2024, arXiv:2406.15280, §5.1): Taylor-expanding the
lens equation around each observed image position makes the back-traced source position linear in
the source centre, so the optimal centre has a closed form. Each back-traced position ``β̂ᵢ`` is
weighted by its precision tensor ``Wᵢ = Aᵢ⁻ᵀ Θᵢ Aᵢ⁻¹`` (``A = ∂β/∂θ`` is the lensing Jacobian,
``Θᵢ = σᵢ⁻² I`` the image-plane precision), and the solved centre is the precision-weighted mean

    β* = (Σᵢ Wᵢ)⁻¹ Σᵢ Wᵢ β̂ᵢ

with the likelihood analytically marginalized over the centre (a flat prior; the fit's
``marginalization_term`` property is the resulting log-determinant contribution).

The tensor weighting is also a better error model than the scalar ``µ²/σ²`` weighting of
``FitPositionsSource``. The eigenvalues of ``Wᵢ`` are ``λ²/σᵢ²`` with ``λ`` the linear stretch
along each eigendirection: isotropically each stretch is ``√µ``, while near a critical curve the
tangential stretch is ``≈ µ`` — the scalar ``µ²`` convention coincides with the tensor only in that
near-critical tangential limit, so it over-weights images everywhere else. The
``weighting`` class attribute selects the convention: ``"jacobian"`` (default, the tensor) or
``"magnification"`` (the scalar, retained for comparisons with the traditional Lenstool-style
convention).

__Solved image-plane variants__

``FitPositionsImagePairRepeatSolved`` and ``FitPositionsImagePairAllSolved`` reuse the same solved
``β*`` to drive the forward lens-equation solve, then apply their scheme's image-plane pairing
chi-squared unchanged. These are **not** from Lombardi (2024) — the paper keeps the centre free in
its image-plane likelihoods. They are a PyAutoLens extension in the spirit of glafic's
source-position optimization (Oguri 2010, PASJ 62, 1017), which likewise eliminates the source
position from the sampled space of an image-plane chi-squared.

__Solved fluxes and time delays__

``FitFluxesSolved`` solves the source flux the same way (following Lombardi 2024 §6.1, ported to
flux space to match PyAutoLens's flux-space Gaussian noise maps): ``F* = Σᵢ µᵢ f̂ᵢ/σᵢ² / Σᵢ µᵢ²/σᵢ²``
— magnification-first, mirroring ``FitFluxes.model_data = |µᵢ|·F`` — plus its marginalization term.
``FitTimeDelaysSolved`` replaces the reference-image min-subtraction of ``FitTimeDelays`` with a
precision-weighted analytic reference time ``T*``. Both are selected via the ``fit_flux_cls`` /
``fit_time_delays_cls`` inputs of ``FitPointDataset`` / ``AnalysisPoint``, mirroring
``fit_positions_cls``.

__Missing-image penalty__

``FitPositionsImagePairAll``'s mixture normalization already implements the principled
over-prediction Occam factor (the ``1/P^I`` term of Lombardi 2024's mixture likelihood).
``FitPositionsImagePairRepeat``'s ``unmatched_model_policy`` heuristics stay as they are:
best-match pairing is not a normalized mixture, so no principled combinatorial term applies to it.

__Benchmark evidence (truth-anchored, A100)__

The defaults above are not conventions — they were decided by a truth-anchored benchmark campaign
(PyAutoLens#678): every likelihood option was evaluated *at the simulator-truth model* on
galaxy-scale (quad) and cluster-scale (multi-plane, two sources) datasets, and searches were scored
by ``delta = max_log_likelihood - truth_log_likelihood``. A small positive delta means the search
found the truth basin; a large positive delta means the likelihood *mis-ranks* models (a wrong
model beats truth — the failure a default must never have); a negative delta means the search
failed to reach the basin. The result JSONs live in ``autolens_profiling/results/searches/`` with
a written synthesis in ``autolens_profiling/results/notes/point_source_defaults_campaign.md``.
The headline numbers:

- **Pairing robustness (why all-to-all is the default).** On a quad with one true image removed
  from the dataset — a missing image, e.g. lost under the lens light — ``PairAllSolved`` recovered
  truth cleanly (delta +1.3) while ``PairRepeatSolved`` mis-ranked it catastrophically (truth
  log likelihood -183389, delta +183402): nearest-image pairing has no way to leave an unobserved
  model image unmatched, so the truth pays a huge spurious residual. On clean data the two pairings
  are statistically equivalent (delta +2.85 vs +2.88) at near-identical cost (147 s vs 163 s
  Nautilus wall on an A100) — robustness, not performance, decides the default.

- **Tensor vs scalar source-plane weighting (why ``"jacobian"`` is the default).** The scalar
  ``µ²/σ²`` weighting mis-ranks models when magnifications are extreme: on the galaxy quad
  (one image at |µ| = 367) the true model's scalar source-plane log likelihood is -33788 while
  wrong models score around -300 — a catastrophic inversion. The tensor weighting ranks truth
  first at both tiers (galaxy delta +0.6; cluster truth log likelihood +61 vs +36 scalar-solved
  and +40 free-scalar). The solved centre is the orthogonal win — it removes parameters — but the
  *weighting* is what fixes the ranking.

- **Free vs solved centres under gradient searches.** ``af.MultiStartProdigy`` converges on the
  solved image-plane likelihood (delta +1.95, 118 s) but stalls below truth with free centres
  (delta -75.7 at 256 starts) — and this persists after the log-sum-exp stabilization, so it is a
  genuine geometry effect, not a numerical cliff. Nautilus handles both. If you use gradient
  searches, use solved centres.

- **Posterior honesty.** The solved image-plane fits are plug-in profiles (no marginalization term
  over the centre), raising the worry that their posteriors are overconfident. The like-for-like
  Nautilus comparison shows the opposite on the benchmark quad: the solved fit's
  ``einstein_radius`` standard deviation is 0.038 vs 0.029 free — slightly wider, not narrower.

- **Near-caustic domain of validity.** With the source at 0.95x the tangential caustic the tensor
  source-plane fit still recovers truth (delta +2.0), as do the solved image-plane fits — no
  breakdown of the linearization was observed at this proximity.

- **Spurious extra positions.** The complementary discriminator — one spurious observed position
  injected into the dataset — did not run to completion for *either* pairing: both searches were
  stopped by an 8 hour wall clock after roughly 200x the wall of their clean-data siblings. That
  non-result is the finding. A spurious position cannot be explained by any model, so it imposes
  a floor of order -1e4 on the log likelihood of every sample alike; the live set never
  compresses (the all-to-all arm finished with 92% of its points still live and an effective
  sample size of 12) and the sampler cannot converge. The practical warning is therefore stronger
  than a slowdown: a contaminant position destroys convergence outright rather than quietly
  biasing the fit — so a fit that will not converge is itself evidence to re-examine your
  positions. Vet your position catalogues. Because both pairings fail here symmetrically, this
  arm does not discriminate between them; the missing-image arm above is what decides the
  default.

__Choosing at cluster scale__

Profiling on the standard cluster model (see the likelihood-breakdown scripts referenced above)
puts the analytic ``β*`` solve at timing-noise-level overhead per likelihood call — net +3% on the
image-plane likelihood, +9–30% on sub-0.1 s eager source-plane totals — while removing 2 free
parameters per point source from the non-linear space. The recommendation for cluster fits is
therefore to sharpen the workflow above: **search with ``FitPositionsSourceSolved`` (with
``al.ps.PointSolved`` sources, tensor weighting), validate with the default image-plane
chi-squared** on the max-likelihood model. Reserve free-centre ``FitPositionsSource`` for direct
comparisons with codes that sample the source position (its ``weighting = "magnification"`` scalar
convention matches Lenstool's). One search-strategy caveat from the benchmarks: at cluster scale
the gradient optimizer was defeated by every objective tested, converging to basins hundreds to
thousands of log likelihood below truth (source-plane tensor -11062, source-plane solved -15,
image-plane solved -1724) while Nautilus recovered each of them. **Use Nautilus for cluster-scale
point-source searches.** Gradient searches remain the right tool at galaxy scale with solved
centres; at cluster scale they are not yet competitive on any of these likelihoods.

__Demonstration__

The code below builds a toy under- and over-predicting fit so the policies are visible in numbers,
using a mock solver so it runs in seconds.
"""

import numpy as np

import autolens as al

"""
An isothermal lens with a point source: two bright observed images near the Einstein radius. The
mock solver lets us hand the fit whatever "model" images we want, isolating the pairing behaviour
from the lens equation.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
)
source = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.0, 0.0)))
tracer = al.Tracer(galaxies=[lens, source])

data = al.Grid2DIrregular([(0.0, 1.05), (0.0, -0.95)])
noise_map = al.ArrayIrregular([0.5, 0.5])

"""
__Case 1 — perfect prediction plus a demagnified central image__

The model predicts both observed images exactly, plus a third image at 0.01" from the lens centre
where an isothermal profile demagnifies to |mu| ~ 0.01. Under the default policy the central image
is exempt: chi-squared stays zero.
"""
model_data = al.Grid2DIrregular([(0.0, 1.05), (0.0, -0.95), (0.0, 0.01)])
solver = al.m.MockPointSolver(model_positions=model_data)

fit = al.FitPositionsImagePairRepeat(
    name="point_0", data=data, noise_map=noise_map, tracer=tracer, solver=solver
)
print("Case 1 — extra demagnified central image (default policy):")
print(f"  n_unmatched_model_positions = {int(fit.n_unmatched_model_positions)}")
print(f"  chi_squared = {float(fit.chi_squared):.4f}  (exempt below |mu| = 0.1)\n")

"""
__Case 2 — a bright unobserved image__

Move the extra image out to 3" — magnification order unity, no exemption. The model now pays for
predicting an image the data does not show.
"""
model_data = al.Grid2DIrregular([(0.0, 1.05), (0.0, -0.95), (0.0, 3.0)])
solver = al.m.MockPointSolver(model_positions=model_data)

fit = al.FitPositionsImagePairRepeat(
    name="point_0", data=data, noise_map=noise_map, tracer=tracer, solver=solver
)
print("Case 2 — bright unobserved image:")
print(f"  n_unmatched_model_positions = {int(fit.n_unmatched_model_positions)}")
print(f"  chi_squared = {float(fit.chi_squared):.4f}\n")

"""
__Case 3 — under-prediction__

The model produces only one of the two observed images. The unmatched observed image pays its full
distance to the surviving image — under-prediction is never free, under any scheme or policy.
"""
model_data = al.Grid2DIrregular([(0.0, 1.05)])
solver = al.m.MockPointSolver(model_positions=model_data)

fit = al.FitPositionsImagePairRepeat(
    name="point_0", data=data, noise_map=noise_map, tracer=tracer, solver=solver
)
print("Case 3 — missing image:")
print(f"  residuals = {[round(float(r), 3) for r in np.asarray(fit.residual_map)]}")
print(f"  chi_squared = {float(fit.chi_squared):.4f}")

"""
__Case 4 — solved source centre__

The solved variants swap the source's ``al.ps.Point`` (free centre) for the parameter-free
``al.ps.PointSolved``. The source-plane fit needs no solver at all: it back-traces the observed
positions and solves the centre analytically. The fit exposes the solved centre via
``source_plane_coordinate`` and the analytic-marginalization contribution via
``marginalization_term``.
"""
source_solved = al.Galaxy(redshift=1.0, point_0=al.ps.PointSolved())
tracer_solved = al.Tracer(galaxies=[lens, source_solved])

fit = al.FitPositionsSourceSolved(
    name="point_0", data=data, noise_map=noise_map, tracer=tracer_solved, solver=None
)
print("Case 4 — solved source centre (source-plane fit, no free centre parameters):")
print(
    f"  solved centre beta* = {tuple(round(float(c), 4) for c in fit.source_plane_coordinate)}"
)
print(f"  chi_squared = {float(fit.chi_squared):.4f}")
print(f"  marginalization_term = {float(fit.marginalization_term):.4f}")

"""
For the production-scale picture — real solver, multi-plane cluster tracer, timings —
see ``scripts/cluster/likelihood_function.py`` and the profiling breakdowns referenced above.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
