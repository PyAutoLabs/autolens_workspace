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
over/under-prediction policies, the solver settings that interact with them at cluster scale, and
the source-plane vs image-plane chi-squared trade-off. It is the reference the cluster examples
(``scripts/cluster/``, including the Lenstool walkthrough in ``scripts/cluster/lenstool/``) point
at for likelihood choices.

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

- ``FitPositionsImagePairRepeat`` (the model-fit default): every observed position pairs to its
  *nearest* model position, repeats allowed. Under-prediction is penalized by construction (an
  unmatched observed image pays its distance to the nearest surviving image; if the solver returns
  no images at all, a large finite floor applies). Over-prediction is governed by the
  ``unmatched_model_policy`` described below.

- ``FitPositionsImagePair``: Hungarian (linear-sum-assignment) pairing without repeats. Unmatched
  observed positions (under-prediction) now contribute their distance to the nearest model
  position — this scheme previously *dropped* them, which rewarded under-predicting models and is
  why its docstring long carried a do-not-use warning. Repeats-forbidden pairing is mainly useful
  when images are well separated and you want strict one-to-one bookkeeping.

- ``FitPositionsImagePairAll``: a mixture likelihood — each observed position marginalizes over
  every model position, and the 1/n_permutations normalization acts as an Occam factor that
  mildly penalizes extra images. Statistically the most principled, and differentiable end to end
  (it is the scheme the JAX point-source likelihood tests exercise); its penalties are implicit
  rather than tunable.

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
Lenstool-users example does.

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
Finished. For the production-scale picture — real solver, multi-plane cluster tracer, timings —
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
