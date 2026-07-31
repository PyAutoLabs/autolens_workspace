The `pixelization` folder contains example scripts for fitting a multi-galaxy strong lens with a **pixelized
source reconstruction** — the source reconstructed on a mesh of pixels whose fluxes are solved by linear algebra,
rather than assumed to follow an analytic profile.

# The multi-galaxy difference

At galaxy scale, a pixelized source is close to an unqualified improvement: real sources are irregular, an
analytic profile cannot represent them, and the residuals it leaves get absorbed by the mass model.

In this regime it carries a specific cost, and it is large enough to change how you use the feature.

`multi_galaxy/modeling.py` establishes the regime's central degeneracy: the data constrains the **total**
deflection of a multi-galaxy lens well, and the **split** between the deflectors much less well. A free-form
source mesh weakens the constraint on that split further, because when the split is wrong the mesh can rearrange
itself to keep reproducing the arcs — where a parametric source, having a fixed functional form, simply fails.

`fit.py` measures it. Holding the pair's total Einstein radius at its true 1.8" and varying **only** how it is
apportioned between the two deflectors:

| split | r_0 / r_1 | parametric | pixelized |
|---|---|---|---|
| 0.556 (**true**) | 1.00 / 0.80 | 12,379 | 11,467 |
| 0.600 | 1.08 / 0.72 | −7,955 | 5,907 |
| 0.700 | 1.26 / 0.54 | −110,273 | −32,606 |
| 0.800 | 1.44 / 0.36 | −182,813 | −76,774 |

As the penalty each model pays relative to its own best:

| split | parametric | pixelized | ratio |
|---|---|---|---|
| 0.600 | 20,334 | **5,560** | 3.7× |
| 0.700 | 122,652 | **44,073** | 2.8× |
| 0.800 | 195,193 | **88,241** | 2.2× |

A **26% error** in `lens_0`'s Einstein radius costs a parametric fit 20,334 in log likelihood and a pixelized fit
only 5,560 — and leaves the pixelized fit still *above* the parametric model's value at the true split. The mesh
absorbs roughly three quarters of the evidence against a wrong mass split.

At the true split the parametric source wins slightly (12,379 vs 11,467), which is the expected sanity check: the
simulated source *is* a cored Sérsic, so the parametric model has exactly the right functional form and the mesh
pays a regularization penalty to approximate it.

Log likelihoods shift a percent or two per re-simulation (unseeded Poisson noise); the pattern does not.

# What follows from it

None of this argues against pixelized sources — a Sérsic fitted to a genuinely irregular source biases the mass
model in its own way, which is why `multi_galaxy/slam.py` moves to a pixelized source for its final stages. It
argues for three things the package already does:

- **Use the positions likelihood.** Multiple-image positions constrain the mass model *directly*, independent of
  how well the source is reconstructed — precisely the constraint a free-form mesh weakens.
- **Do not free the mass split and the source at the same time.** SLaM's ordering (initialize the mass with a
  parametric source, then hold it while the source becomes pixelized) is what stops the two absorbing each other.
- **Check the reconstructed source for structure that tracks the deflectors** rather than its own morphology.
  That is the visible signature of the mesh absorbing a mass-model error.

# Mesh and regularization pairings

Two constraints, both of which fail loudly rather than silently:

- **`AdaptSplit` will not pair with a rectangular mesh.** It regularizes using a cross of four points around each
  pixel centre and needs split-cross mappings the rectangular meshes do not provide; the pairing raises a
  `PixelizationException` naming the incompatibility. The Delaunay meshes do support it.
- **The `Adapt` schemes need adapt-images**, an estimate of the source's surface brightness. A standalone script
  has no earlier fit to derive one from, so these examples use `Constant`. `multi_galaxy/slam.py` uses `Adapt`
  because by that stage `source_pix[1]` has produced one.

# Files

- `modeling`: Fitting a multi-galaxy lens with a pixelized source, and the positions likelihood that compensates.
- `fit`: The measurement above, run directly — the mass-split experiment with both source models.

# Not yet written

`adaptive`, `delaunay`, `cpu_fast_modeling`, `slam`, `source_science` and `plot` are present in
`imaging/features/pixelization` but not yet here. Until they land, those scripts apply with the single lens galaxy
swapped for the `lens_0`, `lens_1`, ... loop of this package.

# Related

- `multi_galaxy/modeling`: the parametric-source fit of the same lens, and the mass-split degeneracy.
- `multi_galaxy/slam.py`: the pipeline whose stage ordering exists to manage this.
- `imaging/features/pixelization`: the galaxy-scale walkthrough, with the full mesh and regularization API.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
