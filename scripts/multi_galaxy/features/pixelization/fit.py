"""
Fits: Pixelization (Multi Galaxy)
=================================

This script fits a multi-galaxy strong lens with a **pixelized source reconstruction** without a non-linear
search, and measures the one thing about pixelized sources that matters more at multi-galaxy scale than at galaxy
scale: **a free-form source absorbs an incorrect mass split.**

__Contents__

- **The Claim:** What a pixelized source can hide that a parametric one cannot.
- **Dataset, Mask & Over Sampling:** Set up.
- **Lens Composition:** The mass split as a single dial.
- **Parametric vs Pixelized:** The measurement.
- **What The Numbers Mean:** Reading the result.
- **What To Do About It:** Practical consequences.
- **Wrap Up:** Where to go next.

__The Claim__

`multi_galaxy/modeling.py` establishes the regime's central degeneracy: the data constrains the **total**
deflection of a multi-galaxy lens well, and the **split** between the deflectors much less well. The split is what
the science wants — each galaxy's mass, and the ratio between them.

A pixelized source reconstruction makes that degeneracy worse, and it is worth understanding why before using
one. A parametric source has a fixed functional form: get the mass split wrong, the ray-traced image of a Sersic
no longer matches the arcs, and the likelihood punishes you. A pixelized source has one free parameter per source
pixel. Get the mass split wrong, and the mesh can **rearrange itself** to keep reproducing the arcs — paying some
regularization penalty, but far less than the parametric model pays.

The mesh is not doing anything wrong. It is doing exactly what it is for: reconstructing whatever source best
explains the data. The problem is that "whatever source best explains the data" is a much weaker constraint on
the mass model, and at multi-galaxy scale the mass model has a degenerate direction to exploit.

This script measures the size of that effect.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `simple` multi-galaxy dataset — the same co-dominant pair fitted by `multi_galaxy/modeling.py`.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

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

Scale the faint contaminant out of the fit, as `multi_galaxy/modeling.py` explains. This matters more for a
pixelized source than a parametric one: unmodelled flux inside the mask is exactly the kind of thing a free-form
mesh will happily reconstruct as source structure.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at both deflector centres.
"""
main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=3.0,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 1],
        radial_list=[0.3, 0.6],
        centre_list=main_lens_centres,
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Lens Composition__

The experiment needs the mass split as a **single dial**, with everything else held fixed.

`split` is `lens_0`'s share of the pair's total Einstein radius, and the total is held at its true value of 1.8"
throughout. So every model below produces very nearly the same *total* deflection — which the data constrains well
— and differs only in how that deflection is apportioned between the two galaxies, which is the degenerate
direction.

The true split is 1.0 / 1.8 = 0.556.

Each galaxy also carries its true light profile, made linear so its intensity is solved rather than assumed.
Omitting the lens light entirely is a mistake worth naming: the `simple` dataset has two bright foreground
galaxies, and a model without them is dominated by foreground residuals that swamp the mass-split signal this
script is trying to isolate.
"""
TRUE_EINSTEIN_RADII = [1.0, 0.8]
TOTAL_EINSTEIN_RADIUS = sum(TRUE_EINSTEIN_RADII)

true_split = TRUE_EINSTEIN_RADII[0] / TOTAL_EINSTEIN_RADIUS


def lens_galaxies_from(split: float):
    """
    The co-dominant pair, with `split` controlling how the fixed total Einstein radius is apportioned.
    """
    einstein_radius_0 = TOTAL_EINSTEIN_RADIUS * split
    einstein_radius_1 = TOTAL_EINSTEIN_RADIUS - einstein_radius_0

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
            einstein_radius=einstein_radius_0,
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
            einstein_radius=einstein_radius_1,
        ),
    )

    return [lens_0, lens_1], einstein_radius_0, einstein_radius_1


shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__The Two Source Models__

A parametric source (the truth's functional form, so it is given every advantage) and a pixelized one.

The pixelization is a `RectangularUniform` mesh with `Constant` regularization — the simplest possible choice, so
the effect measured below is attributable to the mesh's freedom rather than to an adaptive scheme's cleverness.
The better meshes are covered in `imaging/features/pixelization` (`adaptive.py`, `delaunay.py`); they are not yet
written for this package.
"""


def parametric_fit_from(split: float):
    galaxies, _, _ = lens_galaxies_from(split)

    source = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_linear.SersicCore(
            centre=(0.0, 0.03),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            effective_radius=0.15,
            sersic_index=1.0,
        ),
    )

    return al.FitImaging(
        dataset=dataset,
        tracer=al.Tracer(galaxies=galaxies + [shear_galaxy, source]),
    )


def pixelized_fit_from(split: float):
    galaxies, _, _ = lens_galaxies_from(split)

    source = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.RectangularUniform(shape=(30, 30)),
            regularization=al.reg.Constant(coefficient=1.0),
        ),
    )

    return al.FitImaging(
        dataset=dataset,
        tracer=al.Tracer(galaxies=galaxies + [shear_galaxy, source]),
    )


"""
__The Fit At The True Split__

First, both models at the correct mass split, to see the baseline.
"""
parametric_truth = parametric_fit_from(true_split)
pixelized_truth = pixelized_fit_from(true_split)

aplt.subplot_fit_imaging(fit=pixelized_truth)

print(f"True split = {true_split:.3f} (r_0 = 1.00\", r_1 = 0.80\")")
print(f"  parametric source : {parametric_truth.log_likelihood:.1f}")
print(f"  pixelized source  : {pixelized_truth.log_likelihood:.1f}")

"""
The parametric source wins slightly at the true split, which is expected and reassuring: the simulator's source
*is* a cored Sersic, so the parametric model has the exactly correct functional form and the mesh is paying a
regularization penalty to approximate it.

__Parametric vs Pixelized At A Wrong Split__

Now walk the mass split away from the truth, holding the total fixed, and watch how much likelihood each source
model gives up.
"""
print(f"\n{'split':>6} {'r_0':>5} {'r_1':>5} | {'parametric':>12} {'pixelized':>12}")

for split in (true_split, 0.60, 0.70, 0.80):

    _, einstein_radius_0, einstein_radius_1 = lens_galaxies_from(split)

    parametric_log_likelihood = parametric_fit_from(split).log_likelihood
    pixelized_log_likelihood = pixelized_fit_from(split).log_likelihood

    print(
        f"{split:6.3f} {einstein_radius_0:5.2f} {einstein_radius_1:5.2f} | "
        f"{parametric_log_likelihood:12.1f} {pixelized_log_likelihood:12.1f}"
    )

"""
__What The Numbers Mean__

Measured on this dataset (values shift a percent or two per re-simulation, since the simulator's Poisson noise is
unseeded; the *pattern* does not):

     split    r_0   r_1 |   parametric    pixelized
     0.556   1.00  0.80 |     12,379.2     11,466.6      <- true split
     0.600   1.08  0.72 |     -7,955.3      5,907.2
     0.700   1.26  0.54 |   -110,272.7    -32,605.9
     0.800   1.44  0.36 |   -182,813.4    -76,774.0

Read it as the **penalty** each source model imposes for a given error in the mass split — how far the likelihood
falls below that model's own value at the truth:

     split    parametric penalty    pixelized penalty    ratio
     0.600            20,334               5,560         3.7x
     0.700           122,652              44,073         2.8x
     0.800           195,193              88,241         2.2x

At every wrong split, the pixelized source gives up **two to four times less** likelihood than the parametric one.
That difference is the mesh rearranging itself to absorb the error.

The 0.600 row is the one to dwell on. That is a 26% error in `lens_0`'s Einstein radius — 1.08" instead of 1.00",
with `lens_1` at 0.72" instead of 0.80". A parametric fit rejects it decisively, falling 20,334 below its own
best. The pixelized fit falls only 5,560, and is still *above* the parametric model's true-split likelihood. A
search comparing models would find that wrong split far more competitive with a pixelized source than with a
parametric one.

__What This Does Not Say__

It does not say pixelized sources are wrong, or that you should use a parametric source for multi-galaxy lenses.
Real sources are irregular and clumpy; a Sersic fitted to one leaves residuals that bias the mass model in their
own way, which is why `multi_galaxy/slam.py` moves to a pixelized source for its final stages. The effect measured
here is a cost to weigh, not a verdict.

It also does not say the mesh is misbehaving. Reconstructing whatever source best explains the data is its job.
The issue is that this job is a weaker constraint on the lens model, and multi-galaxy lenses have a degenerate
direction ready to absorb the slack.

__What To Do About It__

Three practical consequences, all of which the rest of this package already does for other reasons:

 - **Use the positions likelihood.** `multi_galaxy/slam.py` passes
   `positions_likelihood_list` to its pixelized stages. Multiple-image positions constrain the mass model
   *directly*, independently of how well the source is reconstructed, which is exactly the constraint a
   free-form mesh weakens.
 - **Do not free the mass split at the same time as the source.** SLaM's stage ordering — initialize the mass with
   a parametric source (`source_lp[1]`), then hold it while the source becomes pixelized — is not an arbitrary
   convention. It is what stops the two from absorbing each other.
 - **Check the reconstructed source for structure that tracks the deflectors.** A source whose reconstruction has
   features aligned with the lens galaxies rather than with itself is the visible signature of this effect.

__Wrap Up__

Where to go next:

 - `multi_galaxy/features/pixelization/modeling.py` — fitting this model with a search.
 - `imaging/features/pixelization/adaptive.py` — meshes that adapt to the source, and `delaunay.py` for the
   Delaunay alternative (not yet written for this package).
 - `multi_galaxy/slam.py` — the pipeline whose stage ordering exists to manage exactly this.
 - `imaging/features/pixelization` — the galaxy-scale walkthrough, with the full pixelization API.
"""
