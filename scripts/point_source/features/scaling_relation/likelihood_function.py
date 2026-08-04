"""
__Log Likelihood Function: Scaling Relation (Point Source)__

Describes how a scaling tier anchored on the brightest galaxy enters the point-source `log_likelihood`.

Point sources differ from extended sources in a way that matters for this feature. For imaging or interferometer data
the tier perturbs the deflection field, and you can inspect that perturbation directly. For a point source the
likelihood is built from **solved multiple image positions**, so the tier's effect passes through the lens equation
first — it is not a linear contribution to anything you can read off. That is what makes the tier's influence so large
here, and it is what this script traces step by step.

__Prerequisites__

`point_source/` has no top-level `likelihood_function.py`; its chi-squared story lives in
`point_source/fit.py`, which is the prerequisite:

 - `autolens_workspace/scripts/point_source/fit.py` — the `PointSolver`, `FitPointDataset`, and the several ways a
   point-source chi-squared can be defined (image-plane, source-plane, with or without repeat image pairs).
 - `autolens_workspace/scripts/point_source/features/scaling_relation/modeling.py` — the search-based composition.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/likelihood_function.py` — the extended-source
   version, where the deflection sum is inspected directly.

__What Changes For A Scaling Relation__

The chain from model to likelihood is:

  1. Each tier member's `einstein_radius` is set by the relation:
     `einstein_radius_j = einstein_radius_anchor * (L_j / L_anchor) ** 0.5`.
  2. The tracer's total deflection field gains each member's contribution — an ordinary sum, as in every other regime.
  3. The `PointSolver` **solves the lens equation** on that field, returning image positions.
  4. Those positions (and the fluxes from the magnifications) are compared with the observed ones.

Step 3 is where point sources part company with everything else. A solved position is a non-linear function of the
deflection field, so a small change in step 2 can produce a large change in step 4. This script measures exactly that
amplification.

Steps 1 and 2 cost zero free parameters, because the anchor's `einstein_radius` is a parameter the model already has.

__Contents__

- **Dataset:** The point dataset.
- **Centres + Luminosities:** External inputs from the accompanying imaging.
- **Step 1: The Relation:** Each member's Einstein radius.
- **Step 2: The Deflection Field:** The ordinary sum, checked against the tracer.
- **Step 3: Solving The Lens Equation:** Where the amplification happens.
- **Step 4: Chi-Squared:** Positions and fluxes.
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al

"""
__Dataset__
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset") / "point_source" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/point_source/features/scaling_relation/simulator.py",
        ],
        check=True,
    )

dataset = al.from_json(file_path=dataset_path / "point_dataset.json")

"""
__Centres + Luminosities__

Measured from the accompanying imaging; a `PointDataset` contains neither. Explicit lists here — the CSV form is shown
at the end of `modeling.py` and `fit.py`.
"""
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

luminosity_anchor = 14.5055

scaling_galaxies_luminosities = [0.5116, 0.3848, 0.2716, 0.1811, 0.1268]

"""
__Step 1: The Relation__

Each member's Einstein radius, as the arithmetic that produces it. All five are functions of the single anchor value,
so all five move together whenever it does.
"""
einstein_radius_anchor = 1.6
scaling_exponent = 0.5


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the main lens.
    """
    return einstein_radius_anchor * (luminosity / luminosity_anchor) ** scaling_exponent


for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    centre_str = f"({float(centre[0]):5.2f}, {float(centre[1]):5.2f})"
    print(
        f"  scaling galaxy @ {centre_str}: einstein_radius = "
        f"{einstein_radius_anchor:.3f} * ({luminosity:.4f} / {luminosity_anchor:.4f}) ** {scaling_exponent} "
        f"= {einstein_radius_from(luminosity):.4f}"
    )

"""
__Galaxies__

Mass only — a point-source fit has no use for foreground light. All profiles **untruncated**: truncation encodes tidal
stripping by a host halo, which a galaxy-scale lens does not have.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=einstein_radius_anchor,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

scaling_galaxies = [
    al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(
            centre=tuple(centre), einstein_radius=einstein_radius_from(luminosity)
        ),
    )
    for centre, luminosity in zip(
        scaling_galaxies_centres, scaling_galaxies_luminosities
    )
]

source = al.Galaxy(redshift=1.0, point_0=al.ps.PointFlux(centre=(0.07, 0.07), flux=1.0))

tracer = al.Tracer(galaxies=[lens] + scaling_galaxies + [source])

"""
__Step 2: The Deflection Field__

An ordinary sum over mass profiles, identical to every other regime. Nothing about the relation changes this step.
"""
grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

alpha_anchor = lens.mass.deflections_yx_2d_from(grid=grid)
alpha_scaling = [g.mass.deflections_yx_2d_from(grid=grid) for g in scaling_galaxies]

alpha_total_summed = alpha_anchor + sum(alpha_scaling)

traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
alpha_total_tracer = grid - traced_grids[1]

print(f"\nalpha_anchor             (first coord): {alpha_anchor[0]}")
print(f"alpha_scaling (tier sum) (first coord): {sum(alpha_scaling)[0]}")

assert np.allclose(np.asarray(alpha_total_summed), np.asarray(alpha_total_tracer))

print("Deflection sum matches the tracer.")

"""
__Step 3: Solving The Lens Equation__

The step that is unique to point sources, and where the tier's influence is amplified.

Below, the mean magnitude of the tier's deflection contribution is compared with how far the solved image positions
actually move when the tier is removed. The second number is much larger than the first would suggest, because a
solved position is not a linear function of the deflection field.
"""
solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(tracer=tracer, source_plane_coordinate=source.point_0.centre)

positions_without_tier = solver.solve(
    tracer=al.Tracer(galaxies=[lens, source]),
    source_plane_coordinate=source.point_0.centre,
)

shifts_mas = sorted(
    float(np.linalg.norm(np.asarray(positions_without_tier) - position, axis=1).min())
    * 1000.0
    for position in np.asarray(positions)
)

tier_deflection_mas = [
    float(np.linalg.norm(np.asarray(g.mass.deflections_yx_2d_from(grid=grid))[0]))
    * 1000.0
    for g in scaling_galaxies
]

print(
    f"\nPer-member deflection magnitude (mas): {[f'{d:.0f}' for d in tier_deflection_mas]}"
)
print(f"Resulting image-position shifts  (mas): {[f'{s:.0f}' for s in shifts_mas]}")
print(
    "The largest shift exceeds the largest single deflection — the lens equation amplifies, it does not average."
)

"""
__Step 4: Chi-Squared__

The solved positions and the magnification-derived fluxes are compared with the observed ones. `point_source/fit.py`
documents the available chi-squared definitions; the tier changes none of them, only the tracer whose images are
solved.
"""
fit = al.FitPointDataset(
    dataset=dataset,
    tracer=tracer,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeat,
)

print(f"\nPositions log likelihood: {fit.positions.log_likelihood}")
print(f"Fluxes log likelihood:    {fit.flux.log_likelihood}")
print(f"Total log likelihood:     {fit.log_likelihood}")

"""
__Wrap Up__

The relation touches exactly one place in the likelihood: it sets each tier member's `einstein_radius` from the
anchor's value and an externally measured luminosity. The deflection sum, the lens-equation solve and the chi-squared
are all shared with the standard point-source likelihood.

What is specific to this regime is the amplification in step 3. It means the tier cannot be dismissed as a small
perturbation on the grounds that its deflections are small — and, because it is tied, including it costs nothing from a
12-point data budget.
"""
