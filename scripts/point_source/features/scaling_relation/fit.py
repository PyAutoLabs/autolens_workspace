"""
Features (Point Source): Scaling Relation Fit
=============================================

Fits the point-source `scaling_relation` dataset at the simulator's truth values, so the tied tier can be inspected
without a non-linear search in the way.

For a point source the interesting quantity is not a residual image but **where the multiple images land**. This script
therefore does something the extended-source versions cannot: it solves the lens equation with the tier and again
without it, and measures how far each image moves. That number is the tier's whole physical justification.

__Prerequisites__

 - `autolens_workspace/scripts/point_source/fit.py` — the `PointSolver`, `FitPointDataset` and the several ways a
   point-source chi-squared can be defined.
 - `autolens_workspace/scripts/point_source/features/scaling_relation/modeling.py` — the search-based version.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/fit.py` — the CCD-imaging equivalent, where the
   deflection sum is inspected directly rather than through solved image positions.

__Mass Only__

A `PointDataset` is positions and fluxes, so lens and companions alike are mass-only here. The luminosities driving the
relation are measured from the accompanying imaging — see `modeling.py`. All profiles are **untruncated**; truncated
`dPIEMass` members belong to the group and cluster workflows.

__Contents__

- **Dataset:** The point dataset and its accompanying imaging.
- **Centres + Luminosities:** From the accompanying imaging, not the point data.
- **The Relation:** One function, evaluated per companion.
- **Galaxies:** Anchor, scaling tier, `PointFlux` source — all at simulator truth.
- **Point Solver:** Solve for the multiple images.
- **How Much Does The Tier Move The Images?:** Solve with and without it.
- **Fit:** `FitPointDataset` and the log likelihood.
- **CSV Interface.**
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

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

print("Point Dataset Info:")
print(dataset.info)

data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.05)

aplt.plot_array(array=data, title="Accompanying Imaging (companions visible)")

"""
__Centres + Luminosities__

Both measured from the accompanying imaging above; the point data contains neither. Explicit lists here, CSV at the end.
"""
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

luminosity_anchor = 14.5055

scaling_galaxies_luminosities = [0.5116, 0.3848, 0.2716, 0.1811, 0.1268]

"""
__The Relation__

The anchor's Einstein radius is a fixed number here (simulator truth) rather than a free parameter, so the relation
evaluates to a plain float per companion. In `modeling.py` the identical expression multiplies the model's free
`einstein_radius` instead.
"""
einstein_radius_anchor = 1.6
scaling_exponent = 0.5


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the main lens.
    """
    return einstein_radius_anchor * (luminosity / luminosity_anchor) ** scaling_exponent


"""
__Galaxies__

All at simulator truth. Mass only, and the source is a `PointFlux` named `point_0` to pair with the dataset.
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

print(
    f"\nAnchor: einstein_radius = {einstein_radius_anchor:.4f}, L = {luminosity_anchor:.4f}"
)

for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    centre_str = f"({float(centre[0]):5.2f}, {float(centre[1]):5.2f})"
    print(
        f"  scaling galaxy @ {centre_str}: "
        f"{einstein_radius_anchor:.3f} * ({luminosity:.4f} / {luminosity_anchor:.4f}) ** {scaling_exponent} "
        f"= {einstein_radius_from(luminosity):.4f}"
    )

"""
__Point Solver__
"""
grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(tracer=tracer, source_plane_coordinate=source.point_0.centre)

print(f"\nMultiple images solved with the tier: {len(positions)}")
print(positions)

"""
__How Much Does The Tier Move The Images?__

The measurement that justifies the tier. We solve the same system with the companions removed and pair each image with
its nearest counterpart.

Expect shifts of hundreds to over a thousand milliarcseconds, against the 5 mas astrometric precision of the data. The
tier is not a small correction to a point-source model — it is a dominant one. This is the opposite of the intuition
that a faint neighbour is negligible, and the reason is that an image position is the *solution* of the lens equation:
near the ring the images lie along a nearly-degenerate direction, so a 0.2" deflection slides an image much further
than 0.2" before ray-tracing rebalances.
"""
tracer_without_tier = al.Tracer(galaxies=[lens, source])

positions_without_tier = solver.solve(
    tracer=tracer_without_tier, source_plane_coordinate=source.point_0.centre
)

print(f"Multiple images solved without the tier: {len(positions_without_tier)}")

positions_array = np.asarray(positions)
positions_without_array = np.asarray(positions_without_tier)

shifts_mas = sorted(
    float(np.linalg.norm(positions_without_array - position, axis=1).min()) * 1000.0
    for position in positions_array
)

print(f"\nPer-image shift caused by the tier (mas): {[f'{s:.0f}' for s in shifts_mas]}")
print(f"Astrometric precision of the data (mas): 5")
print(f"Smallest shift is {shifts_mas[0] / 5.0:.0f}x the precision.")

assert shifts_mas[0] > 5.0, (
    "The tier should move every image by far more than the astrometric precision; if it does not, the dataset has "
    "drifted from the configuration this example describes."
)

"""
__Fit__

`FitPointDataset` compares the solved image positions and fluxes with the observed ones. `point_source/fit.py`
describes the several chi-squared definitions available; the tier changes none of them — it only changes the tracer
whose images are solved.
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
__CSV Interface__

`al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV and returns a `GalaxyTable` with `.centres`,
`.luminosities` and `.redshifts`:

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

Both columns come from one photometric measurement on the accompanying imaging, so one file for both is the natural
form here.
"""
scaling_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

print(f"\nTier luminosities from CSV: {list(scaling_table.luminosities)}")

"""
__Wrap Up__

The relation fixes what each companion's `einstein_radius` is set to; the `PointSolver` and the chi-squared are
untouched. What this script adds over its extended-source siblings is the measurement above: the tier moves every
multiple image by orders of magnitude more than the astrometric precision, so it must be in the model — and because it
is tied, having it in the model costs nothing from a 12-point data budget.

Next: `modeling.py` fits this with a search. There is no `slam.py` here — with no light in a `PointDataset` there is no
light stage in which to measure anything.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

This script asserts on solved image positions. Under SMALL_DATASETS the grid is
capped to 15x15, at which resolution `PointSolver` returns 2 degenerate images
instead of 4 and the measured shift collapses to 0 mas, tripping the assertion.

ENV: full_datasets
"""
