"""
Solver Grid Check: Does The Solver Find Every Image?
====================================================

The ``PointSolver`` finds the multiple images of a point source by tiling the image plane with
triangles of a starting size, keeping the triangles whose source-plane image contains the source and
subdividing them until ``pixel_scale_precision`` is reached. Only triangles that contain the source at
the starting size are refined. An image whose image-plane region is smaller than one starting
triangle can therefore be skipped entirely, and the solver returns too few images however fine
``pixel_scale_precision`` is.

At cluster scale this happens. Images that form close to a cluster member or a critical curve occupy
small regions of the image plane. On the Simple cluster dataset, a 1" starting grid finds four of the
five images of ``point_0``: the image at (10.36", 8.14") is skipped, and its nearest prediction is
2.8" away.

This script checks a solver grid against a lens model. For each source and each candidate starting
scale it solves the lens equation and reports:

 - how many observed images have a predicted image within 3 times their positional uncertainty, with
   each observed image paired to a distinct prediction;
 - whether the grid finds every image that a fine 0.1" reference grid finds.

Use the coarsest scale that passes both checks for every source. A finer starting grid costs more per
solve, so there is no reason to go finer than that.

The cluster scripts in this workspace start at 0.1". Checked on Simple, Pandora, MACS J0416 and SMACS
0723, every coarser scale (1", 0.5", 0.25" and 0.2") missed an image on at least one cluster, and a
finer scale is not always safer: on Pandora and MACS J0416, 0.2" missed images that 0.25" found,
because whether a small image region is caught depends on how the triangles line up with it. If this
check passes a coarser scale on every source of your own cluster, you can use it to speed up your fits.

A model that cannot reproduce an observed image fails the first check at every scale, including the
0.1" reference. That is a property of the lens model, not of the grid, and the table shows it as a
miss at the reference scale.

__Contents__

- **Imports:** The libraries we'll use.
- **Dataset:** The point datasets and true tracer of the Simple cluster.
- **Source Positions:** The source-plane position of each source.
- **Candidate Grids:** The starting scales to check and the field they cover.
- **Recovery Check:** Solve every source at every scale and compare with the observations.
- **Results:** The per-source table and the coarsest passing scale.
- **Your Own Cluster:** How to run the check on your own lens model.
"""

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autolens as al
import numpy as np
from scipy.optimize import linear_sum_assignment

"""
__Dataset__

We use the Simple cluster dataset, which ``simulator.py`` creates if it is not already on your
system. Its true ``Tracer`` is the lens model we check the grid against.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "cluster", dataset_name)

al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/cluster/simulator.py"
)

dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
tracer = al.from_json(file_path=dataset_path / "tracer.json")

"""
__Source Positions__

Solving the lens equation needs each source's source-plane position. If the tracer has a ``Point``
component named after the dataset, its centre is used.

Otherwise the position is solved analytically from the observed images with
``al.FitPositionsSourceSolved``, the same calculation the cluster fits use. This covers a fitted model
whose sources are parameter-free ``al.ps.PointSolved`` components (the cluster default) and the
Simple tracer saved by ``simulator.py``, whose sources carry only light. ``source_centre_from`` adds a
``PointSolved`` component at the dataset's redshift so that the fit can pair it with the dataset.
"""


def source_centre_from(dataset, tracer):
    for galaxy in tracer.galaxies:
        point = getattr(galaxy, dataset.name, None)
        if point is not None and hasattr(point, "centre"):
            return tuple(point.centre)

    solved_source = al.Galaxy(
        redshift=dataset.redshift, **{dataset.name: al.ps.PointSolved()}
    )
    fit = al.FitPointDataset(
        dataset=dataset,
        tracer=al.Tracer(
            galaxies=list(tracer.galaxies) + [solved_source],
            cosmology=tracer.cosmology,
        ),
        solver=None,
        fit_positions_cls=al.FitPositionsSourceSolved,
    )
    return tuple(float(value) for value in fit.positions.source_plane_coordinate)


"""
__Candidate Grids__

Each candidate solver covers the same square field, here the 100" field used by ``modeling.py``, so
its ``shape_native`` is the field width divided by the starting scale. The field must contain every
observed image with some margin.

The last scale, 0.1", is the reference grid. The refinement precision and magnification threshold
match ``modeling.py``.
"""
field = 100.0
scales = [1.0, 0.5, 0.25, 0.2, 0.1]
pixel_scale_precision = 0.001
magnification_threshold = 0.1

# A trial image matches a reference image within this distance (arcsec).
reference_tolerance = 0.01
# An observed image is recovered within this many positional uncertainties.
sigma_factor = 3.0


def solver_for(scale):
    shape = round(field / scale)
    grid = al.Grid2D.uniform(shape_native=(shape, shape), pixel_scales=scale)
    return al.PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=pixel_scale_precision,
        magnification_threshold=magnification_threshold,
    )


"""
__Recovery Check__

For every source and scale we solve the lens equation, then:

 - pair observed and predicted images one-to-one, minimising the total separation (the Hungarian
   algorithm, ``scipy.optimize.linear_sum_assignment``), and count the observed images whose paired
   prediction lies within ``sigma_factor`` positional uncertainties;
 - count the reference images with no trial image within ``reference_tolerance``.
"""


def solve(solver, dataset):
    predicted = solver.solve(
        tracer=tracer,
        source_plane_coordinate=source_centre_from(dataset, tracer),
        plane_redshift=dataset.redshift,
    )
    return np.atleast_2d(np.asarray(predicted)).reshape(-1, 2)


def observed_recovered(dataset, predicted):
    observed = np.atleast_2d(np.asarray(dataset.positions))
    noise = np.asarray(dataset.positions_noise_map)
    if len(predicted) == 0:
        return 0
    separations = np.linalg.norm(observed[:, None] - predicted[None, :], axis=-1)
    rows, cols = linear_sum_assignment(separations)
    return int(np.sum(separations[rows, cols] <= sigma_factor * noise[rows]))


def reference_missed(predicted, reference):
    if len(predicted) == 0:
        return len(reference)
    separations = np.linalg.norm(reference[:, None] - predicted[None, :], axis=-1)
    return int(np.sum(separations.min(axis=1) > reference_tolerance))


solvers = {scale: solver_for(scale) for scale in scales}

table = {}
for dataset in dataset_list:
    predicted = {scale: solve(solvers[scale], dataset) for scale in scales}
    reference = predicted[scales[-1]]
    table[dataset.name] = {
        scale: (
            observed_recovered(dataset, predicted[scale]),
            len(dataset.positions),
            reference_missed(predicted[scale], reference),
        )
        for scale in scales
    }

"""
__Results__

Each cell reads ``recovered / observed``, followed by ``ref ok`` when the grid finds every reference
image, or the number of reference images it misses. A scale passes when every source recovers all
its observed images and misses no reference image.
"""
print(f'Field {field}" | precision {pixel_scale_precision}"')
print(f"{'scale':>7} " + " ".join(f"{name:>16}" for name in table))

passing = []
for scale in scales:
    cells = []
    passes = True
    for recovered, observed, missed in (table[name][scale] for name in table):
        reference_text = "ref ok" if missed == 0 else f"ref -{missed}"
        cells.append(f"{recovered}/{observed} {reference_text:>8}")
        passes = passes and recovered == observed and missed == 0
    if passes:
        passing.append(scale)
    print(f'{scale:>6}" ' + " ".join(f"{cell:>16}" for cell in cells))

if passing:
    print(f'\nCoarsest scale finding every image: {max(passing)}"')
else:
    print(
        '\nNo scale recovers every observed image. If the 0.1" reference also misses an '
        "image, the lens model cannot reproduce it; check the model before the grid."
    )

"""
__Your Own Cluster__

To check the grid for your own cluster, replace ``dataset_list`` with your point datasets and
``tracer`` with your max-likelihood tracer (``result.max_log_likelihood_tracer``), and set ``field`` to
the width of the solver grid in your modeling script. Run the check again whenever the model changes
substantially, for example after adding families near cluster members.
"""
