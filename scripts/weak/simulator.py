"""
Simulator: Weak Lensing
=======================

This script simulates a weak gravitational lensing shear catalogue. Unlike the imaging simulator (which produces
a 2D image of the lensed source) the weak-lensing simulator produces a *catalogue* of (gamma_2, gamma_1) shear
measurements at the (y, x) positions of a population of background source galaxies.

The shear computation itself comes from `Tracer.shear_yx_2d_via_hessian_from`, which differentiates the
deflection-angle field. On top of that the simulator adds Gaussian shape noise per galaxy (the dominant noise
source in real weak-lensing data — each galaxy has a random unlensed ellipticity around 0.2-0.4 per component).

__Contents__

**Model:** Compose the lens model the shear field is computed from.
**Dataset Paths:** The `dataset_type` and `dataset_name` define the on-disk output folder.
**Ray Tracing:** Build a Tracer from an Isothermal lens galaxy.
**Source Positions:** Draw a uniform-random distribution of background source galaxy positions.
**Simulator:** Construct a `SimulatorShearYX` with the desired shape-noise level and random seed.
**Output:** Save the simulated `WeakDataset` and the `Tracer` to JSON.
**Visualize:** Placeholders only — the visualization layer is built in `prompt/weak/2_visualization.md`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path

import autolens as al

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, weak-lensing shear catalogue) and
`dataset_name` gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The shear catalogue will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/dataset.json`.
 - The tracer used to simulate the dataset will be output alongside as `tracer.json`.
"""
dataset_type = "weak"
dataset_name = "simple"

dataset_path = Path("dataset") / dataset_type / dataset_name

"""
__Ray Tracing__

We define the lens galaxy's mass distribution as an `Isothermal` profile (no external shear, no source light —
weak-lensing measurements are sensitive to the shear field induced by the lens mass alone).

Because the source-galaxy positions are an irregular catalogue rather than a 2D pixel grid, this simulator
does not need PSF convolution, over-sampling, or background-sky modelling — those are all imaging-specific
concerns.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(redshift=1.0)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Simulator__

`SimulatorShearYX` takes a shape-noise level and an optional random seed. A `noise_sigma` of 0.3 is a typical
ground-based survey value; reduce it to 0.0 to inspect the noise-free shear field.

The `via_tracer_random_positions_from` helper draws `n_galaxies` uniform-random source positions inside a square
of half-width `grid_extent` (in arc-seconds). For finer control, build your own `aa.Grid2DIrregular` of (y, x)
positions and call `simulator.via_tracer_from(tracer=tracer, grid=grid)` instead.
"""
simulator = al.SimulatorShearYX(noise_sigma=0.3, seed=1)

dataset = simulator.via_tracer_random_positions_from(
    tracer=tracer,
    n_galaxies=200,
    grid_extent=3.0,
    name=dataset_name,
)

"""
__Output__

Save the simulated `WeakDataset` and the `Tracer` to the dataset folder as JSON, ensuring the inputs to the
simulation are reproducible and inspectable later.
"""
dataset_path.mkdir(parents=True, exist_ok=True)

al.output_to_json(obj=dataset, file_path=dataset_path / "dataset.json")
al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")

"""
__Visualize__

The visualization layer for `WeakDataset` is added in a follow-up prompt
(`admin_jammy/prompt/weak/2_visualization.md`). It will use `matplotlib.quiver` (with `headwidth=0` so each
shear vector is drawn as a headless line segment, the standard weak-lensing convention) to plot the shear
field on top of the source-galaxy positions.

For now the simulator produces the dataset — the calls below are placeholders that the visualization prompt
will replace with real plotting code.
"""
# TODO(2_visualization.md): aplt.subplot_weak_dataset(dataset=dataset)
# TODO(2_visualization.md): aplt.plot_shear_yx_2d(
#     shear_yx=dataset.shear_yx, output_path=dataset_path, output_format="png"
# )
print(dataset.info)
print(f"Wrote dataset to {dataset_path}")
