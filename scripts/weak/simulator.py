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

- **Model:** Compose the lens model the shear field is computed from.
- **Dataset Paths:** The `dataset_type` and `dataset_name` define the on-disk output folder.
- **Ray Tracing:** Build a Tracer from an Isothermal lens galaxy.
- **Source Positions:** Draw a uniform-random distribution of background source galaxy positions.
- **Simulator:** Construct a `SimulatorShearYX` with the desired shape-noise level and random seed.
- **Output:** Save the simulated `WeakDataset` and the `Tracer` to JSON.
- **Visualize:** Plot the shear field and the dataset subplot mosaic via `aplt`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path

import autolens as al
import autolens.plot as aplt

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

The shear field is visualised with `matplotlib.quiver` rendered as *headless line segments*
(`headwidth=0, headlength=0, headaxislength=0`) — the standard weak-lensing convention, because shear is a
spin-2 quantity and a 180-degree rotation maps it back to itself, so an arrowhead would suggest a
directionality the data does not have.

`aplt.subplot_weak_dataset` produces a 2x2 mosaic combining the shear field, the per-galaxy noise map, the
shear magnitude `|gamma|`, and the position angle `phi`. `aplt.plot_shear_yx_2d` writes a single-panel
quiver of the shear field alone — useful for high-resolution figures where the mosaic is too dense.
"""
aplt.subplot_weak_dataset(
    dataset=dataset,
    output_path=dataset_path,
    output_format="png",
)

aplt.plot_shear_yx_2d(
    shear_yx=dataset.shear_yx,
    output_path=dataset_path,
    output_format="png",
)

"""
__Convergence Map__

The shear field can be inverted directly into a map of the convergence `kappa` (the dimensionless projected
mass density) using the Kaiser-Squires (1993) technique: in Fourier space shear and convergence are related
algebraically, so two FFTs turn the catalogue into a "dark matter map" with no mass model assumed. This is
the classic visualization used for merging clusters (e.g. the Bullet cluster) and survey mass maps.

`aplt.plot_convergence_map` bins the irregular catalogue onto a regular grid, applies a small Gaussian
smoothing (raw per-cell shears are shape-noise dominated) and plots the E-mode reconstruction. For this
Isothermal lens the map peaks at the lens centre at (0.0", 0.0"). Two caveats to remember: the mean of the
map is unconstrained (the mass-sheet degeneracy) and FFT periodicity causes artefacts near the field edges —
for quantitative masses, fit a mass model with `scripts/weak/modeling.py` instead.
"""
aplt.plot_convergence_map(
    shear_yx=dataset.shear_yx,
    shape_native=(30, 30),
    smoothing_sigma_pixels=1.0,
    output_path=dataset_path,
    output_format="png",
)

print(dataset.info)
print(f"Wrote dataset to {dataset_path}")
