"""
Simulator: Combined Strong + Weak Lensing
=========================================

This script simulates the two faces of the same gravitational lens: an `Imaging` dataset of its strongly
lensed arcs, and a `WeakDataset` of the weak shear it imprints on background galaxies at larger radii.
Both are generated from **one** `Tracer`, so the datasets share a single true mass distribution — the setup
the fit and modeling examples in this folder use to demonstrate joint strong+weak constraints.

__Scientific Context__

This combination is how weak lensing is used around strong-lens galaxy clusters and groups (it is *not* the
cosmic-shear or galaxy-galaxy-lensing regime). Strong lensing constrains the mass distribution superbly, but
only inside the Einstein radius where arcs and multiple images form; weak shear extends the constraint to
several times that radius, where most of the halo's mass lives. Joint analyses of this kind include
hybrid-Lenstool's simultaneous strong+weak cluster reconstructions (Niemiec et al. 2020) and the combined
strong and weak lensing analysis of 28 group-to-cluster scale lenses in the Sloan Giant Arcs Survey
(Oguri et al. 2012).

The scales here are galaxy/group-like and kept small so the examples run quickly: arcs at the Einstein
radius of 1.6", and a shear catalogue extending to 10" — about six Einstein radii, far beyond the 3.0"
region the imaging data constrains.

__Contents__

- **Dataset Paths:** Both datasets are output to a single `dataset/weak/strong_lensing/` folder.
- **Ray Tracing:** The single Tracer both datasets are simulated from.
- **Imaging Simulation:** The strong-lensing image of the lensed source (no lens light, for simplicity).
- **Weak Simulation:** The surrounding shear catalogue, with space-like shape noise.
- **Output:** .fits (imaging), .json (weak catalogue + tracer) and .png visualizations.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

Both datasets describe the same lens, so they live together in one folder:

 - The imaging data will be output to `data.fits` / `noise_map.fits` / `psf.fits`.
 - The shear catalogue will be output to `dataset.json`, and the shared truth to `tracer.json`.
"""
dataset_type = "weak"
dataset_name = "strong_lensing"

dataset_path = Path("dataset") / dataset_type / dataset_name

"""
__Ray Tracing__

The lens is an elliptical `Isothermal` mass distribution (Einstein radius 1.6", axis-ratio 0.8 at 45 degrees)
with no light profile — omitting lens light keeps the imaging side of the example simple, exactly as in the
`imaging/features/no_lens_light` example. The source is a compact cored-Sersic.

The ellipticity is deliberately pronounced: the outer quadrupole of the mass distribution is the quantity the
weak shear constrains best, so it is where the joint fit visibly improves on imaging alone.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.05, 0.05),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Imaging Simulation__

The strong-lensing image is simulated exactly as in `scripts/imaging/simulator.py`: a 100 x 100 grid at
0.1"/pixel (a 10" field of view whose central ~3" contains the arcs), a Gaussian PSF, and Poisson + sky noise.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

psf = al.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

dataset_imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
__Weak Simulation__

The weak shear catalogue is simulated from the *same tracer* at 400 uniform-random background-galaxy
positions inside a 10" half-width square. At the Einstein radius the shear is of order unity, but by 10" it
has fallen to |gamma| ~ 0.08 — each individual galaxy is a noisy probe, and the signal lives in their
ensemble.

A shape noise of `noise_sigma = 0.1` per component corresponds to deep space-based imaging (ground-based
surveys are nearer 0.3); it gives this small catalogue a total detection significance high enough for the
example's joint fit to visibly tighten the mass model.
"""
simulator_weak = al.SimulatorShearYX(noise_sigma=0.1, seed=1)

dataset_weak = simulator_weak.via_tracer_random_positions_from(
    tracer=tracer,
    n_galaxies=400,
    grid_extent=10.0,
    name=dataset_name,
)

"""
__Output__

The imaging dataset is output as .fits (the standard astronomical format), the weak catalogue and the shared
true tracer as .json.
"""
dataset_path.mkdir(parents=True, exist_ok=True)

aplt.fits_imaging(
    dataset=dataset_imaging,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

al.output_to_json(obj=dataset_weak, file_path=dataset_path / "dataset.json")
al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")

"""
__Visualize__

The two datasets side by side make the complementarity obvious: all the imaging information sits inside the
central few arc-seconds, while the shear catalogue's quivers cover a field over six Einstein radii across.
The Kaiser-Squires convergence map of the shear field peaks on the strong lens — the same mass seen two ways.
"""
aplt.subplot_imaging_dataset(
    dataset=dataset_imaging, output_path=dataset_path, output_format="png"
)

aplt.subplot_weak_dataset(
    dataset=dataset_weak, output_path=dataset_path, output_format="png"
)

aplt.plot_convergence_map(
    shear_yx=dataset_weak.shear_yx,
    shape_native=(30, 30),
    smoothing_sigma_pixels=1.0,
    output_path=dataset_path,
    output_format="png",
)

print(dataset_weak.info)
print(f"Wrote combined strong+weak dataset to {dataset_path}")
