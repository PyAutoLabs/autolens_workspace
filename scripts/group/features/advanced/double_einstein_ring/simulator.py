"""
Simulator: Group Double Einstein Ring
=====================================

A double Einstein ring lens in a group-scale context, where two source galaxies at different redshifts are lensed
by multiple main lens galaxies at the lens-plane redshift.

This script simulates an `Imaging` dataset of a 'group-scale' strong lens where:

 - There are TWO main lens galaxies at z=0.5, each with a `SersicSph` light profile and an `IsothermalSph` mass
   profile.
 - The first source galaxy at z=1.0 has a `SersicSph` light profile AND an `IsothermalSph` mass profile. This
   source acts as both a light source AND a deflector for the second source.
 - The second source galaxy at z=2.0 has a `SersicSph` light profile only.

The multi-plane ray-tracing accounts for all main lens galaxy masses at the lens redshift, plus the intermediate
source galaxy's mass acting as a secondary lens for the more distant source.

__Contents__

- **Dataset Paths:** The dataset name and output folder.
- **Grid:** Define the 2D image-plane grid.
- **Galaxy Centres:** Centres of the two main lens galaxies, saved as JSON for the modeling scripts to load.
- **Over Sampling:** Adaptive over-sampling at the main lens galaxy centres.
- **Main Lens Galaxies:** Two galaxies at z=0.5.
- **Source Galaxies:** source_0 at z=1.0 (light + mass), source_1 at z=2.0 (light only).
- **Ray Tracing:** Build the multi-plane Tracer.
- **Dataset:** Simulate the imaging dataset and write .fits.
- **Tracer JSON:** Save the simulator Tracer for provenance.
- **Centres JSON:** Save the main lens centres for the modeling scripts.

__Start Here Notebook__

If any code in this script is unclear, refer to:

 - `autolens_workspace/scripts/group/simulator.py` — the canonical group-scale simulator.
 - `autolens_workspace/scripts/imaging/features/advanced/double_einstein_ring/simulator.py` — the single-lens
   double Einstein ring simulator.
"""

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The dataset is written to `dataset/group/double_einstein_ring/`.
"""
dataset_type = "group"
dataset_name = "double_einstein_ring"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

A 200x200 grid at 0.1"/px gives a 20" field of view, large enough to contain a group-scale double Einstein ring
where the two main lens galaxies are separated by ~1.5".
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.1,
)

"""
__Galaxy Centres__

The two main lens galaxies are separated by ~1.5" along the x-axis. These centres are saved as JSON so the
modeling and fit scripts can load them via `al.from_json(...)`.
"""
main_lens_centres = [(0.0, -0.75), (0.0, 0.75)]

"""
__Over Sampling__

Adaptive over-sampling is applied at each main lens galaxy's centre and at the origin.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
Simulate a simple Gaussian PSF.
"""
psf = al.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
)

"""
Imaging simulator: exposure time, background sky, noise, PSF.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Main Lens Galaxies__

Two `SersicSph` + `IsothermalSph` galaxies at z=0.5. Their Einstein radii are chosen so the combined deflection
produces a clearly visible primary Einstein ring around the group's centroid.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=main_lens_centres[0],
        intensity=0.7,
        effective_radius=1.0,
        sersic_index=4.0,
    ),
    mass=al.mp.IsothermalSph(
        centre=main_lens_centres[0],
        einstein_radius=1.2,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=main_lens_centres[1],
        intensity=0.7,
        effective_radius=1.0,
        sersic_index=4.0,
    ),
    mass=al.mp.IsothermalSph(
        centre=main_lens_centres[1],
        einstein_radius=1.2,
    ),
)

main_lens_galaxies = [lens_0, lens_1]

"""
__Source Galaxies__

`source_0` at z=1.0 has BOTH a light profile (its light forms the primary Einstein ring) AND a mass profile
(its mass deflects the light from `source_1` and contributes to the second Einstein ring).

`source_1` at z=2.0 has a light profile only, and is offset so its lensed images form a distinct second ring.
"""
source_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(0.0, 0.0),
        intensity=1.2,
        effective_radius=0.1,
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.25),
)

source_1 = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(-0.3, 0.3),
        intensity=0.6,
        effective_radius=0.07,
    ),
)

"""
__Ray Tracing__

The tracer is composed of the main lens galaxies followed by the two source galaxies. PyAutoLens orders galaxies
internally by redshift, so the multi-plane deflection chain runs:

  image-plane → source_0-plane (deflected by both main lens galaxies)
              → source_1-plane (deflected by both main lens galaxies AND source_0's mass)
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + [source_0, source_1])

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Group DSPL Image")

"""
__Dataset__

Pass the simulator a tracer to produce the simulated `Imaging` dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
Write the simulated dataset to .fits files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Tracer JSON__

Save the simulator `Tracer` so the true profiles can be inspected later.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centres JSON__

Save the main lens centres so the modeling scripts can load them via `al.from_json(...)`. This mirrors the
canonical `group/simulator.py` convention.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)
