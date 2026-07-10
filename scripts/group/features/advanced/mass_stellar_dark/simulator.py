"""
Simulator: Group Mass Stellar Dark
==================================

A group-scale strong lens where each main lens galaxy carries a decomposed mass model: a stellar component tied
to the galaxy's own light via a mass-to-light ratio, plus a separately-parameterized dark matter halo.

This script simulates an `Imaging` dataset of a 'group-scale' strong lens where:

 - There are TWO main lens galaxies at z=0.5. Each carries a `lmp.Sersic` bulge (acting as light AND stellar
   mass via `mass_to_light_ratio`) and a spherical `NFWSph` dark matter halo aligned with the bulge centre.
 - The first main lens galaxy additionally carries an `ExternalShear` representing the group-scale shear from
   the wider environment.
 - The source galaxy at z=1.0 has a `SersicCore` light profile.

The total deflection at every image-plane coordinate is the SUM over all main lens galaxies of the per-galaxy
stellar + dark deflections, plus the external shear contribution.

__Contents__

- **Dataset Paths:** The dataset name and output folder.
- **Grid:** Define the 2D image-plane grid.
- **Galaxy Centres:** Centres of the two main lens galaxies, saved as JSON for the modeling scripts to load.
- **Over Sampling:** Adaptive over-sampling at the main lens galaxy centres.
- **Main Lens Galaxies:** Two galaxies at z=0.5, each decomposed into stellar (lmp.Sersic) and dark (NFWSph).
- **Source Galaxy:** A single source at z=1.0 with a SersicCore light profile.
- **Ray Tracing:** Build the Tracer.
- **Dataset:** Simulate the imaging dataset and write .fits.
- **Tracer JSON:** Save the simulator Tracer for provenance.
- **Centres JSON:** Save the main lens centres for the modeling scripts.

__Start Here Notebook__

If any code in this script is unclear, refer to:

 - `autolens_workspace/scripts/group/simulator.py` — the canonical group-scale simulator.
 - `autolens_workspace/scripts/imaging/features/advanced/mass_stellar_dark/simulator.py` — the single-galaxy
   decomposed-mass simulator.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The dataset is written to `dataset/group/mass_stellar_dark/`.
"""
dataset_type = "group"
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

A 200x200 grid at 0.1"/px gives a 20" field of view, large enough to contain a group-scale lens where the two
main lens galaxies are separated by ~1.5".
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.1,
)

"""
__Galaxy Centres__

The two main lens galaxies are separated by ~1.5" along the y-axis. These centres are saved as JSON so the
modeling and fit scripts can load them via `al.from_json(...)`.
"""
main_lens_centres = [(-0.75, 0.0), (0.75, 0.0)]

"""
__Over Sampling__

Adaptive over-sampling is applied at each main lens galaxy's centre, so the stellar mass-to-light coupling is
evaluated accurately at the peak of each bulge.
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
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
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

Two galaxies at z=0.5, each with a `lmp.Sersic` bulge (coupled to its own stellar mass via `mass_to_light_ratio`)
and an `NFWSph` dark matter halo aligned with the bulge.

The first galaxy additionally carries an `ExternalShear` — a single shear field representing the wider group
environment, conventionally attached to `lens_0` (matches the group SLaM convention in
`scripts/group/features/advanced/double_einstein_ring/slam.py`).
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.Sersic(
        centre=main_lens_centres[0],
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
        mass_to_light_ratio=0.2,
    ),
    dark=al.mp.NFWSph(centre=main_lens_centres[0], kappa_s=0.1, scale_radius=20.0),
    shear=al.mp.ExternalShear(gamma_1=-0.02, gamma_2=0.005),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.Sersic(
        centre=main_lens_centres[1],
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        intensity=0.8,
        effective_radius=0.7,
        sersic_index=4.0,
        mass_to_light_ratio=0.25,
    ),
    dark=al.mp.NFWSph(centre=main_lens_centres[1], kappa_s=0.08, scale_radius=20.0),
)

main_lens_galaxies = [lens_0, lens_1]

"""
__Source Galaxy__

A single compact source at z=1.0 with a `SersicCore` light profile, positioned near the group centre so its
lensed image forms a clearly visible Einstein-ring-like configuration around the two main lens galaxies.
"""
source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

"""
__Ray Tracing__

The tracer is composed of the two main lens galaxies followed by the source. PyAutoLens orders galaxies
internally by redshift, so the deflection chain runs:

  image-plane → source-plane (deflected by both main lens galaxies' stellar + dark + shear contributions)
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + [source])

aplt.plot_array(
    array=tracer.image_2d_from(grid=grid), title="Group Mass Stellar Dark Image"
)

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
canonical `group/simulator.py` and `group/features/advanced/double_einstein_ring/simulator.py` conventions.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

"""
Finished.
"""
