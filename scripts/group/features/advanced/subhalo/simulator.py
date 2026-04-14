"""
Simulator: Subhalo (Group)
==========================

If a low mass dark matter halo overlaps the lensed source emission, it perturbs it in a unique and observable way.

This script simulates a group-scale strong lens dataset that includes a dark matter subhalo. The subhalo is a
small mass perturber near the lensed images, and its effect on the lensed source can be detected through
careful lens modeling.

__Contents__

**Dataset Paths:** The ``dataset_type`` describes the type of data being simulated.
**Grid:** Define the 2d grid of (y,x) coordinates for the simulation.
**Galaxy Centres:** Define the centres of the main lens galaxies and extra galaxies.
**Over Sampling:** Set up the adaptive over-sampling grid.
**Main Lens Galaxies:** The main lens galaxy, which includes a dark matter subhalo.
**Extra Galaxies:** Two companion galaxies near the lens system.
**Source Galaxy:** The source galaxy whose lensed images we simulate.
**Ray Tracing:** Use all galaxies to set up a tracer.
**Dataset:** Simulate and output the dataset.
**Centre JSON Files:** Save the centres as JSON files.
**Subhalo Difference Image:** Visualize the effect of the subhalo.

__Model__

This script simulates ``Imaging`` of a 'group-scale' strong lens where:

 - The main lens galaxy's light is a ``SersicSph``, total mass is an ``IsothermalSph``, and it includes
   a dark matter subhalo modeled as an ``NFWTruncatedMCRLudlowSph``.
 - The extra galaxies have ``SersicSph`` light and ``IsothermalSph`` mass profiles.
 - A single source galaxy with ``SersicCore`` light.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The dataset is output to ``dataset/group/dark_matter_subhalo``.
"""
dataset_type = "group"
dataset_name = "dark_matter_subhalo"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2d grid of (y,x) coordinates for the simulation.
"""
grid = al.Grid2D.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
)

"""
__Galaxy Centres__

Define the centres of the main lens galaxies and extra galaxies. We also include the subhalo centre
in the over-sampling to ensure accurate evaluation near the subhalo.
"""
main_lens_centres = [(0.0, 0.0)]
extra_galaxies_centres = [(3.5, 2.5), (-4.4, -5.0)]
subhalo_centre = (1.6, 0.0)

"""
__Over Sampling__

Adaptive oversampling at all galaxy centres including the subhalo position.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres + extra_galaxies_centres + [subhalo_centre],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
Create the simulator for the imaging data.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Main Lens Galaxies__

The main lens galaxy includes a dark matter subhalo as an ``NFWTruncatedMCRLudlowSph`` mass component. The
subhalo is positioned near the lensed images at (1.6, 0.0) with a mass of 1e10 solar masses.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
    subhalo=al.mp.NFWTruncatedMCRLudlowSph(centre=(1.6, 0.0), mass_at_200=1.0e10),
)

main_lens_galaxies = [lens_0]

"""
__Extra Galaxies__

Two companion galaxies near the lens system.
"""
extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

extra_galaxies = [extra_galaxy_0, extra_galaxy_1]

"""
__Source Galaxy__

The source galaxy whose lensed images we simulate.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

"""
__Ray Tracing__

Use all galaxies to set up a tracer.
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + extra_galaxies + [source_galaxy])

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Dataset__

Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset.
"""
aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

"""
__Tracer json__

Save the ``Tracer`` in the dataset folder as a .json file.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Save the centres of the main lens galaxies and extra galaxies as JSON files.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

al.output_to_json(
    obj=al.Grid2DIrregular(extra_galaxies_centres),
    file_path=Path(dataset_path, "extra_galaxies_centres.json"),
)

"""
__Subhalo Difference Image__

An informative way to visualize the effect of a subhalo on a strong lens is to subtract the image-plane image of
the tracer with and without the subhalo.

This creates a subhalo residual-map showing the regions of the image plane where the subhalo's effects are
located.
"""
lens_0_no_subhalo = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

tracer_no_subhalo = al.Tracer(
    galaxies=[lens_0_no_subhalo] + extra_galaxies + [source_galaxy]
)

image = tracer.image_2d_from(grid=grid)
image_no_subhalo = tracer_no_subhalo.image_2d_from(grid=grid)

subhalo_residual_image = image - image_no_subhalo

aplt.plot_array(
    array=subhalo_residual_image,
    title="Subhalo Residual Image",
    output_path=dataset_path,
    output_format="png",
)

"""
The dataset can be viewed in the folder ``autolens_workspace/dataset/group/dark_matter_subhalo``.
"""
