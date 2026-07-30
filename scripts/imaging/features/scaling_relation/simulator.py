"""
Simulator: Extra and Scaling Galaxies
=====================================

This script simulates a galaxy-scale strong lens with **two populations of foreground extra galaxies** in front of the
lensed source:

 - Two **individually-modelled** extras close to the lens, each bright enough to warrant its own free Einstein radius
   in the lens model.
 - Two **scaling-relation** extras further out / fainter, whose Einstein radii are tied together via a shared
   luminosity-mass relation in the modeling stage.

Both populations are dumped to separate JSON centre files (`extra_galaxies_centres.json` and
`scaling_galaxies_centres.json`) so the modeling script can load each independently and apply the appropriate
strategy. They both still live under the umbrella of "extra galaxies" in imaging-context terminology.

This dataset is consumed by `scripts/imaging/features/scaling_relation/modeling.py` and its
sibling `fit.py` and `likelihood_function.py` scripts.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__
"""
dataset_type = "imaging"
dataset_name = "extra_and_scaling_galaxies"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

A galaxy-scale field of view: 130x130 pixels at 0.1"/pixel = 13" wide. Big enough to enclose the lens, two close
companions, two further-out companions, and the lensed source; small enough to remain a galaxy-scale tutorial.
"""
grid = al.Grid2D.uniform(
    shape_native=(130, 130),
    pixel_scales=0.1,
)

"""
__Galaxy Centres__

Two centre lists, one per modeling strategy. Both populations are foreground galaxies near the main lens — the split
is purely about how they're modelled downstream.
"""
extra_galaxies_centres = [(3.5, 2.5), (-2.0, -3.5)]
scaling_galaxies_centres = [(5.0, -1.0), (-1.0, 5.0)]

all_galaxy_centres = [(0.0, 0.0)] + extra_galaxies_centres + scaling_galaxies_centres

"""
__Over Sampling__

Adaptive over-sampling at every galaxy centre.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=all_galaxy_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
__PSF + Simulator__
"""
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

"""
__Lens Galaxy__

A standard galaxy-scale primary lens: spherical Sersic light + Isothermal mass at the origin.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=1.5, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6),
)

"""
__Individually-Modelled Extras__

Two close, brighter companions. Each gets its own Sersic light + Isothermal mass with a non-trivial Einstein radius —
in the modeling script the corresponding tier gives each a free `einstein_radius` parameter.
"""
extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.6, sersic_index=2.5
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.4),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-2.0, -3.5), intensity=0.8, effective_radius=0.6, sersic_index=2.5
    ),
    mass=al.mp.IsothermalSph(centre=(-2.0, -3.5), einstein_radius=0.5),
)

individual_extras = [extra_galaxy_0, extra_galaxy_1]

"""
__Scaling-Relation Extras__

Two further-out, fainter companions whose true Einstein radii are consistent with
``einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5`` (luminosities ~0.45 -> Einstein radii ~0.135). In the modeling script
they share two scaling-relation priors regardless of how many are added here.
"""
scaling_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(5.0, -1.0), intensity=0.45, effective_radius=0.5, sersic_index=2.5
    ),
    mass=al.mp.IsothermalSph(centre=(5.0, -1.0), einstein_radius=0.135),
)

scaling_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-1.0, 5.0), intensity=0.45, effective_radius=0.5, sersic_index=2.5
    ),
    mass=al.mp.IsothermalSph(centre=(-1.0, 5.0), einstein_radius=0.135),
)

relational_extras = [scaling_galaxy_0, scaling_galaxy_1]

"""
__Source Galaxy__
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.2,
        sersic_index=1.0,
    ),
)

"""
__Ray Tracing__

Tracer order: lens, individual extras, relational extras, source.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy] + individual_extras + relational_extras + [source_galaxy]
)

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Dataset__
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__
"""
aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

"""
__Tracer json__
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Two JSON files, one per population, matching the names the modeling script loads.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(extra_galaxies_centres),
    file_path=Path(dataset_path, "extra_galaxies_centres.json"),
)

al.output_to_json(
    obj=al.Grid2DIrregular(scaling_galaxies_centres),
    file_path=Path(dataset_path, "scaling_galaxies_centres.json"),
)

"""
__Galaxy Population CSVs__

The modeling script loads luminosities (and centres) for the scaling-relation tier from a CSV
written here. The simulator knows the truth values of the per-galaxy luminosities so we write
them out alongside the centre JSONs.

The CSV schema is `y, x, luminosity, redshift?` -- see `al.galaxy_table_from_csv` /
`al.galaxy_table_to_csv` (`autogalaxy/galaxy/galaxy_table.py`). Centre JSONs above are kept for
backward compatibility; new consumers should prefer the CSV.
"""
extra_galaxies_luminosities = [0.9, 0.8]
scaling_galaxies_luminosities = [0.45, 0.45]

al.galaxy_table_to_csv(
    centres=extra_galaxies_centres,
    luminosities=extra_galaxies_luminosities,
    file_path=Path(dataset_path, "extra_galaxies.csv"),
)

al.galaxy_table_to_csv(
    centres=scaling_galaxies_centres,
    luminosities=scaling_galaxies_luminosities,
    file_path=Path(dataset_path, "scaling_galaxies.csv"),
)
