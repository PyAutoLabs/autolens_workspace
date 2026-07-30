"""
Simulator: Extra Galaxies
=========================

Lensed point sources, for example a lensed quasar or supernova, frequently have small galaxies projected near the
main lens galaxy. Unlike for extended sources, these galaxies do not blend with the source emission — a point
source has no extended surface brightness to contaminate. What they do instead is perturb the deflection field,
shifting the positions (and magnifications) of the multiple images by amounts that are small but easily
measurable, because point-source astrometry is precise to a few milli-arcseconds.

This script simulates a `PointDataset` of a quadruply imaged point source which includes two extra galaxies whose
mass perturbs the multiple image positions. It is used to illustrate the extra galaxies API in the script
`autolens_workspace/*/point_source/features/extra_galaxies/modeling`.

__Contents__

- **Model:** Compose the lens model used to simulate the data.
- **Other Scripts:** This dataset is used in the following scripts.
- **Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a.
- **Ray Tracing:** Setup the lens galaxy's mass and the source galaxy `Point` for this simulated lens.
- **Extra Galaxies:** Two extra galaxies whose mass perturbs the multiple image positions of the point source.
- **Point Solver:** We use a `PointSolver` to locate the multiple images.
- **Fluxes:** The flux of each multiple image, which adds the data points that pay for the extra free parameters.
- **Point Dataset:** Create the point-source dataset and output it to a `.json` file.
- **Extra Galaxies Centres:** Output the centres of the extra galaxies, which set up the model.
- **Imaging:** Simulate the accompanying imaging, which in real data is where the extra galaxy centres come from.
- **Visualize:** Output a subplot of the simulated dataset and the tracer's quantities.
- **Tracer json:** Save the `Tracer` in the dataset folder as a .json file, ensuring the true mass profiles.

__Model__

This script simulates `PointDataset` data of a 'galaxy-scale' strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a `PointFlux` (a point source with a flux).
 - There are two extra galaxies whose mass perturbs the multiple image positions of the source.

The `ExternalShear` is not included in the mass model. As explained in `point_source/modeling.py`, a quadruply
imaged point source provides only 8 positional data points, so an `Isothermal` + `ExternalShear` mass model (9
parameters) is already under-constrained before any extra galaxies are added. Simulating fluxes as well as
positions (below) is what makes room for the extra galaxies in the model.

__Other Scripts__

This dataset is used in the following script:

 `autolens_workspace/*/point_source/features/extra_galaxies/modeling.ipynb`

To illustrate how to compose and fit a point-source lens model which includes the extra galaxies as mass profiles.

__Start Here Notebook__

If any code in this script is unclear, refer to the `point_source/simulator.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.
"""
dataset_type = "point_source"
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / dataset_type / dataset_name

"""
__Ray Tracing__

Setup the lens galaxy's mass and the source galaxy `Point` for this simulated lens.

We include a faint extended light profile for the source galaxy for visualization purposes, in order to show where
the multiple images of the lensed source appear in the image-plane. The lens galaxy is given a light profile for the
same reason — the accompanying imaging simulated at the end of this script is what a real observer would use to
locate the extra galaxy centres.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialCore(
        centre=(0.07, 0.07), intensity=0.1, effective_radius=0.02, radius_break=0.025
    ),
    point_0=al.ps.PointFlux(centre=(0.07, 0.07), flux=1.0),
)

"""
__Extra Galaxies__

Two extra galaxies whose mass perturbs the multiple image positions of the point source.

They are placed just outside the Einstein radius of 1.6", at radii of ~2.2" and ~2.4", and are given
`einstein_radius` values of 0.1" and 0.15" — typical of the faint companions seen projected near real lensed
quasars, and an order of magnitude below the main lens galaxy's.

Small as those deflections are, their effect on the multiple images is not small. Re-solving this system with the
extra galaxies removed moves the four images by 50, 177, 537 and 795 mas respectively. That is one to two orders
of magnitude above the ~5 mas astrometric precision of the data.

The reason is that a multiple image position is the solution of the lens equation, not a linear readout of the
deflection field. Near the Einstein ring the images sit in a nearly-degenerate direction, so adding a 0.1"
deflection does not shift an image by 0.1" — it slides it along the ring until the ray-tracing balances again.
This extreme sensitivity of image positions to perturbing mass is precisely what makes lensed point sources such
a powerful probe of substructure, and it is the reason extra galaxies cannot simply be ignored here in the way a
distant, faint neighbour often can be in extended-source imaging.

The two galaxies are nonetheless far enough from the ring that they do not produce multiple images of their own,
so the system remains a quad.

Note that their redshift is the same as the main lens galaxy, which is not necessarily the case in real
observations. If they are at a different redshift, multi-plane ray-tracing is performed automatically.
"""
extra_galaxy_0_centre = (1.6, 1.5)

extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    light=al.lp.ExponentialSph(
        centre=extra_galaxy_0_centre, intensity=2.0, effective_radius=0.3
    ),
    mass=al.mp.IsothermalSph(centre=extra_galaxy_0_centre, einstein_radius=0.1),
)

extra_galaxy_1_centre = (-2.0, -1.4)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    light=al.lp.ExponentialSph(
        centre=extra_galaxy_1_centre, intensity=2.0, effective_radius=0.4
    ),
    mass=al.mp.IsothermalSph(centre=extra_galaxy_1_centre, einstein_radius=0.15),
)

"""
Use these galaxies to setup a tracer, which will compute the multiple image positions of the simulated dataset.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

"""
__Point Solver__

We use a `PointSolver` to locate the multiple images, by ray tracing triangles from the image plane back to the
source plane and iteratively refining those which contain the source-plane centre. A full description of the
solver, including the `pixel_scale_precision` and `magnification_threshold` settings, is given in
`point_source/simulator.py`.

The solver is passed the tracer above, which includes the two extra galaxies. The positions it returns therefore
already carry their perturbation — this is the signal the modeling script recovers.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

"""
We now add Gaussian noise to the multiple image positions to simulate observational measurement errors.

We use a positional uncertainty of 0.005" (5 mas), which is representative of HST point-source astrometry in the
strong-lensing literature and is discussed in full in `point_source/simulator.py`. It is worth pausing on this
number in the context of extra galaxies: against the 50-795 mas shifts quoted above, the extra galaxies are a
signal detected at high significance, not a marginal systematic. A model which omits them cannot fit these
positions to 5 mas, and will instead distort the main lens galaxy's mass distribution trying to absorb them.
"""
position_noise = 0.005

positions_with_noise = positions + np.random.normal(
    loc=0.0, scale=position_noise, size=positions.shape
)

positions_with_noise = al.Grid2DIrregular(
    values=positions_with_noise,
)

"""
__Fluxes__

The flux of each multiple image, which adds the data points that pay for the extra free parameters.

A quadruply imaged point source gives 8 positional data points. Adding an `einstein_radius` for each extra galaxy
brings the model to 10 free parameters (see `modeling.py`), so positions alone would leave the fit
under-constrained. Simulating fluxes adds 4 more data points, which is what makes the extra-galaxies model
identifiable.

This is a real feature of point-source modeling rather than a convenience of this example. Point-source datasets
are information-poor compared to imaging — a handful of numbers, not tens of thousands of pixels — so every free
parameter added to the model has to be paid for out of a very small budget. It is the reason the extra galaxies
API fixes their centres to the observed light and caps their `einstein_radius`, as `modeling.py` describes.

Fluxes are computed from the magnification at each multiple image position, exactly as in
`point_source/simulator.py`. Note that in real data fluxes are affected by microlensing, which this simulation
does not include.
"""
magnifications = al.LensCalc.from_tracer(
    tracer=tracer
).magnification_2d_via_hessian_from(grid=positions)

flux = 1.0
fluxes = [flux * np.abs(magnification) for magnification in magnifications]
fluxes = al.ArrayIrregular(values=fluxes)

"""
We now add Gaussian noise to the fluxes, adopting the 5% relative flux error motivated in
`point_source/simulator.py`.
"""
flux_rel_noise = 0.05

fluxes_with_noise = fluxes + np.random.normal(
    loc=0.0, scale=flux_rel_noise * np.asarray(fluxes), size=len(fluxes)
)

fluxes_with_noise = al.ArrayIrregular(values=fluxes_with_noise)

fluxes_noise_map = al.ArrayIrregular(values=flux_rel_noise * np.asarray(fluxes))

"""
__Point Dataset__

All the quantities computed above are stored in a `PointDataset` object, which is labeled with the `name`
`point_0`. This name pairs the dataset to the `PointFlux` in the lens model, as described in
`point_source/modeling.py`.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions_with_noise,
    positions_noise_map=position_noise,
    fluxes=fluxes_with_noise,
    fluxes_noise_map=fluxes_noise_map,
)

al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset.json",
)

"""
__Extra Galaxies Centres__

Output the centres of the extra galaxies to a .json file, so that they can be used to set up the model in the
modeling script.

In this simulation we know the centres exactly, because we chose them. For real data they are measured from the
accompanying imaging — see the `Imaging` section below and the data preparation tutorial
`autolens_workspace/*/imaging/data_preparation/examples/optional/extra_galaxies_centres.py`.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(values=[extra_galaxy_0_centre, extra_galaxy_1_centre]),
    file_path=dataset_path / "extra_galaxies_centres.json",
)

"""
__Imaging__

Simulate the accompanying imaging, which in real data is where the extra galaxy centres come from.

Point-source data almost always arrives alongside imaging of the lens: the multiple image positions have to be
measured from an image in the first place. For extra galaxies this imaging is doubly important, because it is the
*only* place the extra galaxy centres can come from. A `PointDataset` is a list of image positions and fluxes — it
contains no information whatsoever about where a faint companion galaxy sits on the sky.

We therefore simulate imaging of this system and output it alongside the point dataset. The two extra galaxies are
visible in it, which is what makes writing down their centres possible.

If you are not familiar with the imaging simulator API, checkout the `imaging/simulator.py` example.
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

imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.fits_imaging(
    dataset=imaging,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset and the tracer's quantities to the dataset path as .png files.
"""
aplt.subplot_point_dataset(
    dataset=dataset, output_path=dataset_path, output_format="png"
)

aplt.subplot_imaging_dataset(dataset=imaging)
aplt.plot_array(array=imaging.data, title="Data")

aplt.subplot_tracer(
    tracer=tracer, grid=grid, output_path=dataset_path, output_format="png"
)
aplt.subplot_galaxies_images(
    tracer=tracer, grid=grid, output_path=dataset_path, output_format="png"
)

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true mass profiles and galaxies are safely
stored and available to check how the dataset was simulated in the future.

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=dataset_path / "tracer.json",
)

"""
The dataset can be viewed in the folder `autolens_workspace/dataset/point_source/extra_galaxies`.
"""
