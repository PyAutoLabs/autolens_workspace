"""
Simulator: Point Source
=======================

This script simulates `PointDataset` data of a strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source `Galaxy` is a `Point`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. 
"""
dataset_type = "point_source"
dataset_name = "double_einstein_cross"

"""
The path where the dataset will be output.
"""
dataset_path = Path("dataset") / dataset_type / dataset_name

"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE+Shear) and source galaxy `Point` for this simulated lens. We include a 
faint dist in the source for purely visualization purposes to show where the multiple images appear.

For lens modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the `convert` module to determine the elliptical components from the axis-ratio and angle.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    mass=al.mp.Isothermal(
        centre=(0.02, 0.03),
        einstein_radius=0.2,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
    ),
    light=al.lp.ExponentialCore(
        centre=(0.02, 0.03), intensity=0.1, effective_radius=0.02
    ),
    point_0=al.ps.Point(centre=(0.02, 0.03)),
)


source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    light=al.lp.ExponentialCore(
        centre=(0.0, 0.0), intensity=0.1, effective_radius=0.02
    ),
    point_1=al.ps.Point(centre=(0.0, 0.0)),
)

"""
Use these galaxies to setup a tracer, which will compute the multiple image positions of the simulated dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

"""
__Point Solver__

We use a `PointSolver` to locate the multiple images. 
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now pass the `Tracer` to the solver. 

This finds the image-plane coordinates that map directly to the source-plane centres (0.02", 0.03") and (0.0", 0.0").

A double Einstein ring is a multi-plane lensing system, therefore for each source we also input their redshifts into
the solver so that it finds the multiple images properly accounting for the multi-plane lensing.
"""
positions_0 = solver.solve(
    tracer=tracer,
    source_plane_coordinate=source_galaxy_0.point_0.centre,
    plane_redshift=source_galaxy_0.redshift,
)

positions_0_with_noise = positions_0 + np.random.normal(
    loc=0.0, scale=grid.pixel_scale, size=positions_0.shape
)

positions_0_with_noise = al.Grid2DIrregular(
    values=positions_0_with_noise,
)

positions_1 = solver.solve(
    tracer=tracer,
    source_plane_coordinate=source_galaxy_1.point_1.centre,
    plane_redshift=source_galaxy_1.redshift,
)

positions_1_with_noise = positions_1 + np.random.normal(
    loc=0.0, scale=grid.pixel_scale, size=positions_0.shape
)

positions_1_with_noise = al.Grid2DIrregular(
    values=positions_1_with_noise,
)

"""
Use the positions to compute the magnification of the `Tracer` at every position.
"""
magnifications_0 = al.LensCalc.from_tracer(tracer=tracer).magnification_2d_via_hessian_from(grid=positions_0)
magnifications_1 = al.LensCalc.from_tracer(tracer=tracer).magnification_2d_via_hessian_from(grid=positions_1)

"""
We can now compute the observed fluxes of the `Point`, give we know how much each is magnified.
"""
flux = 1.0
fluxes_0 = [flux * np.abs(magnification) for magnification in magnifications_0]
fluxes_0 = al.ArrayIrregular(values=fluxes_0)
fluxes_0_with_noise = fluxes_0 + np.random.normal(
    loc=0.0, scale=np.sqrt(fluxes_0), size=len(fluxes_0)
)
fluxes_0_noise_map = al.ArrayIrregular(
    values=[np.sqrt(flux) for _ in range(len(fluxes_0_with_noise))]
)

fluxes_1 = [flux * np.abs(magnification) for magnification in magnifications_1]
fluxes_1 = al.ArrayIrregular(values=fluxes_1)
fluxes_1_with_noise = fluxes_1 + np.random.normal(
    loc=0.0, scale=np.sqrt(fluxes_1), size=len(fluxes_1)
)
fluxes_1_noise_map = al.ArrayIrregular(
    values=[np.sqrt(flux) for _ in range(len(fluxes_1_with_noise))]
)

"""
We now output the image of this strong lens to `.fits` which can be used for visualize when performing point-source 
modeling and to `.png` for general inspection.
"""

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

aplt.subplot_tracer(tracer=tracer, grid=grid, output_path=dataset_path, output_format="png")
aplt.subplot_galaxies_images(tracer=tracer, grid=grid, output_path=dataset_path, output_format="png")

"""
__Point Datasets__

Create a point-source data object and output this to a `.json` file, which is the format used to load and
analyse the dataset.
"""
dataset_0 = al.PointDataset(
    name="point_0",
    positions=positions_0_with_noise,
    positions_noise_map=grid.pixel_scale,
    fluxes=fluxes_0_with_noise,
    fluxes_noise_map=fluxes_0_noise_map,
)
dataset_1 = al.PointDataset(
    name="point_1",
    positions=positions_1,
    positions_noise_map=grid.pixel_scale,
    fluxes=fluxes_1_with_noise,
    fluxes_noise_map=fluxes_1_noise_map,
)


""""
We now output the point datasets to the dataset path as a .json file, which is loaded in the point source modeling
examples.
"""
al.output_to_json(
    obj=dataset_0,
    file_path=dataset_path / "point_dataset_0.json",
)

al.output_to_json(
    obj=dataset_1,
    file_path=dataset_path / "point_dataset_1.json",
)

"""
__Visualize__

Output a subplot of the simulated point source dictionary and the tracer's quantities to the dataset path as .png files.
"""
aplt.subplot_point_dataset(
    dataset=dataset_0, output_path=dataset_path, output_format="png"
)
aplt.subplot_point_dataset(
    dataset=dataset_1, output_path=dataset_path, output_format="png"
)

aplt.subplot_tracer(tracer=tracer, grid=grid, output_path=dataset_path, output_format="png")
aplt.subplot_galaxies_images(tracer=tracer, grid=grid, output_path=dataset_path, output_format="png")

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=dataset_path / "tracer.json",
)

"""
Finished.
"""
