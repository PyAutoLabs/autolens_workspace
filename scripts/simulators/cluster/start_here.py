"""
Simulator: Start Here
=====================

This script is the starting point for simulating strong lens cluster datasets, which typically have a Brighest Custer
Galaxy (BCG), large host dark matter halo, 20+ lens galaxies and 5+ source galaxies.

Given how complex these systems are, modeling typically uses the point source API such that only the brightest pixels
in each multiple image in the image plane are fitted. This script therefore outputs both this information and CCD
imaging data of the cluster for visualization. Advanced PyAutoLens tools support extended source modeling of clusters.

After reading this script, the `examples` folder provide examples for simulating more complex lenses in different ways.

__Scaling Relation__

This example uses a scaling-relation lens model using the dual Pseudo-Isothermal Elliptical (dPIE)
mass distribution introduced in Eliasdottir 2007: https://arxiv.org/abs/0710.5636.

It relates the luminosity of every galaxy to a cut radius (r_cut), a core radius (r_core) and a mass normaliaton b0:

$r_cut = r_cut^* (L/L^*)^{0.5}$

$r_core = r_core^* (L/L^*)^{0.5}$

$b0 = b0^* (L/L^*)^{0.25}$

This mass model differs from the `Isothermal` profile used commonly throughout the **PyAutoLens** examples. The dPIE
is more commonly used in strong lens cluster studies where scaling relations are used to model the lensing contribution
of many cluster galaxies.

__Model__

This script simulates `PointDataset` and `Imaging` data of a strong lens where:

 - There are 50 lens galaxies whose mass is a `dPIEMass` profile with parameters set via a scaling relation.
 - There are 3 source galaxies each described as a point source.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import pandas as pd
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `PointDataset` data) and `dataset_name` 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/positions.json`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/noise_map.json`.
"""
dataset_type = "cluster"
dataset_name = "simple"

"""
The path where the dataset will be output. 

In this example, this is: `/autolens_workspace/dataset/cluster/simple`
"""
dataset_path = Path("dataset") / dataset_type / dataset_name

"""
__Grid__

Define the 2d grid of (y,x) coordinates that the lens and source galaxy images are evaluated and therefore simulated 
on, via the inputs:

 - `shape_native`: The (y_pixels, x_pixels) 2D shape of the grid defining the shape of the data that is simulated.
 - `pixel_scales`: The arc-second to pixel conversion factor of the grid and data.

Note how for a cluster lens this spans a large arc second region, from -50.0" to 50.0".
"""
grid = al.Grid2D.uniform(
    shape_native=(400, 400),
    pixel_scales=0.1,
)

"""
__CSV__

For cluster strong lenses, there are often 50 or more galaxies, whose centres, luminosities and other parameters we
need to keep track of.

We use .csv files to store and load the parameter values of all galaxies, noting that .csv files are not used
elsewhere in the workspace where lens models are simpler.

We load a .csv containing the centres and luminosities of the 50 cluster member galaxies as a pandas data frame.  
"""
df = pd.read_csv(dataset_path / "lens_galaxies.csv")

print(df)

"""
__Centres__

Before using the scaling relation, we need to define the centres of the 50 lens galaxies galaxies. 

The 50 values of centres load via a .csv below are drawn from a realisitic strong lens cluster model.
"""
extra_galaxies_centre_list = []

for _, row in df.iterrows():
    centre_0 = row["centre_y"]
    centre_1 = row["centre_x"]

    extra_galaxies_centre_list.append((centre_0, centre_1))

print(extra_galaxies_centre_list)

"""
__Luminosities__

We also need the luminosity of each galaxy, which in this example is the measured property we relate to mass via
the scaling relation.

We again use luminosity values drawn from a realistic galaxy cluster.

This could be other measured properties, like stellar mass or velocity dispersion.
"""
extra_galaxies_luminosity_list = []

for _, row in df.iterrows():
    luminosity = row["luminosity"]

    extra_galaxies_luminosity_list.append(luminosity)

print(extra_galaxies_luminosity_list)

"""
__Scaling Relation__

We now compose our mass parameters using the scaling relation models, which works as follows:

- Compute the parameters for a given mass model using the scaling relations defined at the top of this example
  with the parameters `ra_star`, `rs_star` and `b0_star` .

- For every extra galaxy centre and luminosity, create a mass profile (using `dPIEPotentialSph`), where  the centre 
  of the mass profile is the extra galaxy centres and its other parameters are set via the scaling relation  priors.

- Make each extra galaxy a galaxy (via `Galaxy`) and associate it with th mass profile, where the
  redshifts of the extra galaxies are set to the same values as the lens galaxy.

We also include a `Sersic` lens light component for each galaxy. This is not used in modeling, but it will mean we
output an image where the lens's are visible, making the cluster's lensing configuration easier to inspect.
"""
ra_star = 0.1
rs_star = 2.0
b0_star = 1e2 * 0.0
luminosity_star = 1e9

extra_galaxies_list = []

for extra_galaxy_centre, extra_galaxy_luminosity in zip(
        extra_galaxies_centre_list, extra_galaxies_luminosity_list
):
    bulge = al.lp.SersicSph(
        centre=extra_galaxy_centre,
        intensity=extra_galaxy_luminosity / luminosity_star,
        effective_radius=1.0,
        sersic_index=2.0,
    )

    ra = ra_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    rs = rs_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    b0 = b0_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5

    mass = al.mp.dPIEMassSph(centre=extra_galaxy_centre, ra=ra, rs=rs, b0=b0)

    extra_galaxy = al.Galaxy(redshift=0.5, bulge=bulge, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

"""
__Source Galaxies__

We now create the centres of the 5 source galaxies being lensed by the cluster.

Their image-plane multiple images will be solved for, and these are the observations
which are input into cluster lens modeling examples in order to constrain the lens model.

We also include a `Sersic` lens light component for each galaxy. This is not used in modeling, 
but it will mean we output an image where the source arcS are visible, making the cluster's 
lensing configuration easier to inspect.
"""
source_galaxy_centre_list = [
    (0.01, 0.01),
    #   (1.15, 0.15),
    #   (0.2, 0.2),
    #   (-2.0, -3.0),
    #   (3.0, 2.0),
]

source_galaxy_list = []

for i, source_galaxy_centre in enumerate(source_galaxy_centre_list):
    bulge = al.lp.SersicSph(
        centre=source_galaxy_centre,
        intensity=1.0,
        effective_radius=0.1,
        sersic_index=1.0,
    )

    point = al.ps.Point(centre=source_galaxy_centre)

    source_galaxy = al.Galaxy(redshift=1.0, **{f"point_{i}": point})

    source_galaxy_list.append(source_galaxy)

"""
__Ray Tracing__

Setup the lens galaxies and source galaxies for this simulated cluster lens. 

For lens modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the `convert` module to determine the elliptical components from the axis-ratio and angle.
"""
dark = al.mp.NFWMCRLudlow(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    mass_at_200=1e14,
    redshift_object=0.5,
    redshift_source=1.0,
)

lens = al.Galaxy(redshift=0.5, dark=dark)

tracer = al.Tracer(galaxies=[lens] + extra_galaxies_list + source_galaxy_list)

"""
__Point Solver__

For cluster sources, our goal is to find the (y, x) coordinates in the image plane that map directly to the center of 
the source's centre in the source plane—these are its "multiple images." This is achieved using a `PointSolver`, 
which  determines the multiple images of the mass model for a point source located at a given (y, x) position in the 
source plane.

The solver works by ray tracing triangles from the image plane back to the source plane and checking whether the 
source-plane (y, x) center lies inside each triangle. It iteratively refines this process by ray tracing progressively 
smaller triangles, allowing the multiple image positions to be determined with sub-pixel precision.

The `PointSolver` requires an initial grid of (y, x) coordinates in the image plane (defined above), which defines the 
first set of triangles to ray trace spanning the whole cluster. It also needs a `pixel_scale_precision` parameter, 
specifying the resolution at which the multiple images are computed. Smaller values increase precision but requiring 
longer computation times. The value of 0.001 used here balances efficiency and accuracy.

Strong lens mass models often predict a "central image," a multiple image that is usually heavily demagnified and thus 
not observed. Since the `PointSolver` finds all valid multiple images, it will locate this central image regardless of 
its visibility. To avoid including this unobservable image, we set a `magnification_threshold=0.1`, which discards any 
images with magnifications below this value.

If your dataset does include a detectable central image, you should lower this threshold accordingly to include it in 
your analysis.

We now compute the multiple image positions by creating a `PointSolver` object and passing it the tracer of our 
strong lens system.
"""
solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now pass the tracer to the solver, to determine the image-plane multiple images for the source centres.

The solver finds the image-plane coordinates that map directly to the source-plane coordinates defined above, it is
called iteratively for each source centre with the multiple images stored in a list.
"""
positions_list = []

for i, source_galaxy_centre in enumerate(source_galaxy_centre_list):
    point = getattr(source_galaxy_list[i], f"point_{i}")

    positions = solver.solve(tracer=tracer, source_plane_coordinate=point.centre)

    # remove infinitys from point solver required for JAX

    print(positions)

    mask = jnp.all(jnp.isfinite(positions.array), axis=1)
    positions = positions[mask]

    print(positions)

    positions_list.append(positions)

"""
__Point Datasets__

All the quantities computed above are stored in a `PointDataset` object, which organizes information about the multiple 
images of a point-source strong lens system.

Each dataset is labeled with a `name` (e.g. `point_0`, `point_1`), identifying it as corresponding to a single point 
source called. The name is essential for associating each dataset with the correct point source in the lens model 
for lens modeling.

The dataset contains the image-plane coordinates of the multiple images and their corresponding noise-map values. 
Typically, the noise value for each position is set to the pixel scale of the CCD image, representing the area the 
point source occupies. Although sub-pixel accuracy can be achieved with more detailed analysis, this example does not 
cover those techniques.

Note also that this dataset does not contain fluxes or time delays, which are often included in point source datasets
and are simulated elsewhere in the workspace.
"""
dataset_list = []

for i, positions in enumerate(positions_list):
    dataset = al.PointDataset(
        name=f"point_{i}",
        positions=positions,
        positions_noise_map=grid.pixel_scale,
    )

    dataset_list.append(dataset)

""""
We now output the point dataset to the dataset path as a .json file, which is loaded in the point source modeling
examples.

In this example, there is just one point source dataset. However, for group and cluster strong lenses there
can be many point source datasets in a single dataset, and separate .json files are output for each.
"""
for i, dataset in enumerate(dataset_list):
    al.output_to_json(
        obj=dataset,
        file_path=dataset_path / f"point_dataset_{i}.json",
    )

"""
__Visualize__

Output a subplot of the simulated point source dataset as a .png file.
"""
mat_plot_1d = aplt.MatPlot1D(output=aplt.Output(path=dataset_path, format="png"))
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

point_dataset_plotter = aplt.PointDatasetPlotter(
    dataset=dataset, mat_plot_1d=mat_plot_1d, mat_plot_2d=mat_plot_2d
)
point_dataset_plotter.subplot_dataset()

"""
Output subplots of the tracer's images, including the positions of the multiple images on the image.
"""
visuals = aplt.Visuals2D(multiple_images=positions)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d, visuals_2d=visuals
)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_galaxies_images()

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
__Imaging__

Strong lens clusters typically comes with imaging data, for example showing the foreground lens distribution and
arcs of the sources.

This is used to measure the locations of the point source multiple images in the first place, and is also useful for 
visually confirming the images we are using are in  right place. It may also contain emission from the lens galaxy's 
light, which can be used to perform point-source modeling.

We therefore simulate imaging dataset of this point source and output it to the dataset folder in an `imaging` folder
as .fits and .png files. 

The grid for point sources tracing above had a pixel scale of 0.1", which would make a very low resolution CCD imaging.
We therefore use a higher resolution grid for this image.

If you are not familiar with the imaging simulator API, checkout the `simulators/imaging/start_here.py` example 
in the `autolens_workspace`.
"""
grid = al.Grid2D.uniform(
    shape_native=(1000, 1000), pixel_scales=0.1, over_sample_size=1
)

psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

imaging_path = dataset_path / "imaging"

mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=imaging_path, format="png"))

imaging_plotter = aplt.ImagingPlotter(
    dataset=imaging, mat_plot_2d=mat_plot_2d, visuals_2d=visuals
)
imaging_plotter.subplot_dataset()

imaging.output_to_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

dataset_plotter = aplt.ImagingPlotter(dataset=imaging, mat_plot_2d=mat_plot_2d)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

"""
Finished.
"""
