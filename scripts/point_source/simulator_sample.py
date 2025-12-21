"""
Simulator: Hubble Constant Time Delays
======================================

A multiply imaged lensed quasar with time delays can measure the Hubble constant, which is a fundamental
Cosmological parameter that describes the rate of expansion of the universe. This is because the
the difference between the geometric time delay and the physical time delay is proportional to the Hubble constant.

This script illustrates how to simulate a sample of `PointDataset` datasets of lensed quasars, which
can easily be used to simulate hundreds or thousands of strong lenses. These, as a sample, can be used to constrain
the Hubble constant.

To simulate the sample of lenses, each lens and source galaxies are set up using the `Model` object which is also used
in  the `modeling` scripts. This means that the parameters of each simulated strong lens are drawn from the
distributions  defined via priors, which can be customized to simulate a wider range of strong lenses.

The sample is used in `autolens_workspace/notebooks/advanced/graphical` to illustrate how a graphical and hierarchical
model can be fitted to a large sample of double Einstein ring strong lenses in order to improve the constraints on
Cosmological parameters.

__Model__

This script simulates a sample of `PointDataset` data of 'galaxy-scale' strong lenses where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a `Point`.
 - The Cosmology is `Planck15` and has a Hubble constant which can be constrained by the time delays.

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
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. 
"""
dataset_label = "samples"
dataset_type = "point_source"
dataset_sample_name = "hubble_constant_time_delays"

"""
The path where the dataset will be output.
"""
dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_sample_name

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
__Sample Model Distributions__

To simulate a sample, we draw random instances of lens and source galaxies where the parameters of their mass profiles 
and point source profiles are drawn from distributions. These distributions are defined via priors -- the same objects 
that are used when defining the priors of each parameter for a non-linear search.

Below, we define the distributions the lens galaxy's mass profiles are drawn from alongside the source's point
source centre.
"""
mass = af.Model(al.mp.Isothermal)

mass.centre = (0.0, 0.0)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=1.8)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

point = af.Model(al.ps.Point)

point.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
point.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

"""
We also include a light profile component of the source, which is to aid visualization of the simulated dataset
by providing an image where the point source is located.
"""
bulge = af.Model(al.lp.ExponentialSph)

bulge.centre_0 = point.centre_0
bulge.centre_1 = point.centre_1
bulge.intensity = 1.0
bulge.effective_radius = 0.02
bulge.signal_to_noise_ratio = 10.0

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge, point=point)

"""
__Simulate__

Simulate the image data using a (y,x) grid with the adaptive over sampling scheme.
"""
grid = al.Grid2D.uniform(
    shape_native=(150, 150),
    pixel_scales=0.1,
)

psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.2, pixel_scales=grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Sample Instances__

Within a for loop, we will now generate instances of the lens and source galaxies using the `Model`'s defined above.
This loop will run for `total_datasets` iterations, which sets the number of lenses that are simulated.

Each iteration of the for loop will then create a tracer and use this to simulate the point dataset.
"""
total_datasets = 3

for sample_index in range(total_datasets):
    dataset_sample_path = dataset_path / f"dataset_{sample_index}"

    lens_galaxy = lens.random_instance()
    source_galaxy = source.random_instance()

    """
    __Ray Tracing__

    Use the sample's lens and source galaxies to setup a tracer, which will generate the multiple image positions 
    and time delays for the simulated `PointDataset` dataset.
    """
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

    """
    __Positions__

    We now pass the `Tracer` to the solver to find the multiple image positions.

    """
    positions = solver.solve(
        tracer=tracer, source_plane_coordinate=source_galaxy.point.centre
    )
    positions_noise_map = grid.pixel_scale

    """
    __Time Delays__

    We next compute the time delays of the multiple images, which are used to constrain the Hubble constant.
    """
    time_delays = tracer.time_delays_from(grid=positions)
    time_delays_noise_map = al.ArrayIrregular(values=time_delays * 0.25)

    """
    __Point Dataset__

    We now output the `PointDataset` dataset, which contains the multiple image positions, their noise levels,
    the time delays and their noise levels.

    We output this to a .json file which can be loaded in point source modeling examples.
    """
    dataset = al.PointDataset(
        name="point_0",
        positions=positions,
        positions_noise_map=grid.pixel_scale,
        time_delays=time_delays,
        time_delays_noise_map=time_delays_noise_map,
    )

    al.output_to_json(
        obj=dataset,
        file_path=dataset_sample_path / "point_dataset_with_time_delays.json",
    )

    """
    __Imaging__

    We also generate an `Imaging` dataset of the lens, which is used to visualize the lens and source galaxies
    and the multiple image positions.
    """
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Output__

    Output the simulated dataset to the dataset path as .fits files.

    This uses the updated `dataset_path_sample` which outputs this sample lens to a unique folder.
    """
    dataset.output_to_fits(
        data_path=dataset_sample_path / "data.fits",
        psf_path=dataset_sample_path / "psf.fits",
        noise_map_path=dataset_sample_path / "noise_map.fits",
        overwrite=True,
    )

    """
    __Visualize__

    Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.
    """
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_sample_path, format="png")
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    dataset_plotter.subplot_dataset()
    dataset_plotter.figures_2d(data=True)

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
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
        file_path=dataset_sample_path / "tracer.json",
    )

    """
    The dataset can be viewed in the 
    folder `autolens_workspace/dataset/point/samples/hubble_constant_time_delays/{sample_index]`.
    """
