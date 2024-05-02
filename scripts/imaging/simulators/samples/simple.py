"""
Simulator: Sample
=================

This script illustrates how to simulate a sample of `Imaging` datasets of 'galaxy-scale' strong lenses, which can
easily be used to simulate hundreds or thousands of strong lenses.

To simulate the sample of lenses, each lens and source galaxies set up using the `Model` object which is also used in
the `modeling` scripts. This means that the parameters of each simulated strong lens are drawn from the distributions
defined via priors, which can be customized to simulate a wider range of strong lenses.

This script simulate a sample of `Imaging` datasets of 'galaxy-scale' strong lenses, whose light and mass profiles are
the same as those used in the `start_here` script.

It is used in `autolens_workspace/notebooks/imaging/advanced/hierarchical` to illustrate how a hierarchical model can
be fitted to a large sample of strong lenses in order to infer the glboal properties of the lens sample.

This script uses the signal-to-noise based light profiles described in the
script `imaging/simulators/misc/manual_signal_to_noise_ratio.ipynb`, to make it straight forward to ensure the lens
and source galaxies are visible in each image.

__Model__

This script simulates a sample of `Imaging` data of 'galaxy-scale' strong lenses where:

 - The lens galaxies light profiles are `Sersic`'s.
 - The lens galaxies total mass distributions are `Isothermal` models with `ExternalShear`'s.
 - The source galaxies light profiles are `Sersic`'s.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and `dataset_name`
gives it a descriptive name. 
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "simple"

"""
The path where the dataset will be output, which in this case is:
`/autolens_workspace/dataset/imaging/sample/light_sersic__mass_sie__source_sersic_0`
"""
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_sample_name)

"""
__Simulate__

Simulate the image using a `Grid2D` with the `OverSamplingIterate` object.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
    over_sampling=al.OverSamplingIterate(
        fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
    ),
)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Sample Model Distributions__

To simulate a sample, we draw random instances of lens and source galaxies where the parameters of their light and 
mass profiles are drawn from distributions. These distributions are defined via priors -- the same objects that are used 
when defining the priors of each parameter for a non-linear search.

Below, we define the distributions the lens galaxy's bulge light and mass profiles are drawn from alongside
the soruce's bulge. 
"""
bulge = af.Model(al.lp_snr.Sersic)

bulge.centre = (0.0, 0.0)
bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.signal_to_noise_ratio = af.UniformPrior(lower_limit=20.0, upper_limit=60.0)
bulge.effective_radius = af.UniformPrior(lower_limit=1.0, upper_limit=5.0)
bulge.sersic_index = af.GaussianPrior(
    mean=4.0, sigma=0.5, lower_limit=0.8, upper_limit=5.0
)

mass = af.Model(al.mp.Isothermal)

mass.centre = (0.0, 0.0)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
mass.einstein_radius = af.UniformPrior(lower_limit=0.2, upper_limit=1.8)

shear = af.Model(al.mp.ExternalShear)

shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

bulge = af.Model(al.lp_snr.Sersic)

bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.signal_to_noise_ratio = af.UniformPrior(lower_limit=10.0, upper_limit=30.0)
bulge.effective_radius = af.UniformPrior(lower_limit=0.01, upper_limit=3.0)
bulge.sersic_index = af.GaussianPrior(
    mean=2.0, sigma=0.5, lower_limit=0.8, upper_limit=5.0
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

"""
__Sample Instances__

Within a for loop, we will now generate instances of the lens and source galaxies using the `Model`'s defined above.
This loop will run for `total_datasets` iterations, which sets the number of lenses that are simulated.

Each iteration of the for loop will then create a tracer and use this to simulate the imaging dataset.
"""
total_datasets = 3

for sample_index in range(total_datasets):
    dataset_sample_path = path.join(dataset_path, f"dataset_{sample_index}")

    lens_galaxy = lens.random_instance()
    source_galaxy = source.random_instance()

    """
    __Ray Tracing__
    
    Use the sample's lens and source galaxies to setup a tracer, which will generate the image for the 
    simulated `Imaging` dataset.
    
    The steps below are expanded on in other `imaging/simulator` scripts, so check them out if anything below is unclear.
    """
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Output__
    
    Output the simulated dataset to the dataset path as .fits files.
    
    This uses the updated `dataset_path_sample` which outputs this sample lens to a unique folder.
    """
    dataset.output_to_fits(
        data_path=path.join(dataset_sample_path, "data.fits"),
        psf_path=path.join(dataset_sample_path, "psf.fits"),
        noise_map_path=path.join(dataset_sample_path, "noise_map.fits"),
        overwrite=True,
    )

    """
    __Visualize__
    
    Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.

    For a faster run time, the tracer visualization uses the binned grid instead of the iterative grid.
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
        file_path=path.join(dataset_sample_path, "tracer.json"),
    )

    """
    The dataset can be viewed in the 
    folder `autolens_workspace/imaging/sample/light_sersic__mass_sie__source_sersic_{sample_index]`.
    """
