"""
Simulator: Dark Matter Subhalo
==============================

If a low mass dark matter halo overlaps the lensed source emission, it perturbs it in a unique and observable way.

This script simulates an imaging dataset which includes a dark matter subhalo, which is high enough mass to
detect for Hubble Space Telescope imaging.

This is used in `advanced/subhalo` to illustrate how to fit a lens model which includes a dark matter subhalo.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's light profiles are an `Sersic` and `Exponential`.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The subhalo`s `MassProfile` is a `NFWSph`.
 - The source galaxy's light is an `Sersic`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
"""
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and `dataset_name`
gives it a descriptive name. 
"""
dataset_type = "imaging"
dataset_name = "dark_matter_subhalo"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Simulate__

Simulate the image using a `Grid2D` with the `OverSamplingIterate` object.
"""
grid = al.Grid2D.uniform(
    shape_native=(150, 150),
    pixel_scales=0.05,
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
Create the simulator for the imaging data, which defines the exposure time, background sky, noise levels and psf.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Ray Tracing__

Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.

the `lens_galaxy` below includes a dark matter `subhalo` mass component.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    subhalo=al.mp.NFWTruncatedMCRLudlowSph(centre=(1.601, 0.0), mass_at_200=1.0e10),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
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
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
dataset.output_to_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.

For a faster run time, the tracer visualization uses the binned grid instead of the iterative grid.
"""
mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

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
    file_path=path.join(dataset_path, "tracer.json"),
)

"""
__Subhalo Difference Image__

An informative way to visualize the effect of a subhalo on a strong lens is to subtract the image-plane image of the 
tracer with and without the subhalo included. 

This effectively creates a subhalo residual-map, showing the regions of the image-plane where the subhalo's
effects are located (e.g. near the location of the subhalo).

If this image creates very small residuals (e.g. below the noise level), it means that the subhalo is not detectable 
in the image. Inspecting this image will therefore save you a lot of time, as you will avoid searching for
subhalos that do not produce strong enough effects to be visible in the image!

On the other hand, if the resduals are large, it does not necessarily confirm that the subhalo is detectable. This is
because the subhalo effect may be degenerate with the lens model, whereby the lens mass model or source parameters
can change their parameters to account for the subhalo's effect. Only full end-to-end lens modeling can robustly
confirm whether a subhalo is detectable.
"""
lens_galaxy_no_subhalo = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

tracer_no_subhalo = al.Tracer(galaxies=[lens_galaxy_no_subhalo, source_galaxy])

image = tracer.image_2d_from(grid=grid)
image_no_subhalo = tracer_no_subhalo.image_2d_from(grid=grid)

subhalo_residual_image = image - image_no_subhalo

mat_plot = aplt.MatPlot2D(
    output=aplt.Output(
        path=dataset_path, filename="subhalo_residual_image", format="png"
    )
)

array_plotter = aplt.Array2DPlotter(array=subhalo_residual_image, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__No Lens Light__

The code below simulates the same lens, but without a lens light component.
"""
dataset_name = "dark_matter_subhalo_no_lens_light"
dataset_path = path.join("dataset", dataset_type, dataset_name)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    subhalo=al.mp.NFWTruncatedMCRLudlowSph(centre=(1.601, 0.0), mass_at_200=1.0e10),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

dataset.output_to_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_galaxies_images()

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/dark_matter_subhalo`.
"""
