"""
Simulator: Lens Light Asymmetric
================================

The morphological of massive elliptical galaxies which act as strong lens are often asymmetric and irregular, with
features such as isophotal twists or radially varying elliptical components.

This script uses a basis of 14 elliptical Gaussians to simulate the light of a massive elliptical galaxy which has
these irregular and asymmetric features. The parameters of the Gaussian basis are derived from a Multi-Gaussian
fit to a real strong lens.

This dataset is used in the `modeling/features/multi_gaussian_expansion.py` script to illustrate how to fit these
features using a Multi-Gaussian Expansion (MGE).

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's light is a superposition of 14 `Gaussian` profiles.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.

The lens galaxy's light is derived from a Multi-Gaussian Expansion (MGE) fit to a massive elliptical galaxy.

The simulated galaxy has irregular and asymmetric features in the galaxy, including a twist in the isophotes of its
emission.

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
dataset_name = "lens_light_asymmetric"
dataset_path = path.join("dataset", dataset_type, dataset_name)

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
Create the simulator for the imaging data, which defines the exposure time, background sky, noise levels and psf.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Ray Tracing__

Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.

The lens galaxy uses 14 elliptical Gaussians, which represent a complex galaxy morphology with irregular and
asymmetric features such as an isophotal twist (which symmetric profiles like a Sersic cannot capture).
"""
centre_y_list = [
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
]

centre_x_list = [
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
]

ell_comps_0_list = [
    0.05843285,
    0.0,
    0.05368621,
    0.05090395,
    0.0,
    0.25367341,
    0.01677313,
    0.03626733,
    0.15887384,
    0.02790297,
    0.12368768,
    0.38624915,
    -0.10490247,
    0.0385585,
]

ell_comps_1_list = [
    0.05932136,
    0.0,
    0.04267542,
    -0.06920487,
    -0.0,
    -0.15141799,
    0.01464508,
    0.03084128,
    -0.17983965,
    0.02215257,
    -0.16271084,
    -0.15945967,
    -0.3969543,
    -0.03808391,
]

intensity_list = [
    0.52107394,
    4.2933716,
    2.40608609,
    4.98902608,
    2.72773562,
    1.10429021,
    1.08190372,
    0.30007753,
    0.6462658,
    0.15766566,
    0.24687923,
    0.04815128,
    0.02559108,
    0.06763223,
]

sigma_list = [
    0.01607907,
    0.04039063,
    0.06734373,
    0.08471335,
    0.16048498,
    0.13531624,
    0.25649938,
    0.46096968,
    0.34492195,
    0.92418119,
    0.71803244,
    1.23547346,
    1.2574071,
    2.69979461,
]

gaussians = []

for gaussian_index in range(len(centre_x_list)):
    gaussian = al.lp.Gaussian(
        centre=(centre_y_list[gaussian_index], centre_x_list[gaussian_index]),
        ell_comps=(
            ell_comps_0_list[gaussian_index],
            ell_comps_1_list[gaussian_index],
        ),
        intensity=intensity_list[gaussian_index],
        sigma=sigma_list[gaussian_index],
    )

    gaussians.append(gaussian)

basis = al.lp_basis.Basis(profile_list=gaussians)

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
)

lens_galaxy = al.Galaxy(redshift=0.5, bulge=basis, mass=mass)

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
The dataset can be viewed in the folder `autolens_workspace/imaging/lens_light_asymmetric`.
"""
