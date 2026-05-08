"""
Simulator: Start Here
=====================

This script is the starting point for simulating point source strong lens datasets, for example a lensed quasar
or supernova, and it provides an overview of the lens simulation API.

After reading this script, the `examples` folder provide examples for simulating more complex lenses in different ways.

__Contents__

- **Model:** Compose the lens model fitted to the data.
- **Dataset Paths:** The `dataset_type` describes the type of data being simulated (in this case, `PointDataset` data).
- **Ray Tracing:** Setup the lens galaxy's mass (SIE) and source galaxy (a point source) for this simulated lens.
- **Point Solver:** For a point source, our goal is to find the (y, x) coordinates in the image plane that map directly.
- **Point Datasets:** All the quantities computed above are stored in a `PointDataset` object, which organizes.
- **Visualize:** Output a subplot of the simulated point source dataset as a .png file.
- **Tracer json:** Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass.
- **Imaging:** Point-source data typically comes with imaging data of the strong lens, for example showing the 4.
- **Fluxes:** Another measurable quantity of a point source is its flux—the total amount of light received from.
- **Point Dataset:** The fluxes are not input a `PointDataset` object, alongside the image-plane coordinates of the.
- **Time Delays:** Another measurable quantity of a point source is its time delay—the time it takes for light to.

__Model__

This script simulates `PointDataset` data of a strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a `Point`.

__Pre-requisites__

It is strongly recommended you read the `autolens_workspace/scripts/point_source/start_here` notebook before
running this script, as it gives a full overview of the point source modeling API and how lensing calculations
are performed.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
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
dataset_type = "point_source"
dataset_name = "simple"

"""
The path where the dataset will be output. 

In this example, this is: `/autolens_workspace/dataset/positions/simple`
"""
dataset_path = Path("dataset") / dataset_type / dataset_name

"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE) and source galaxy (a point source) for this simulated lens. 

We include a faint extended light profile for the source galaxy for visualization purposes, in order to show where 
the multiple images of the lensed source appear in the image-plane.

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

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialCore(
        centre=(0.07, 0.07), intensity=0.1, effective_radius=0.02, radius_break=0.025
    ),
    point_0=al.ps.Point(centre=(0.07, 0.07)),
)

"""
Use these galaxies to setup a tracer, which will compute the multiple image positions of the simulated dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Point Solver__

For a point source, our goal is to find the (y, x) coordinates in the image plane that map directly to the center of 
the point source in the source plane—these are its "multiple images." This is achieved using a `PointSolver`, which 
determines the multiple images of the mass model for a point source located at a given (y, x) position in the 
source plane.

The solver works by ray tracing triangles from the image plane back to the source plane and checking whether the 
source-plane (y, x) center lies inside each triangle. It iteratively refines this process by ray tracing progressively 
smaller triangles, allowing the multiple image positions to be determined with sub-pixel precision.

The `PointSolver` requires an initial grid of (y, x) coordinates in the image plane, which defines the first set of 
triangles to ray trace. It also needs a `pixel_scale_precision` parameter, specifying the resolution at which the 
multiple images are computed. Smaller values increase precision but require longer computation times. The value 
of 0.001 used here balances efficiency and accuracy.

Strong lens mass models often predict a "central image," a multiple image that is usually heavily demagnified and thus 
not observed. Since the `PointSolver` finds all valid multiple images, it will locate this central image regardless of 
its visibility. To avoid including this unobservable image, we set a `magnification_threshold=0.1`, which discards any 
images with magnifications below this value.

If your dataset does include a detectable central image, you should lower this threshold accordingly to include it in 
your analysis.

We now compute the multiple image positions by creating a `PointSolver` object and passing it the tracer of our 
strong lens system.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now pass the tracer to the solver, to determine the image-plane multiple images for the source centre.

The solver will find the image-plane coordinates that map directly to the source-plane coordinate (0.07", 0.07").
"""
positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

"""
We now add Gaussian noise to the multiple image positions to simulate observational measurement errors.

The positional uncertainty in real observations is *not* the pixel scale of the imaging — that is the
detector's sampling, not its centroiding precision. Bright point sources (e.g. lensed quasars or supernovae)
are localised by fitting the instrumental PSF to the image, and the resulting centroid uncertainty is
typically a small fraction of a pixel. For HST/ACS or WFC3, this corresponds to ~3–5 mas (0.003–0.005");
for adaptive optics on Keck or VLT it is similar; for VLBI radio observations of lensed quasars it can
be sub-mas. We adopt a default of 0.005" (5 mas), which is representative of HST point-source astrometry
in the strong-lensing literature (CASTLES, TDCOSMO/H0LiCOW). Setting the precision close to the imaging
pixel scale (~0.05") would inflate lens-model parameter uncertainties well beyond what real data deliver.

Centroid uncertainties from PSF fitting are well-approximated as Gaussian via Laplace's approximation
around the fitted likelihood maximum, so a Gaussian noise model is appropriate here.
"""
position_noise = 0.005

positions_with_noise = positions + np.random.normal(
    loc=0.0, scale=position_noise, size=positions.shape
)

positions_with_noise = al.Grid2DIrregular(
    values=positions_with_noise,
)

"""
__Point Datasets__

All the quantities computed above are stored in a `PointDataset` object, which organizes information about the multiple 
images of a point-source strong lens system.

This dataset is labeled with the `name` `point_0`, identifying it as corresponding to a single point source called 
`point_0`. The name is essential for associating the dataset with the correct point source in the lens model during 
fitting.

The dataset contains the image-plane coordinates of the multiple images and their corresponding noise-map values.
The `positions_noise_map` is set to the same `position_noise` defined above (0.005", i.e. 5 mas), reflecting
realistic PSF-centroiding precision rather than the imaging pixel scale.

Note also that this dataset does not contain fluxes or time delays, which are often included in point source datasets
and are included in a separate simulation below.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions_with_noise,
    positions_noise_map=position_noise,
)

""""
We now output the point dataset to the dataset path as a .json file, which is loaded in the point source modeling
examples.

In this example, there is just one point source dataset. However, for group and cluster strong lenses there
can be many point source datasets in a single dataset, and separate .json files are output for each.
"""
al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset_positions_only.json",
)

"""
__CSV Output__

In addition to JSON, a point dataset can be written to a CSV file.  CSV is a hand-editable,
spreadsheet-friendly format that becomes especially convenient for cluster-scale datasets
with tens of sources where a single file with one row per observed image is easier to curate
than many per-source JSON files.
"""
dataset.to_csv(
    file_path=dataset_path / "point_dataset_positions_only.csv",
)

"""
__Visualize__

Output a subplot of the simulated point source dataset as a .png file.
"""
aplt.subplot_point_dataset(
    dataset=dataset, output_path=dataset_path, output_format="png"
)

"""
Output subplots of the tracer's images, including the positions of the multiple images on the image.
"""

aplt.subplot_tracer(
    tracer=tracer, grid=grid, output_path=dataset_path, output_format="png"
)
aplt.subplot_galaxies_images(
    tracer=tracer, grid=grid, output_path=dataset_path, output_format="png"
)

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

Point-source data typically comes with imaging data of the strong lens, for example showing the 4 multiply
imaged point-sources (e.g. the quasar images).

Whilst this data may not be used for point-source modeling, it is often used to measure the locations of the point
source multiple images in the first place, and is also useful for visually confirming the images we are using are in 
right place. It may also contain emission from the lens galaxy's light, which can be used to perform point-source 
modeling.

We therefore simulate imaging dataset of this point source and output it to the dataset folder in an `imaging` folder
as .fits and .png files. 

If you are not familiar with the imaging simulator API, checkout the `imaging/simulator.py` example 
in the `autolens_workspace`.
"""
psf = al.Convolver.from_gaussian(
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


aplt.subplot_imaging_dataset(dataset=imaging)

aplt.fits_imaging(
    dataset=imaging,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

aplt.subplot_imaging_dataset(dataset=imaging)
aplt.plot_array(array=imaging.data, title="Data")

"""
__Fluxes__

Another measurable quantity of a point source is its flux—the total amount of light received from each multiple image of 
the point source (e.g., the quasar images).

In practice, fluxes are often measured but not used directly when analyzing lensed point sources such as quasars or 
supernovae. This is because fluxes can be significantly affected by microlensing, which many lens models do not 
accurately capture. However, in this simulation, microlensing is not included, so the fluxes can be simulated and fitted reliably.

We now simulate the fluxes of the multiple images of this point source.

Given a mass model and the (y, x) image-plane coordinates of each image, the magnification at each point can be 
calculated.

Below, we compute the magnification for every multiple image coordinate, which will then be used to simulate their 
fluxes.
"""
magnifications = al.LensCalc.from_tracer(
    tracer=tracer
).magnification_2d_via_hessian_from(grid=positions)

"""
To simulate the fluxes, we assume the source galaxy point-source has a total flux of 1.0.

Each observed image has a flux that is the source's flux multiplied by the magnification at that image-plane coordinate.
"""
flux = 1.0
fluxes = [flux * np.abs(magnification) for magnification in magnifications]
fluxes = al.ArrayIrregular(values=fluxes)

"""
We now add Gaussian noise to the fluxes to simulate observational measurement errors.

For lensed quasars and supernovae, photometric measurements of multiple-image fluxes are not generally
photon-noise-limited — the dominant uncertainty is microlensing, in which stars in the lens galaxy
distort the apparent flux of each image by amounts that depend on the source size, the macro-magnification,
and where each image lies in the lens-plane stellar field. Lens models that explicitly *exclude*
microlensing therefore typically assume a few-percent flux uncertainty per image rather than a Poisson
floor. We adopt a 5% relative flux error here, which is consistent with this practice and produces
realistic flux-ratio constraints on the mass model.
"""
flux_rel_noise = 0.05

fluxes_with_noise = fluxes + np.random.normal(
    loc=0.0, scale=flux_rel_noise * np.asarray(fluxes), size=len(fluxes)
)

fluxes_with_noise = al.ArrayIrregular(values=fluxes_with_noise)

fluxes_noise_map = al.ArrayIrregular(values=flux_rel_noise * np.asarray(fluxes))

"""
__Point Dataset__

The fluxes are not input a `PointDataset` object, alongside the image-plane coordinates of the multiple images
and their associated noise-map values. 

We again give the dataset the name `point_0`, which is a label given to the dataset to indicate that it is a dataset 
of a single point-source.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions_with_noise,
    positions_noise_map=position_noise,
    fluxes=fluxes_with_noise,
    fluxes_noise_map=fluxes_noise_map,
)

""""
We now output the point dataset to the dataset path as a .json file, which is loaded in the point source modeling
examples.

In this example, there is just one point source dataset. However, for group and cluster strong lenses there
can be many point source datasets in a single dataset, and separate .json files are output for each.
"""
al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset_with_fluxes.json",
)

"""
__Time Delays__

Another measurable quantity of a point source is its time delay—the time it takes for light to travel from the
source to the observer for each multiple image of the point source (e.g., the quasar images). This is often expressed
as the relative time delay between each image and the image with the shortest time delay, which is often referred to as
the "reference image."

Time delays are commonly used in strong lensing analyses, for example to measure the Hubble constant, since
they are less affected by microlensing and can provide robust cosmological constraints.

We now simulate the same point source dataset, but this time including the time delays of the multiple images.

Given a mass model and (y, x) image-plane coordinates, the time delay at each image-plane position can be
calculated from the mass model. It includes the contribution of both the geometric time delay (the time it takes
different light rays to travel from the source to the observer) and the Shapiro time delay (the time it takes
light to travel through the gravitational potential of the lens galaxy).
"""
time_delays = tracer.time_delays_from(grid=positions)

"""
In real observations, time delays are measured by photometrically monitoring the multiple images over months
to years and cross-correlating their light curves to align the variable signals. State-of-the-art monitoring
campaigns (e.g. COSMOGRAIL, TDCOSMO) routinely achieve ~1–3% relative precision on the longest time delays
in well-sampled quad systems — absolute uncertainties of ~0.5–1.5 days on 30–100 day delays.

For simplicity we adopt a 5% relative uncertainty here. Real-world uncertainties are not strictly
proportional to the delay magnitude (they depend on the cadence and total length of the photometric
monitoring, on microlensing variability that distorts the light curves, and on the lens configuration),
but a constant fractional error is a reasonable simulator default and produces realistic relative weights
between the multiple images.
"""
time_delay_rel_noise = 0.05

time_delays_noise_map = al.ArrayIrregular(values=np.abs(time_delays) * time_delay_rel_noise)

"""
We now add noise to the time delays to simulate observational measurement errors.
"""
time_delays_with_noise = time_delays + np.random.normal(
    loc=0.0, scale=time_delays_noise_map, size=len(time_delays)
)

time_delays_with_noise = al.ArrayIrregular(values=time_delays_with_noise)

"""
__Point Dataset__

The time delays are input into a `PointDataset` object, alongside the image-plane coordinates of the multiple images
and their associated noise-map values. 

We again give the dataset the name `point_0`, which is a label given to the dataset to indicate that it is a dataset 
of a single point-source.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions_with_noise,
    positions_noise_map=position_noise,
    time_delays=time_delays_with_noise,
    time_delays_noise_map=time_delays_noise_map,
)

"""
We now output the point dataset to the dataset path as a .json file, which can be loaded in point source modeling
examples.

While this example contains one point source dataset, group and cluster lenses can contain multiple datasets,
with separate .json files saved for each.
"""
al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset_with_time_delays.json",
)

"""
We output a final point source dataset containing the positions, fluxes and time delays, which could be used
to perform lens modeling of all measurements simultaneously.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions_with_noise,
    positions_noise_map=position_noise,
    fluxes=fluxes_with_noise,
    fluxes_noise_map=fluxes_noise_map,
    time_delays=time_delays_with_noise,
    time_delays_noise_map=time_delays_noise_map,
)

al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset_with_fluxes_and_time_delays.json",
)

"""
__CSV Output__

The full-column dataset (positions, fluxes and time delays) can also be saved to CSV
as the spreadsheet-friendly counterpart to the JSON above.
"""
dataset.to_csv(
    file_path=dataset_path / "point_dataset_with_fluxes_and_time_delays.csv",
)

"""
Finished.
"""
