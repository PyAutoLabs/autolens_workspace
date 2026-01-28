"""
Pixelization: Source Science
============================

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or galaxies).

Using the reconstructed source pixelization, we can compute key quantities such as the magnification, total flux, and
intrinsic size of the source.

For pixelized source reconstructions, these calculations can be quite involved as they required speciifc code to
handle irregular mesh pixels and other quantities. We illustrate how to perform these calculations below.

However, this does make the source reconstructions different to share with other people, as it would mean they need
to understand how to manipulate irregular meshes. The end of this example shows how a .csv source reconstruction file
is output by a pixelization model-fit, which allows anyone to easy interpolate the source reconstruction on to a uniform grid
for analysis without the need for PyAutoLens.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Model Fit__

The code below is identical to the pixelizaiton `modeling` example, crucially creating a model-fit which
outputs the pixelization source reconstruction to a .csv file.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=4,
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

mesh_shape = (20, 20)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 0

preloads = al.Preloads(
    mapper_indices=al.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    ),
    source_pixel_zeroed_indices=al.util.mesh.rectangular_edge_pixel_list_from(
        total_linear_light_profiles=total_linear_light_profiles,
        shape_native=mesh_shape,
    ),
)

mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    preloads=preloads,
)

inversion = fit.inversion

mapper = inversion.cls_list_from(cls=al.AbstractMapper)[
    0
]  # Extract the mapper from the inversion


"""
We plot the fit, confirming that the pixelized source reconstruction provides a good fit to the data.

Note how the pixelized source reconstruction is performed on an irregular adaptive grid of rectangular pixels,
which is denser in regions of high magnification. This non-uniform distribution of pixels means we need to be care
when performing source science calculations, especially a quantity like the magnification which depends on area.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
All information about the pixelized source reconstruction is contained in the `Inversion` object, which can be
accessed via `fit.inversion`.
"""
inversion = fit.inversion
print(f"Inversion Object: {inversion}")

"""
For example, the reconstructed source pixel flux values are stored in the `reconstruction` attribute of the inversion.
"""
reconstruction = inversion.reconstruction

print(f"Reconstructed Source Pixel Fluxes: {reconstruction}")

total_flux = np.sum(reconstruction)

print(f"Total Source Flux via Pixelization: {total_flux} e- s^-1")

"""
In order to perform source science calculations we need to know which flux value corresponds to which pixel in the 
source-plane.

This information is available in the inversion, below we print the (y,x) centre of each source pixel corresponding to 
the `reconstruction` values printed above.
"""
mapper = inversion.cls_list_from(cls=al.AbstractMapper)[
    0
]  # Extract the mapper from the inversion

source_plane_mesh_grid = mapper.mapper_grids.source_plane_mesh_grid

print(f"Source Plane Mesh Grid Coordinates: {source_plane_mesh_grid}")

"""
The image-plane reconstruction can also be computed from the inversion, which is called the `mapped_reconstructed_image` 
and as seen above is needed to compute the magnification.
"""
mapped_reconstructed_image = inversion.mapped_reconstructed_image

print(f"Mapped Reconstructed Image: {mapped_reconstructed_image}")

"""
__Interpolated Source__

The simplest way to perform source science calculations on a pixelized source reconstruction is to interpolate
its values to a uniform 2D grid of pixels, which can therefore be stored using a `Array2D` object,
which is basically just a 2D numpy array (see the `Data Structure` section at the top of this example).

We interpolate the rectangular pixelized source reconstruction to a new uniform grid we call the `interpolation_grid`.
This calculation can be quite slow, so to make this example run fast we use a relatively small grid, but in practice
you may wish to use a larger grid (e.g. 100x1000 pixels or larger for actual science calculations).
"""
from scipy.interpolate import griddata

interpolation_grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

interpolated_reconstruction = griddata(
    points=source_plane_mesh_grid, values=reconstruction, xi=interpolation_grid
)

# As a pure 2D numpy array in case its useful for calculations
interpolated_reconstruction_ndarray = interpolated_reconstruction.reshape(
    interpolation_grid.shape_native
)

interpolated_reconstruction = al.Array2D.no_mask(
    values=interpolated_reconstruction_ndarray,
    pixel_scales=interpolation_grid.pixel_scales,
)

"""
By printing the interpolated array, we confirm it is a 2D array and can see the pixel values of the source 
reconstruction.

We also plot the interpolated source reconstruction using an `Array2DPlotter`.
"""
print(interpolated_reconstruction.native)

plotter = aplt.Array2DPlotter(
    array=interpolated_reconstruction,
)
plotter.figure_2d()

"""
__Source Flux__

A key quantity for a source galaxy is its total flux, which can be used to compute magnitudes (see 
`autolens_workspace/*/guides/units/flux`) example for more details on this).

The total flux of the source reconstruction can now be computed by summing the interpolated array.

The units of the light profile `intensity` are the units of the data the light profile was fitted to. In this example
we will assume everything is in electrons per second (`e- s^-1`), which is typical for Hubble Space Telescope imaging data.
"""
total_source_flux = np.sum(interpolated_reconstruction)

print(f"Total Source Flux via Interpolated Pixelization: {total_source_flux} e- s^-1")

"""
__Zoom__

The interpolation grid above was large in extent (-3.0" to 3.0" in both the y and x directions), meaning that
the source was a small flux was a small region of this grid.

By changing the `extent` of the interpolation grid, we can performed the interpolation zoomed in on only the
regions of the source-plane where the source reconstruction has non-negligible flux. This
makes the interpolation more accurate, as the interpolation ican use more pixels in the region of interest,
and also makes visualizing the source reconstruction easier.
"""
extent = (-1.0, 1.0, -1.0, 1.0)
shape_native = (401, 401)

interpolation_grid_zoom = al.Grid2D.from_extent(
    extent=extent,
    shape_native=shape_native,
)

interpolated_reconstruction = griddata(
    points=source_plane_mesh_grid, values=reconstruction, xi=interpolation_grid_zoom
)


# As a pure 2D numpy array in case its useful for calculations
interpolated_reconstruction_ndarray = interpolated_reconstruction.reshape(
    interpolation_grid_zoom.shape_native
)

interpolated_reconstruction = al.Array2D.no_mask(
    values=interpolated_reconstruction_ndarray,
    pixel_scales=interpolation_grid_zoom.pixel_scales,
)

"""
__Errors__

The interpolated errors on the source reconstruction can also be computed, which will allow you to perform
model-fitting of the source reconstruction.
"""
reconstruction_noise_map = inversion.reconstruction_noise_map

interpolated_noise_map = griddata(
    points=source_plane_mesh_grid, values=reconstruction, xi=interpolation_grid
)

# As a pure 2D numpy array in case its useful for calculations
interpolated_noise_map_ndarray = interpolated_noise_map.reshape(
    interpolation_grid.shape_native
)

interpolated_noise_map = al.Array2D.no_mask(
    values=interpolated_noise_map_ndarray, pixel_scales=interpolation_grid.pixel_scales
)

plotter = aplt.Array2DPlotter(
    array=interpolated_noise_map,
)
plotter.figure_2d()

"""
__Magnification__

The overall magnification of the source is estimated as the ratio of total surface brightness in the image-plane and 
total surface brightness in the source-plane.

Note that the surface brightness is different to the total flux above, as surface brightness is flux per unit area. 
We therefore explicitly mention how area folds into the calculation below.

The interpolated source reconstruction above has different sized pixels in the image-plane and source-plane, so 
we need to explicitly account for area when computing the magnification.

The `pixel_area` attribute of the `Array2D` object gives us the area of each pixel in arcseconds squared, which we
can use to compute the magnification below.
"""
magnification = np.sum(
    mapped_reconstructed_image * mapped_reconstructed_image.pixel_area
) / np.sum(interpolated_reconstruction * interpolated_reconstruction.pixel_area)

print(f"Magnification via Interpolated Source: {magnification}")

"""
__Masking__

Reconstructions can be imperfect, for example having faint source flux in pixels at the edge of the
source-plane that through comparison to the data are not a genuine part of the source. This can impact
the calculation of the source flux and magnification.

If you want to be extra careful, you can use a mask to zero the source-plane pixels that you do not trust and use
this to remove pixels from source science calculations.

Another approach, which we use below, is we create a source-plane signal-to-noise map and use this to create a mask 
that removes all pixels with a signal-to-noise < 5.0.
"""
signal_to_noise_map = reconstruction / reconstruction_noise_map

mesh_pixel_mask = signal_to_noise_map < 5.0

reconstruction_masked = reconstruction.copy()
reconstruction_masked[mesh_pixel_mask] = 0.0

interpolated_reconstruction_masked = griddata(
    points=source_plane_mesh_grid, values=reconstruction_masked, xi=interpolation_grid
)

# As a pure 2D numpy array in case its useful for calculations
interpolated_reconstruction_masked_ndarray = interpolated_reconstruction_masked.reshape(
    interpolation_grid.shape_native
)

interpolated_reconstruction_masked = al.Array2D.no_mask(
    values=interpolated_reconstruction_masked_ndarray,
    pixel_scales=interpolation_grid.pixel_scales,
)

plotter = aplt.Array2DPlotter(
    array=interpolated_reconstruction_masked,
)
plotter.figure_2d()

"""
__Magnification via Mesh__

The calculations above used an interpolation of the source-plane reconstruction to a 2D grid of 1000 x 1000
pixels.

However, we can use directly the irregular rectangular mesh of the pixelized source reconstruction to compute
quantities. This is more accurate as it does not introduce interpolation errors, but requires more care as the
pixels are irregularly spaced and have different areas. 

We have already computed the total source flux using the mesh above, but we can also compute the magnification.

Computed the areas of every pixel in the irregular rectangular mesh is a bit involved, therefore the values can be
accessed from the source code via the `mesh_areas` attribute of the `Mapper` object.
"""
mesh_areas = mapper.areas_for_magnification

magnification = np.sum(
    mapped_reconstructed_image * mapped_reconstructed_image.pixel_area
) / np.sum(reconstruction * mesh_areas)

"""
__Reconstruction CSV__

In the results `image` folder there is a .csv file called `source_plane_reconstruction_0.csv` which contains the
y and x coordinates of the pixelization mesh, the reconstruct values and the noise map of these values.

This file is provides all information on the source reconstruction in a format that does not depend autolens
and therefore be easily loaded to create images of the source or shared collaborations who do not have PyAutoLens
installed.

We now perform a lens model fit, which will create this .csv file in the modeling output folder.

First, lets load `source_plane_reconstruction_0.csv` as a dictionary, using basic `csv` functionality in Python.
"""
# Lens:

mass = af.Model(al.mp.PowerLaw)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:
mesh = af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape)
regularization = af.Model(al.reg.Constant)

pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Nautilus(
    path_prefix=Path("features"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
    iterations_per_quick_update=50000,
)

positions = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "positions.json"))
)

positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

analysis = al.AnalysisImaging(
    dataset=dataset, positions_likelihood_list=[positions_likelihood], preloads=preloads
)

result = search.fit(model=model, analysis=analysis)

import csv

with open(
    search.paths.image_path / "source_plane_reconstruction_0.csv", mode="r"
) as file:
    reader = csv.reader(file)
    header_list = next(reader)  # ['y', 'x', 'reconstruction', 'noise_map']

    reconstruction_dict = {header: [] for header in header_list}

    for row in reader:
        for key, value in zip(header_list, row):
            reconstruction_dict[key].append(float(value))

    # Convert lists to NumPy arrays
    for key in reconstruction_dict:
        reconstruction_dict[key] = np.array(reconstruction_dict[key])

print(reconstruction_dict["y"])
print(reconstruction_dict["x"])
print(reconstruction_dict["reconstruction"])
print(reconstruction_dict["noise_map"])

"""
You can now use standard libraries to performed calculations with the reconstruction on the mesh, again avoiding
the need to use autolens.

For example, we can create a RectangularAdaptDensity mesh using the scipy.spatial library, which is a triangulation
of the y and x coordinates of the pixelization mesh. This is useful for visualizing the pixelization
and performing calculations on the mesh.
"""
import scipy

points = np.stack(arrays=(reconstruction_dict["x"], reconstruction_dict["y"]), axis=-1)

mesh = scipy.spatial.Delaunay(points)

"""
Interpolating the result to a uniform grid is also possible using the scipy.interpolate library, which means the result
can be turned into a uniform 2D image which can be useful to analyse the source with tools which require an uniform grid.

Below, we interpolate the result onto a 201 x 201 grid of pixels with the extent spanning -1.0" to 1.0", which
capture the majority of the source reconstruction without being too high resolution.
"""
from scipy.interpolate import griddata

values = reconstruction_dict["reconstruction"]

interpolation_grid = al.Grid2D.from_extent(
    extent=(-1.0, 1.0, -1.0, 1.0), shape_native=(201, 201)
)

interpolated_array = griddata(points=points, values=values, xi=interpolation_grid)

"""
Finish.
"""
