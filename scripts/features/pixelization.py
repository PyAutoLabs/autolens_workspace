"""
Modeling Features: Pixelization
===============================

A pixelization reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
the solution to have a degree of smoothness.

This script fits a source galaxy model which uses a pixelization to reconstruct the source's light. 

A Voronoi mesh and constant regularization scheme are used, which are the simplest forms of mesh and regularization
with provide computationally fast and accurate solutions in **PyAutoLens**.

For simplicity, the lens galaxy's light is omitted from the model and is not present in the simulated data. It is
straightforward to include the lens galaxy's light in the model.

Pixelizations are covered in detail in chapter 4 of the **HowToLens** lectures.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the light profile intensities is ensured.
**Chaining:** How the advanced modeling feature, non-linear search chaining, can significantly improve lens modeling with pixelizaitons.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Pixelization:** How to create a pixelization, including a description of its inputs.
**Fit:** Perform a fit to a dataset using a pixelization, and visualize its results.
**Model:** Composing a model using a pixelization and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Positions Likelihood:** Removing unphysical pixelized source solutions using a likelihood penalty using the lensed multiple images.
**Run Time:** Profiling of pixelization run times and discussion of how they compare to standard light profiles.
**Model-Fit:** Performs the model fit using standard API.
**Result:** Pixelization results and visualizaiton.
**Interpolated Source:** Interpolate the source reconstruction from an irregular Voronoi mesh to a uniform square grid and output to a .fits file.
**Voronoi:** Using a Voronoi mesh pixelizaiton (instead of Voronoi), which provides better results but requires installing an external library.
**Result (Advanced):** API for various pixelization outputs (magnifications, mappings) which requires some polishing.

__Advantages__

Many strongly lensed source galaxies are complex, and have asymmetric and irregular morphologies. These morphologies
cannot be well approximated by a parametric light profiles like a Sersic, or many Sersics, and thus a pixelization
is required to reconstruct the source's irregular light.

Even basis functions like shapelets or a multi-Gaussian expansion cannot reconstruct a source-plane accurately
if there are multiple source galaxies, or if the source galaxy has a very complex morphology.

To infer detailed components of a lens mass model (e.g. its density slope, whether there's a dark matter subhalo, etc.)
then pixelized source models are required, to ensure the mass model is fitting all of the lensed source light.

There are also many science cases where one wants to study the highly magnified light of the source galaxy in detail,
to learnt about distant and faint galaxies. A pixelization reconstructs the source's unlensed emission and thus
enables this.

__Disadvantages__

Pixelizations are computationally slow and run times are typically longer than a parametric source model. It is not
uncommon for lens models using a pixelization to take hours or even days to fit high resolution imaging
data (e.g. Hubble Space Telescope imaging).

Lens modeling with pixelizations is also more complex than parametric source models, with there being more things
that can go wrong. For example, there are solutions where a demagnified version of the lensed source galaxy is
reconstructed, using a mass model which effectively has no mass or too much mass. These are described in detail below.

It will take you longer to learn how to successfully fit lens models with a pixelization than other methods illustrated
in the workspace!

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For a pixelizaiton, this often produces negative source pixels which over-fit
the data, producing unphysical solutions.

All pixelized source reconstructions use a positive-only solver, meaning that every source-pixel is only allowed
to reconstruct positive flux values. This ensures that the source reconstruction is physical and that we don't
reconstruct negative flux values that don't exist in the real source galaxy (a common systematic solution in lens
analysis).

It may be surprising to hear that this is a feature worth pointing out, but it turns out setting up the linear algebra
to enforce positive reconstructions is difficult to make efficient. A lot of development time went into making this
possible, where a bespoke fast non-negative linear solver was developed to achieve this.

Other methods in the literature often do not use a positive only solver, and therefore suffer from these
unphysical solutions, which can degrade the results of lens model in general.

__Chaining__

Due to the complexity of fitting with a pixelization, it is often best to use **PyAutoLens**'s non-linear chaining
feature to compose a pipeline which begins by fitting a simpler model using a parametric source.

More information on chaining is provided in the `autolens_workspace/notebooks/imaging/advanced/chaining` folder,
chapter 3 of the **HowToLens** lectures.

The script `autolens_workspace/scripts/imaging/advanced/chaining/parametric_to_pixelization.py` explitly uses chaining
to link a lens model using a light profile source to one which then uses a pixelization.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is reconstructed using a `Voronoi` mesh, `Overlay` image-mesh
   and `ConstantSplit` regularization scheme.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
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
__Dataset__

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Pixelization__

We create a `Pixelization` object to perform the pixelized source reconstruction, which is made up of three
components:

- `image_mesh:`The coordinates of the mesh used for the pixelization need to be defined. The way this is performed
depends on pixelization used. In this example, we define the source pixel centers by overlaying a uniform regular grid
in the image-plane and ray-tracing these coordinates to the source-plane. Where they land then make up the coordinates
used by the mesh.

- `mesh:` Different types of mesh can be used to perform the source reconstruction, where the mesh changes the
details of how the source is reconstructed (e.g. interpolation weights). In this exmaple, we use a `Voronoi` mesh,
where the centres computed via the `image_mesh` are the vertexes of every `Voronoi` triangle.

- `regularization:` A pixelization uses many pixels to reconstructed the source, which will often lead to over fitting
of the noise in the data and an unrealistically complex and strucutred source. Regularization smooths the source
reconstruction solution by penalizing solutions where neighboring pixels (Voronoi triangles in this example) have
large flux differences.
"""
image_mesh = al.image_mesh.Overlay(shape=(30, 30))
mesh = al.mesh.Delaunay()
regularization = al.reg.ConstantSplit(coefficient=1.0)

pixelization = al.Pixelization(
    image_mesh=image_mesh, mesh=mesh, regularization=regularization
)

"""
__Fit__

This is to illustrate the API for performing a fit via a pixelization using standard autolens objects like 
the `Galaxy`, `Tracer` and `FitImaging` 

We simply create a `Pixelization` and pass it to the source galaxy, which then gets input into the tracer.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens, source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
By plotting the fit, we see that the pixelized source does a good job at capturing the appearance of the source galaxy
and fitting the data to roughly the noise level.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
Pixelizations have bespoke visualizations which show more details about the source-reconstruction, image-mesh
and other quantities.

These plots use an `InversionPlotter`, which gets its name from the internals of how pixelizations are performed in
the source code, where the linear algebra process which computes the source pixel fluxes is called an inversion.

The `subplot_mappings` overlays colored circles in the image and source planes that map to one another, thereby
allowing one to assess how the mass model ray-traces image-pixels and therefore to assess how the source reconstruction
maps to the image.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.subplot_of_mapper(mapper_index=0)
inversion_plotter.subplot_mappings(pixelization_index=0)

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data.  In this 
example fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source-galaxy's light uses a `Voronoi` mesh [0 parameters].
 
 - The mesh centres of the `Voronoi` mesh are computed using a `Overlay` image-mesh, with a fixed resolution of 
   30 x 30 pixels [0 parameters].
 
 - This pixelization is regularized using a `ConstantSplit` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6. 
 
It is worth noting the `Pixelization`  use significantly fewer parameters (1 parameter) than 
fitting the source using `LightProfile`'s (7+ parameters). 

The lens model therefore includes a mesh and regularization scheme, which are used together to create the 
pixelization. 

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

mass = af.Model(al.mp.PowerLaw)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

image_mesh = af.Model(al.image_mesh.Overlay)
image_mesh.shape = (30, 30)

mesh = af.Model(al.mesh.Delaunay)

regularization = af.Model(al.reg.ConstantSplit)

pixelization = af.Model(
    al.Pixelization, image_mesh=image_mesh, mesh=mesh, regularization=regularization
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the source galaxy's has a mesh and regularization scheme, which are combined into a pixelization.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=4,
)

"""
__Position Likelihood__

We add a penalty term ot the likelihood function, which penalizes models where the brightest multiple images of
the lensed source galaxy do not trace close to one another in the source plane. This removes "demagnified source
solutions" from the source pixelization, which one is likely to infer without this penalty.

A comprehensive description of why we do this is given at the following readthedocs page. I strongly recommend you 
read this page in full if you are not familiar with the positions likelihood penalty and demagnified source 
reconstructions:

 https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html

__Brief Description__

Unlike other example scripts, we also pass the `AnalysisImaging` object below a `PositionsLHPenalty` object, which
includes the positions we loaded above, alongside a `threshold`.

This is because `Inversion`'s suffer a bias whereby they fit unphysical lens models where the source galaxy is 
reconstructed as a demagnified version of the lensed source. 

To prevent these solutions biasing the model-fit we specify a `position_threshold` of 0.5", which requires that a 
mass model traces the four (y,x) coordinates specified by our positions (that correspond to the brightest regions of the 
lensed source) within 0.5" of one another in the source-plane. If this criteria is not met, a large penalty term is
added to likelihood that massively reduces the overall likelihood. This penalty is larger if the ``positions``
trace further from one another.

This ensures the unphysical solutions that bias a pixelization have a lower likelihood that the physical solutions
we desire. Furthermore, the penalty term reduces as the image-plane multiple image positions trace closer in the 
source-plane, ensuring Nautilus converges towards an accurate mass model. It does this very fast, as 
ray-tracing just a few multiple image positions is computationally cheap. 

The threshold of 0.3" is large. For an accurate lens model we would anticipate the positions trace within < 0.01" of
one another. The high threshold ensures only the initial mass models at the start of the fit are resampled.

Position thresholding is described in more detail in the 
script `autolens_workspace/*/modeling/imaging/customize/positions.py`

The arc-second positions of the multiply imaged lensed source galaxy were drawn onto the
image via the GUI described in the file `autolens_workspace/*/data_preparation/imaging/gui/positions.py`.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)
positions_likelihood = al.PositionsLHPenalty(positions=positions, threshold=0.3)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data. 
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=positions_likelihood,
)

"""
__Run Time__

The run time of a pixelization is longer than many other features, with the estimate below coming out at around ~0.5 
seconds per likelihood evaluation. This is because the fit has a lot of linear algebra to perform in order to
reconstruct the source on the pixel-grid.

Nevertheless, this is still fast enough for most use-cases. If run-time is an issue, the following factors determine
the run-time of a a pixelization and can be changed to speed it up (at the expense of accuracy):

 - The number of unmasked pixels in the image data. By making the mask smaller (e.g. using an annular mask), the 
   run-time will decrease.
 
 - The number of source pixels in the pixelization. By reducing the `shape` from (30, 30) the run-time will decrease.

This also serves to highlight why the positions threshold likelihood is so powerful. The likelihood evaluation time
of this step is below 0.001 seconds, meaning that the initial parameter space sampling is extremely efficient even
for a pixelization (this is not accounted for in the run-time estimate below)!
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format (if this 
does not display clearly on your screen refer to `start_here.ipynb` for a description of how to fix this):

This confirms that the source galaxy's has a mesh and regularization scheme, which are combined into a pixelization.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

The end of this example provides a detailed description of all result options for a pixelization.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grids.uniform
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
__Mask Extra Galaxies__

There may be extra galaxies nearby the lens and source galaxies, whose emission blends with the lens and source.

If their emission is significant, and close enough to the lens and source, we may simply remove it from the data
to ensure it does not impact the model-fit. A standard masking approach would be to remove the image pixels containing
the emission of these galaxies altogether. This is analogous to what the circular masks used throughout the examples 
does.

For fits using a pixelization, masking regions of the image in a way that removes their image pixels entirely from 
the fit. This can produce discontinuities in the pixelixation used to reconstruct the source and produce unexpected 
systematics and unsatisfactory results. In this case, applying the mask in a way where the image pixels are not 
removed from the fit, but their data and noise-map values are scaled such that they contribute negligibly to the fit, 
is a better approach. 

We illustrate the API for doing this below, using the `extra_galaxies` dataset which has extra galaxies whose emission
needs to be removed via scaling in this way. We apply the scaling and show the subplot imaging where the extra 
galaxies mask has scaled the data values to zeros, increasing the noise-map values to large values and in turn made 
the signal to noise of its pixels effectively zero.
"""
dataset_name = "extra_galaxies"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=path.join(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=0.1,
    invert=True,  # Note that we invert the mask here as `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=0.1, centre=(0.0, 0.0), radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
We do not explictly fit this data, for the sake of brevity, however if your data has these nearby galaxies you should
apply the mask as above before fitting the data.

__Pixelization / Mapper Calculations__

The pixelized source reconstruction output by an `Inversion` is often on an irregular grid (e.g. a 
Voronoi triangulation or Voronoi mesh), making it difficult to manipulate and inspect after the lens modeling has 
completed.

Internally, the inversion stores a `Mapper` object to perform these calculations, which effectively maps pixels
between the image-plane and source-plane. 

After an inversion is complete, it has computed values which can be paired with the `Mapper` to perform calculations,
most notably the `reconstruction`, which is the reconstructed source pixel values.

By inputting the inversions's mapper and a set of values (e.g. the `reconstruction`) into a `MapperValued` object, we
are provided with all the functionality we need to perform calculations on the source reconstruction.

We set up the `MapperValued` object below, and illustrate how we can use it to interpolate the source reconstruction
to a uniform grid of values, perform magnification calculations and other tasks.
"""
inversion = result.max_log_likelihood_fit.inversion
mapper = inversion.cls_list_from(cls=al.AbstractMapper)[
    0
]  # Only one source-plane so only one mapper, would be a list if multiple source planes

mapper_valued = al.MapperValued(
    mapper=mapper, values=inversion.reconstruction_dict[mapper]
)

"""
__Interpolated Source__

A simple way to inspect the source reconstruction is to interpolate its values from the irregular
pixelization o a uniform 2D grid of pixels.

(if you do not know what the `slim` and `native` properties below refer too, it 
is described in the `results/examples/data_structures.py` example.)

We interpolate the Voronoi triangulation this source is reconstructed on to a 2D grid of 401 x 401 square pixels. 
"""
interpolated_reconstruction = mapper_valued.interpolated_array_from(
    shape_native=(401, 401)
)

"""
If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(interpolated_reconstruction.slim)

plotter = aplt.Array2DPlotter(
    array=interpolated_reconstruction,
)
plotter.figure_2d()

"""
By inputting the arc-second `extent` of the source reconstruction, the interpolated array will zoom in on only these 
regions of the source-plane. The extent is input via the notation (xmin, xmax, ymin, ymax), therefore  unlike the standard 
API it does not follow the (y,x) convention.

Note that the output interpolated array will likely therefore be rectangular, with rectangular pixels, unless 
symmetric y and x arc-second extents are input.
"""
interpolated_reconstruction = mapper_valued.interpolated_array_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_reconstruction.slim)

"""
The interpolated errors on the source reconstruction can also be computed, in case you are planning to perform 
model-fitting of the source reconstruction.
"""
mapper_valued_errors = al.MapperValued(
    mapper=mapper, values=inversion.errors_dict[mapper]
)

interpolated_errors = mapper_valued_errors.interpolated_array_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_errors.slim)

"""
__Magnification__

The magnification of the lens model and source reconstruction can also be computed via the `MapperValued` object,
provided we pass it the reconstruction as the `values`.

This magnification is the ratio of the surface brightness of image in the image-plane over the surface brightness 
of the source in the source-plane.

In the image-plane, this is computed by mapping the reconstruction to the image, summing all reconstructed
values and multiplying by the area of each image pixel. This image-plane image is not convolved with the
PSF, as the source plane reconstruction is a non-convolved image.

In the source-plane, this is computed by interpolating the reconstruction to a regular grid of pixels, for
example a 2D grid of 401 x 401 pixels, and summing the reconstruction values multiplied by the area of each
pixel. This calculation uses interpolation to compute the source-plane image.

The calculation is relatively stable, but depends on subtle details like the resolution of the source-plane 
pixelization and how exactly the interpolation is performed. 
"""
mapper_valued = al.MapperValued(
    mapper=mapper,
    values=inversion.reconstruction_dict[mapper],
)

print("Magnification via Interpolation:")
print(mapper_valued.magnification_via_interpolation_from(shape_native=(401, 401)))

"""
The magnification calculated above used an interpolation of the source-plane reconstruction to a 2D grid of 401 x 401
pixels. 

For a `Rectangular` or `Voronoi` pixelization, the magnification can also be computed using the source-plane mesh
directly, where the areas of the mesh pixels themselves are used to compute the magnification. In certain situations
this is more accurate than interpolation, especially when the source-plane pixelization is irregular. However,
it does not currently work for the `Delanuay` pixelization and is commented out below.
"""
# print("Magnification via Mesh:")
# print(mapper_valued.magnification_via_mesh_from())

"""
The magnification value computed can be impacted by faint source pixels at the edge of the source reconstruction.

The input `mesh_pixel_mask` can be used to remove these pixels from the calculation, such that the magnification
is based only on the brightest regions of the source reconstruction. 

Below, we create a source-plane signal-to-noise map and use this to create a mask that removes all pixels with
a signal-to-noise < 5.0.
"""
reconstruction = inversion.reconstruction_dict[mapper]
errors = inversion.errors_dict[mapper]

signal_to_noise_map = reconstruction / errors

mesh_pixel_mask = signal_to_noise_map < 5.0

mapper_valued = al.MapperValued(
    mapper=mapper,
    values=inversion.reconstruction_dict[mapper],
    mesh_pixel_mask=mesh_pixel_mask,
)

print("Magnification via Interpolation:")
print(mapper_valued.magnification_via_interpolation_from(shape_native=(401, 401)))

"""
__Voronoi__

The pixelization mesh which tests have revealed performs best is the `Voronoi` object, which uses a Voronoi
mesh with a technique called natural neighbour interpolation (full details are provided in the **HowToLens**
tutorials).

I recommend users use this pixelization, how it requires a c library to be installed, thus it is
not the default pixelization used in this tutorial.

If you want to use this pixelization, checkout the installation instructions here:

https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/util/nn

__Wrap Up__

Pixelizations are the most complex but also most powerful way to model a source galaxy.

Whether you need to use them or not depends on the science you are doing. If you are only interested in measuring a
simple quantity like the Einstein radius of a lens, you can get away with using light profiles like a Sersic, MGE or 
shapelets to model the source. Low resolution data also means that using a pixelization is not necessary, as the
complex structure of the source galaxy is not resolved anyway.

However, fitting complex mass models (e.g. a power-law, stellar / dark model or dark matter substructure) requires 
this level of complexity in the source model. Furthermore, if you are interested in studying the properties of the
source itself, you won't find a better way to do this than using a pixelization.

__Chaining__

If your pixelization fit does not go well, or you want for faster computational run-times, you may wish to use
search chaining which breaks the model-fit into multiple Nautilus runs. This is described for the specific case of 
linking a (computationally fast) light profile fit to a pixelization in the script:

`autolens_workspace/scripts/imaging/advanced/chaining/parametric_to_pixelization.py`

__HowToLens__

A full description of how pixelizations work, which comes down to a lot of linear algebra, Bayesian statistics and
2D geometry, is provided in chapter 4 of the **HowToLens** lectures.

__Result (Advanced)__

The code belows shows all additional results that can be computed from a `Result` object following a fit with a
pixelization.

__Max Likelihood Inversion__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit`, which contains the
`Inversion` object we need.
"""
inversion = result.max_log_likelihood_fit.inversion

inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
__Linear Objects__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`. 

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).

- `Mapper`: The linear objected used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper` 
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectnagularMapper` is used for a `Voronoi` mesh).

In this example, the only linear object used to fit the data was a `Pixelization`, thus the `linear_obj_list`
contains just one entry corresponding to a `Mapper`:
"""
print(inversion.linear_obj_list)

"""
To extract results from an inversion many quantities will come in lists or require that we specific the linear object
we with to use. 

Thus, knowing what linear objects are contained in the `linear_obj_list` and what indexes they correspond to
is important.
"""
print(f"Voronoi Mapper = {inversion.linear_obj_list[0]}")

"""
__Grids__

The role of a mapper is to map between the image-plane and source-plane. 

This includes mapping grids corresponding to the data grid (e.g. the centers of each image-pixel in the image and
source plane) and the pixelization grid (e.g. the centre of the Voronoi triangulation in the image-plane and 
source-plane).

All grids are available in a mapper via its `mapper_grids` property.
"""
mapper = inversion.linear_obj_list[0]

# Centre of each masked image pixel in the image-plane.
print(mapper.mapper_grids.image_plane_data_grid)

# Centre of each source pixel in the source-plane.
print(mapper.mapper_grids.source_plane_data_grid)

# Centre of each pixelization pixel in the image-plane (the `Overlay` image_mesh computes these in the image-plane
# and maps to the source-plane).
print(mapper.mapper_grids.image_plane_mesh_grid)

# Centre of each pixelization pixel in the source-plane.
print(mapper.mapper_grids.source_plane_mesh_grid)

"""
__Reconstruction__

The source reconstruction is also available as a 1D numpy array of values representative of the source pixelization
itself (in this example, the reconstructed source values at the vertexes of each Voronoi triangle).
"""
print(inversion.reconstruction)

"""
The (y,x) grid of coordinates associated with these values is given by the `Inversion`'s `Mapper` (which are 
described in chapter 4 of **HowToLens**.
"""
mapper = inversion.linear_obj_list[0]
print(mapper.source_plane_mesh_grid)

"""
The mapper also contains the (y,x) grid of coordinates that correspond to the ray-traced image sub-pixels.
"""
print(mapper.source_plane_data_grid)

"""
__Mapped Reconstructed Images__

The source reconstruction(s) are mapped to the image-plane in order to fit the lens model.

These mapped reconstructed images are also accessible via the `Inversion`. 

Note that any parametric light profiles in the lens model (e.g. the `bulge` and `disk` of a lens galaxy) are not 
included in this image -- it only contains the source.
"""
print(inversion.mapped_reconstructed_image.native)

"""
__Mapped To Source__

Mapping can also go in the opposite direction, whereby we input an image-plane masked 2D array and we use 
the `Inversion` to map these values to the source-plane.

This creates an array which is analogous to the `reconstruction` in that the values are on the source-plane 
pixelization grid, however it bypass the linear algebra and inversion altogether and simply computes the sum of values 
mapped to each source pixel.

[CURRENTLY DOES NOT WORK, BECAUSE THE MAPPING FUNCTION NEEDS TO INCORPORATE THE VARYING VORONOI PIXEL AREA].
"""
mapper_list = inversion.cls_list_from(cls=al.AbstractMapper)

image_to_source = mapper_list[0].mapped_to_source_from(array=dataset.data)

mapper_plotter = aplt.MapperPlotter(mapper=mapper_list[0])
mapper_plotter.plot_source_from(pixel_values=image_to_source)


"""
We can interpolate these arrays to output them to fits.

Although the model-fit used a Voronoi mesh, there is no reason we need to use this pixelization to map the image-plane
data onto a source-plane array.

We can instead map the image-data onto a rectangular pixelization, which has the nice property of giving us a
regular 2D array of data which could be output to .fits format.

[NOT CLEAR IF THIS WORKS YET, IT IS UNTESTED!].
"""
mesh = al.mesh.Rectangular(shape=(50, 50))

source_plane_grid = tracer.traced_grid_2d_list_from(
    grid=dataset.grids.pixelization.over_sampler.over_sampled_grid
)[1]

mapper_grids = mesh.mapper_grids_from(
    mask=mask, source_plane_data_grid=source_plane_grid
)
mapper = al.Mapper(
    mapper_grids=mapper_grids,
    over_sampler=dataset.grids.over_sampler_pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)

image_to_source = mapper.mapped_to_source_from(array=dataset.data)

mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.plot_source_from(pixel_values=image_to_source)

"""
__Linear Algebra Matrices (Advanced)__

To perform an `Inversion` a number of matrices are constructed which use linear algebra to perform the reconstruction.

These are accessible in the inversion object.
"""
print(inversion.curvature_matrix)
print(inversion.regularization_matrix)
print(inversion.curvature_reg_matrix)

"""
__Evidence Terms (Advanced)__

In **HowToLens** and the papers below, we cover how an `Inversion` uses a Bayesian evidence to quantify the goodness
of fit:

https://arxiv.org/abs/1708.07377
https://arxiv.org/abs/astro-ph/0601493

This evidence balances solutions which fit the data accurately, without using an overly complex regularization source.

The individual terms of the evidence and accessed via the following properties:
"""
print(inversion.regularization_term)
print(inversion.log_det_regularization_matrix_term)
print(inversion.log_det_curvature_reg_matrix_term)

"""
__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More magnification calculations.
- Source gradient calculations.
- A calculation which shows differential lensing effects (e.g. magnification across the source plane).
"""
