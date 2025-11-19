"""
Features: Pixelization
======================

A pixelization reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
the solution to have a degree of smoothness.

This script fits a source galaxy model which uses a pixelization to reconstruct the source's light.

A Voronoi mesh and constant regularization scheme are used, which are the simplest forms of mesh and regularization
with provide computationally fast and accurate solutions in **PyAutoLens**.

For simplicity, the lens galaxy's light is omitted from the model and is not present in the simulated data. It is
straightforward to include the lens galaxy's light in the model.

You may wish to first read the `pixelization/fit.py` example in order to see how a pixelization is fitted
to an individual data.

Pixelizations are covered in detail in chapter 4 of the **HowToLens** lectures.

__JAX GPU Run Times__

Pixelizations run time depends on how modern GPU hardware is. GPU acceleration only provides fast run times on
modern GPUs with large amounts of VRAM, or when the number of pixels in the mesh are low (e.g. < 500 pixels).

This script's default setup uses an adaptive 20 x 20 rectangular mesh (400 pixels), which is relatively low resolution
and may not provide the most accurate lens modeling results. On most GPU hardware it will run in ~ 10 minutes,
however if your laptop has a large VRAM (GPU > 20 GB) or you can access a GPU cluster with better hardware you should use these
to perform modeling with increased mesh resolution.

__CPU Run Times__

JAX is not natively designed to provide significant CPU speed up, therefore users using CPUs to perform pixelization
analysis will not see fast run times using JAX (unlike GPUs).

The example `pixelization/cpu` shows how to set up a pixelization to use efficient CPU calculations via the library
`numba`.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the light profile intensities is ensured.
**Chaining:** How the advanced modeling feature, non-linear search chaining, can significantly improve lens modeling with pixelizaitons.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Pixelization:** How to create a pixelization, including a description of its inputs.
**Model:** Composing a model using a pixelization and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Positions Likelihood:** Removing unphysical pixelized source solutions using a likelihood penalty using the lensed multiple images.
**Run Time:** Profiling of pixelization run times and discussion of how they compare to standard light profiles.
**Model-Fit:** Performs the model fit using standard API.
**Result:** Pixelization results and visualizaiton.
**Interpolated Source:** Interpolate the source reconstruction from an irregular Voronoi mesh to a uniform square grid and output to a .fits file.
**Reconstruction CSV:** Output the source reconstruction to a .csv file, which can be used to perform calculations on the source reconstruction.
**Voronoi:** Using a Voronoi mesh pixelizaiton (instead of Voronoi), which allows one to manipulate the source reconstruction without autolens installed.
**Result (Advanced):** API for various pixelization outputs (magnifications, mappings) which requires some polishing.
**Simulate (Advanced):** Simulating a strong lens dataset with the inferred pixelized source.

__Advantages__

Many strongly lensed source galaxies are complex, and have asymmetric and irregular morphologies. These morphologies
cannot be well approximated by a light profiles like a Sersic, or many Sersics, and thus a pixelization
is required to reconstruct the source's irregular light.

Even basis functions like shapelets or a multi-Gaussian expansion cannot reconstruct a source-plane accurately
if there are multiple source galaxies, or if the source galaxy has a very complex morphology.

To infer detailed components of a lens mass model (e.g. its density slope, whether there's a dark matter subhalo, etc.)
then pixelized source models are required, to ensure the mass model is fitting all of the lensed source light.

There are also many science cases where one wants to study the highly magnified light of the source galaxy in detail,
to learnt about distant and faint galaxies. A pixelization reconstructs the source's unlensed emission and thus
enables this.

__Disadvantages__

Pixelizations are computationally slow and run times are typically longer than a source model. It is not
uncommon for lens models using a pixelization to take hours or even days to fit high resolution imaging
data (e.g. Hubble Space Telescope imaging).

Lens modeling with pixelizations is also more complex than source models, with there being more things
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

More information on chaining is provided in the `autolens_workspace/notebooks/guides/modeling/chaining` folder,
chapter 3 of the **HowToLens** lectures.

The script `autolens_workspace/scripts/guides/modeling/chaining/parametric_to_pixelization.py` explitly uses chaining
to link a lens model using a light profile source to one which then uses a pixelization.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is reconstructed using a `Voronoi` mesh, `Overlay` image-mesh
   and `Constant` regularization scheme.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
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
__Dataset__

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

A pixelization uses a separate grid for ray tracing, with its own over sampling scheme, which below we set to a 
uniform grid of values of 2. 

The pixelization only reconstructs the source galaxy, therefore the adaptive over sampling used for the lens galaxy's 
light in other examples is not applied to the pixelization. 

This example does not model lens light, for examples which combine lens light and a pixelization both over sampling 
schemes should be used, with the lens light adaptive and the pixelization uniform.

Note that the over sampling is input into the `over_sample_size_pixelization` because we are using a `Pixelization`.
"""
dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=4,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__JAX & Preloads__

In JAX, calculations must use static shaped arrays with known and fixed indexes. For certain calculations in the
pixelization, this information has to be passed in before the pixelization is performed. Below, we do this for 3
inputs:

- `total_linear_light_profiles`: The number of linear light profiles in the model. This is 0 because we are not
  fitting any linear light profiles to the data, primarily because the lens light is omitted.

- `total_mapper_pixels`: The number of source pixels in the rectangular pixelization mesh. This is required to set up 
  the arrays that perform the linear algebra of the pixelization.

- `source_pixel_zeroed_indices`: The indices of source pixels on its edge, which when the source is reconstructed 
  are forced to values of zero, a technique tests have shown are required to give accruate lens models.

The `image_mesh` can be ignored, it is legacy API from previous versions which may or may not be reintegrated in future
versions.
"""
image_mesh = None
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

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data.  In this 
example fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source-galaxy's light uses a 20 x 20 `RectangularMagnification` mesh [0 parameters].

 - This pixelization is regularized using a `Constant` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6. 

It is worth noting the `Pixelization`  use significantly fewer parameters (1 parameter) than 
fitting the source using `LightProfile`'s (7+ parameters). 

The lens model therefore includes a mesh and regularization scheme, which are used together to create the 
pixelization. 
"""
# Lens:

mass = af.Model(al.mp.PowerLaw)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:
mesh = af.Model(al.mesh.RectangularMagnification, shape=mesh_shape)
regularization = af.Model(al.reg.Constant)

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
    path_prefix=Path("features"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
    iterations_per_quick_update=50000,
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

Unlike other example scripts, we also pass the `AnalysisImaging` object below a `PositionsLH` object, which
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
one another. The high threshold ensures only the initial mass models at the start of the fit are penalized.

Position thresholding is described in more detail in the 
script `autolens_workspace/*/guides/modeling/customize`

The arc-second positions of the multiply imaged lensed source galaxy were drawn onto the
image via the GUI described in the file `autolens_workspace/*/imaging/data_preparation/gui/positions.py`.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "positions.json"))
)

positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data. 
"""
analysis = al.AnalysisImaging(
    dataset=dataset, positions_likelihood_list=[positions_likelihood], preloads=preloads
)

"""
__Run Time__

The run time of a pixelization are fast provided that the GPU VRAM exceeds the amount of memory required to perform
a likelihood evaluation.

Assuming the use of a 20 x 20 mesh grid above means this is the case, the run times of this model-fit on a GPU
should take under 10 minutes. If VRAM is exceeded, the run time will be significantly longer (3+ hours). CPU run
times are also of order hours, but can be sped up using the `numba` library (see the `pixelization/cpu` example).

The run times of pixelizations slow down as the data becomes higher resolution. In this example, data with a pixel
scale of 0.1" gives of order 10 minute run times (when VRAM is under control), for a pixel scale of 0.05" this
becomes around 30 minutes, and an hour for 0.03".

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
    tracer=result.max_log_likelihood_tracer, grid=result.grids.lp
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
The example `pixelization/fit` provides a full description of the different calculations that can be performed
with the result of a pixelization model-fit.

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
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=Path(dataset_path, "mask_extra_galaxies.fits"),
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

__Result Use__

There are many things you can do with the result of a pixelixaiton, including analysing the reconstructing source, 
magnification calculations of the source and much more.

These are documented in the `fit.py` example.

This example centres on the `inversion` object which is easily accessed from the result of a model-fit.
"""
inversion = result.max_log_likelihood_fit.inversion


"""
__Reconstruction CSV__

In the results `image` folder there is a .csv file called `source_plane_reconstruction_0.csv` which contains the
y and x coordinates of the pixelization mesh, the reconstruct values and the noise map of these values.

This file is provides all information on the source reconstruciton in a format that does not depend autolens
and therefore be easily loaded to create images of the source or shared collaobrations who do not have PyAutoLens
installed.

First, lets load `source_plane_reconstruction_0.csv` as a dictionary, using basic `csv` functionality in Python.
"""
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

For example, we can create a RectangularMagnification mesh using the scipy.spatial library, which is a triangulation
of the y and x coordinates of the pixelization mesh. This is useful for visualizing the pixelization
and performing calculations on the mesh.
"""
import scipy

points = np.stack(arrays=(reconstruction_dict["x"], reconstruction_dict["y"]), axis=-1)

mesh = scipy.spatial.RectangularMagnification(points)

"""
Interpolating the result to a uniform grid is also possible using the scipy.interpolate library, which means the result
can be turned into a uniform 2D image which can be useful to analyse the source with tools which require an uniform grid.

Below, we interpolate the result onto a 201 x 201 grid of pixels with the extent spanning -1.0" to 1.0", which
capture the majority of the source reconstruction without being too high resolution.

It should be noted this inteprolation may not be as optimal as the interpolation perforemd above using `MapperValued`, 
which uses specifc interpolation methods for a RectangularMagnification mesh which are more accurate, but it should be sufficent for
most use-cases.
"""
from scipy.interpolate import griddata

values = reconstruction_dict["reconstruction"]

interpolation_grid = al.Grid2D.from_extent(
    extent=(-1.0, 1.0, -1.0, 1.0), shape_native=(201, 201)
)

interpolated_array = griddata(points=points, values=values, xi=interpolation_grid)

"""
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

`autolens_workspace/scripts/guides/modeling/chaining/parametric_to_pixelization.py`

__HowToLens__

A full description of how pixelizations work, which comes down to a lot of linear algebra, Bayesian statistics and
2D geometry, is provided in chapter 4 of the **HowToLens** lectures.

__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More magnification calculations.
- Source gradient calculations.
- A calculation which shows differential lensing effects (e.g. magnification across the source plane).
"""
