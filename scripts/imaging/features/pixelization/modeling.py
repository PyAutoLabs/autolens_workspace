"""
Features: Pixelization Modeling
===============================

A pixelization reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
the solution to have a degree of smoothness.

This script fits a source galaxy model which uses a pixelization to reconstruct the source's light.

A Rectangular mesh and constant regularization scheme are used, which are the simplest forms of mesh and regularization
with provide computationally fast and accurate solutions.

For simplicity, the lens galaxy’s light is omitted from both the simulated data and the model. Including the lens
galaxy’s light is straightforward and can be done in exactly the same framework.

You may wish to first read the pixelization/fit.py example, which demonstrates how a pixelized source reconstruction
is applied to a single dataset.

Pixelizations are covered in detail in chapter 4 of the **HowToLens** lectures.

__Run Time Overview__

Pixelized source reconstructions are computed using either GPU acceleration via JAX or CPU acceleration via `numba`.

The faster option depends on two crucial factors:

#### **1. GPU VRAM Limitations**
JAX only provides significant acceleration on GPUs with **large VRAM (≥16 GB)**.
To avoid excessive VRAM usage, examples often restrict pixelization meshes (e.g. 20 × 20).
On consumer GPUs with limited memory, **JAX may be slower than CPU execution**.

#### **2. Sparse Matrix Performance**

Pixelized inversions require operations on **very large, highly sparse matrices**.

- JAX currently lacks sparse-matrix support and must compute using **dense matrices**, which scale poorly.
- PyAutoLens’s CPU implementation (via `numba`) fully exploits sparsity, providing large speed gains
  at **high image resolution** (e.g. `pixel_scales <= 0.03`).

As a result, CPU execution can outperform JAX even on powerful GPUs for high-resolution datasets.

The example `pixelization/cpu_fast_modeling` shows how to set up a pixelization to use efficient CPU calculations
via the library `numba`.

__Rule of Thumb__

For **low-resolution imaging** (for example, datasets with `pixel_scales > 0.05`), modeling is generally faster using
**JAX with a GPU**, because the computations involve fewer sparse operations and do not require large amounts of VRAM.

For **high-resolution imaging** (for example, `pixel_scales <= 0.03`), modeling can be faster using a **CPU with numba**
and multiple cores. At high resolution, the linear algebra is dominated by sparse matrix operations, and the CPU
implementation exploits sparsity more effectively, especially on systems with many CPU cores (e.g. HPC clusters).

**Recommendation:** The best choice depends on your hardware and dataset. If your data has resolution of 0.1" per pixel
 (e.g. Euclid imaging) or lower, JAX will be the most efficient. For higher resolution imaging (e.g. HST, JWST) it is
 worth benchmarking both approaches (GPU+JAX vs CPU+numba) to determine which performs fastest for your case.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using a pixelization to model a source galaxy.
**Positive Only Solver:** How a positive solution to the reconstructed source pixel fluxes is ensured.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Pixelization:** How to create a pixelization, including a description of its inputs.
**Model:** Composing a model using a pixelization and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Positions Likelihood:** Removing unphysical pixelized source solutions using a likelihood penalty using the lensed multiple images.
**Run Time:** Profiling of pixelization run times and discussion of how they compare to standard light profiles.
**Model-Fit:** Performs the model fit using standard API.
**Result:** Pixelization results and visualizaiton.
**Chaining:** How the advanced modeling feature, non-linear search chaining, can significantly improve lens modeling with pixelizaitons.
**Result (Advanced):** API for various pixelization outputs (magnifications, mappings) which requires some polishing.
**Simulate (Advanced):** Simulating a strong lens dataset with the inferred pixelized source.

__Advantages__

Many strongly lensed source galaxies exhibit complex, asymmetric, and irregular morphologies. Such structures
cannot be well approximated by analytic light profiles such as a Sérsic profile, or even combinations of multiple
Sérsic components. pixelizations are therefore required to accurately reconstruct this irregular source-plane light.

Even alternative basis-function approaches, such as shapelets or multi-Gaussian expansions, struggle to accurately
reconstruct sources with highly complex morphologies or multiple distinct source galaxies.

Pixelized source models are also essential for robustly constraining detailed components of the lens mass
distribution (e.g. the mass density slope or the presence of dark matter substructure). By fitting all of the lensed
source light, they reduce degeneracies between the source and lens mass model.

Finally, many science applications aim to study the highly magnified source galaxy itself, in order to learn about
distant and intrinsically faint galaxies. pixelizations reconstruct the unlensed source emission, enabling detailed
studies of the source-plane structure.

__Disadvantages__

Pixelized source reconstructions are computationally more expensive than analytic source models. For high-resolution
imaging data (e.g. Hubble Space Telescope observations), it is common for lens models using pixelizations to require
hours or even days to fit.

Lens modeling with pixelizations is also conceptually more complex. There are additional failure modes, such as
solutions where the source is reconstructed in a highly demagnified configuration due to an unphysical lens mass
model (e.g. too little or too much mass). These issues are discussed in detail later in the workspace.

As a result, learning to successfully fit lens models with pixelizations typically requires more time and experience
than the simpler modeling approaches introduced elsewhere in the workspace.

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

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is reconstructed using a `RectangularAdaptDensity` mesh
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
"""
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

 - The source-galaxy's light uses a 20 x 20 `RectangularAdaptDensity` mesh [0 parameters].

 - This pixelization is regularized using a `Constant` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6. 

It is worth noting the pixelization fits the source using significantly fewer parameters (1 parameter for 
regularization) than fitting the source using light profiles or an MGE (4+ parameters). 

The lens model therefore includes a mesh and regularization scheme, which are used together to create the 
pixelization. 
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
    path_prefix=Path("imaging"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
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

The `positions_likelihood_list` is passed to the analysis, which applies the likelihood penalty described above
for everyone lens mass model.

The `preloads` are passed to the analysis, which contain the static array information JAX needs to perform
the pixelization calculations.
"""
analysis = al.AnalysisImaging(
    dataset=dataset, positions_likelihood_list=[positions_likelihood], preloads=preloads
)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM 
required by a model.

pixelizations use a lot more VRAM than light profile-only models, with the amount required depending on the size of
dataset and the number of source pixels in the pixelization's mesh. For 400 source pixels, around 0.05 GB per batched
likelihood of VRAM is used. 

This is why the `batch_size` above is 20, lower than other examples, because reducing the batch size ensures a more 
modest amount of VRAM is used. If you have a GPU with more VRAM, increasing the batch size will lead to faster run times.

Given VRAM use is an important consideration, we print out the estimated VRAM required for this 
model-fit and advise you do this for your own pixelization model-fits.
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

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

There are many things you can do with the result of a pixelixaiton, including analysing the reconstructed source, 
magnification calculations of the source and much more.

These are documented in the `fit.py` example.
"""
inversion = result.max_log_likelihood_fit.inversion

"""
__Wrap Up__

Pixelizations are the most complex but also the most powerful way to model a galaxy’s light.

Whether you need to use them depends on the science you are doing. If you are only interested in measuring simple
global quantities (for example, total flux, size, or axis ratio), analytic light profiles such as a Sérsic, MGE, or
shapelets are often sufficient. For low-resolution data, pixelizations are also unnecessary, as the complex
structure of the galaxy is not resolved.

However, modeling galaxies with complex, irregular, or highly structured light distributions requires this level of
flexibility. Furthermore, if you are interested in studying the detailed morphology of a galaxy itself, there is no
better approach than using a pixelization.

__Chaining__

Modeling with a pixelization can be made more efficient, robust, and automated using the non-linear chaining feature
to compose a pipeline that begins by fitting a simpler model using parametric light profiles.

More information on chaining is provided in the
`autogalaxy_workspace/notebooks/guides/modeling/chaining` folder and in chapter 3 of the **HowToGalaxy** lectures.

__HowToGalaxy__

A full description of how pixelizations work—which relies heavily on linear algebra, Bayesian statistics, and
2D geometry—is provided in chapter 4 of the **HowToGalaxy** lectures.

__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More diagnostic quantities for reconstructed galaxy light.
- Gradient calculations of the reconstructed light distribution.
- Quantifying spatial variations in galaxy structure across the image.
"""
