"""
Features: Pixelization
======================

A pixelization reconstructs the source galaxy’s light on a grid of pixels, which is regularized using a prior that
enforces a degree of smoothness in the solution.

This script fits a source galaxy model that uses a pixelization to reconstruct the source’s light. It employs a
rectangular mesh with a constant regularization scheme, which together form the simplest pixelization and
regularization choices available. Despite their simplicity, these choices provide fast and accurate solutions.

For simplicity, the lens galaxy’s light is omitted from both the simulated data and the model. For interferometer
datasets, the lens light is rarely present and this is the common scenario.

You may wish to first read the pixelization/fit.py example, which demonstrates how a pixelized source reconstruction
is applied to a single dataset.

pixelizations are covered in detail in Chapter 4 of the HowToLens lecture series.

__Run Time Overview__

Throughout the workspace, it has been emphasised that pixelized source reconstructions are computed using GPU or CPU
via JAX, where the linear algebra fully exploits sparsity in a way which minimizes VRAM use. This example uses
this functionality, and therefore is suitable for datasets with a low number of visibilities (e.g. < 10000) or
many visibilities (E.g. tens of millions).

This example fits the dataset with 273 visibilities used throughout the workspace, so the modeling runs in under 10
minutes. Fitting a higher resolution dataset will only take an hour to a few hours.

If your dataset contains many visibilities (e.g. millions), setting up the matrices for pixelized source reconstruction
which speed up the linear algebra may take tens of minutes, or hours. Once you are comfortable with the API introduced
in this example, the `feature/pixelization/many_visibilities_preparation` explains how this initial setup can be
performed before lens modeling and saved to hard disk for fast loading before the model fit.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the reconstructed source pixel fluxes can be ensured, but is often disabled for interferometer data.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Pixelization:** How to create a pixelization, including a description of its inputs.
**Model:** Composing a model using a pixelization and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Positions Likelihood:** Removing unphysical pixelized source solutions using a likelihood penalty using the lensed multiple images.
**VRAM:** Profiling of pixelization VRAM use and discussion of how it compares to standard light profiles.
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

For CCD imaging, a disadvantage of pixelized source reconstructions is they are the most computationally expensive
modeling approach. However, for interferometer datasets, the way that JAX and GPUs can exploit the sparsity in the
linear algebra means pixelized source reconstructions are both significantly faster than other approaches (E.g.
light profiles) and can scale to millions of visibilities.

__Disadvantages__

Lens modeling with pixelizations is conceptually more complex. There are additional failure modes, such as
solutions where the source is reconstructed in a highly demagnified configuration due to an unphysical lens mass
model (e.g. too little or too much mass). These issues are discussed in detail later in the workspace.

As a result, learning to successfully fit lens models with pixelizations typically requires more time and experience
than the simpler modeling approaches introduced elsewhere in the workspace.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This could be problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For a pixelizaiton, this often produces negative source pixels which over-fit
the data, producing unphysical solutions.

For CCD imaging datsets pixelized source reconstructions use a positive-only solver, meaning that every source-pixel
is only allowed to reconstruct positive flux values. This ensures that the source reconstruction is physical and
that we don't reconstruct negative flux values that don't exist in the real source galaxy (a common systematic
solution in lens analysis).

However, for interferometer datasets this positive-only solver is often disabled, because negative pixel values
can be observed from the measurement process. All interferometer examples therefore disable the positive only solver,
but you may want to consider if using the positive-only solver is appropriate for your dataset.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is reconstructed using a `RectangularAdaptDensity` mesh
   and `Constant` regularization scheme.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.

__High Resolution Dataset__

A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts large files which
are too big to include in the main `autolens_workspace` repository:

https://github.com/Jammy2211/autolens_workspace_large_files

After downloading the file, place it in the directory:

`autolens_workspace/dataset/interferometer/alma`

You can then perform modeling using this high-resolution dataset by uncommenting the relevant line of code
below.
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
__Mask__

Define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `simple` from .fits files, which we will fit 
with the lens model.

This includes the method used to Fourier transform the real-space image of the strong lens to the uv-plane and compare 
directly to the visiblities. We use a non-uniform fast Fourier transform, which is the most efficient method for 
interferometer datasets containing ~1-10 million visibilities.

If you want to use the high resolution ALMA dataset, uncomment the relevant lines of code below after downloading
the data from the repository described in the "High Resolution Dataset" section above.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__W_Tilde__

Pixelized source modeling requires heavy linear algebra operations. These calculations are greatly accelerated
using an alternative mathematical approach called the **w_tilde formalism**.

You do not need to understand the full details of the method, but the key point is:

- `w_tilde` exploits the **sparsity** of the matrices used in pixelized source reconstruction.
- This leads to a **significant speed-up on GPU or CPU**, using JAX to perform the linear algebra calculations.

To enable this feature, we call `apply_w_tilde()` on the dataset. This computes and stores a `w_tilde_preload` matrix,
which reused in all subsequent pixelized source fits.

On GPU via JAX, this computation is fast even for large datasets with many visibilities, with profiling
of high resolution datasets with over 1 million visibilities showing that computation takes under 20 seconds. For
10s or 100s of millions of visibilities computation on a GPU may stretch to minutes, but this is still very fast.

On CPU, for datasets with over 100000 visibilities and many pixels in their real-space mask, this computation
can take 10 minutes or hours (for the small dataset loaded above its miliseconds). The `show_progress` input outputs 
a progress bar to the terminal so you can monitor the computation, which is useful when it is slow.

When computing it is slow, it is recommend you compute it once, save it to hard-disk, and load it
before modeling. The example `pixelization/many_visibilities_preparation.py` illustrates how to do this.
"""
dataset = dataset.apply_w_tilde(use_jax=True, show_progress=True)

"""
__Settings__

As discussed above, disable the default position only linear algebra solver so the source
reconstruction can have negative pixel values.
"""
settings_inversion = al.SettingsInversion(use_positive_only_solver=False)

"""
__Over Sampling__

If you are familiar with using imaging data, you may have seen that a numerical technique called over sampling is used, 
which evaluates light profiles on a higher resolution grid than the image data to ensure the calculation is accurate.

Interferometer does not observe galaxies in a way where over sampling is necessary, therefore all interferometer
calculations are performed without over sampling.

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
    path_prefix=Path("interferometer"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
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

Create the `AnalysisInterferometer` object defining how the via Nautilus the model is fitted to the data. 

The `positions_likelihood_list` is passed to the analysis, which applies the likelihood penalty described above
for everyone lens mass model.

The `preloads` are passed to the analysis, which contain the static array information JAX needs to perform
the pixelization calculations.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    preloads=preloads,
    settings_inversion=settings_inversion,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM 
required by a model.

pixelizations use a lot less VRAM than light profile-only models, provided the w-tilde sparsity-exploiting
formalism is used (as it is above). In this mode, datasets with tens of millions of visibilities and real space
masks with pixel scales below 0.05" can be stored in just GB's of VRAM, which is remarkable given how much
data they contain.

In w-tilde mode, the **amount of VRAM used is independent of the number of visibilities in the dataset**. 
This is because the w-tilde method compresses all the visibility information into the `w_tilde_preload` matrix, 
whose size depends only on the number of pixels in the real-space mask. VRAM use is therefore mostly driven by
how many pixels are in the real space mask.

VRAM does scale with batch size though, and for high resoluiton datasets may require you to reduce from the value of 
20 set above if your GPU does not have too much VRAM (e.g. < 4GB).
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Time__

The run time of a pixelization are fast provided that the GPU VRAM exceeds the amount of memory required to perform
a likelihood evaluation.

The **run times of a pixelization are independent of the number of visibilities in the dataset**. This is again 
because the w-tilde method compresses all the visibility information into the `curvature_preload` matrix,  whose size 
depends only on the number of pixels in the real-space mask.

Therefore, like VRAM, the main driver of trun time is the number of pixels in the real-space mask,
not the number of visibilities in the dataset. The calculation also runs the same speed irrespective of whether
the real space mask is circular, or irregularly shaped, therefore using a circlular mask is recommended as it is
simpler to set up.

Assuming the use of a 20 x 20 mesh grid above means this is the case, the run times of this model-fit on a GPU
should take under 10 minutes. Increasing the batch size will speed up the fit, provided VRAM allows it.

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

__Result Use__

There are many things you can do with the result of a pixelixaiton, including analysing the reconstructing source, 
magnification calculations of the source and much more.

These are documented in the `fit.py` example.
"""
inversion = result.max_log_likelihood_fit.inversion

"""
__Source Science (Magnification, Flux and More)__

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or galaxies).

Using the reconstructed source model, we can compute key quantities such as the magnification, total flux, and intrinsic 
size of the source.

The example `autolens_workspace/*/guides/source_science` gives a complete overview of how to calculate these quantities,
including examples using a pixelized source reconstruction. 

If you want to study the source galaxy after modeling has reconstructed its unlensed, then check out this example.

__Wrap Up__

pixelizations are the most complex but also most powerful way to model a source galaxy.

Whether you need to use them or not depends on the science you are doing. If you are only interested in measuring a
simple quantity like the Einstein radius of a lens, you can get away with using light profiles like a Sersic, MGE or 
shapelets to model the source. Low resolution data also means that using a pixelization is not necessary, as the
complex structure of the source galaxy is not resolved anyway.

However, fitting complex mass models (e.g. a power-law, stellar / dark model or dark matter substructure) requires 
this level of complexity in the source model. Furthermore, if you are interested in studying the properties of the
source itself, you won't find a better way to do this than using a pixelization.

__Chaining__

Modeling using a pixelization can be more efficient, robust and automated using the non-linear chaining feature to 
compose a pipeline which begins by fitting a simpler model using a parametric source.

More information on chaining is provided in the `autolens_workspace/notebooks/guides/modeling/chaining` folder,
chapter 3 of the **HowToLens** lectures.

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
