"""
Pixelization: CPU Fast Modeling Preparation
===========================================

The example `many_visibility_modeling` demonstrates how to achieve **fast pixelization performance** for
datasets exceeding millions, or hundreds of millions, of visibilities, using a CPU or GPU.

To perform many visibility modeling, a matrix called `curvature_preload` is created and used, which encodes information
and symmetries into the Fourier transform operation performed when modeling interferometer datasets, in a way that
exploits the sparsity of the pixelized source reconstructions and means a very small amount of memory or VRAM is used.

The details can be found in the source code, but you do not need to know them to do science with the code,
nevertheless this ultimately means datasets exceeding millions of visibilities can be modeled in under an hour on a GPU.

The time to compute this matrix can vary between seconds and hours, depending on the number of visibilities in the
dataset and image pixels in the real space mask. If this matrix is not saved and loaded from hard disk, this
recalculation would need to be performed before every model-fit.

This example therefore creates the `curvature_preload` matrix using independent Python code and saves it to hard-disk
for modeling. The `cpu_fast_modeling` example loads this w_tilde matrix from hard-disk if it is available,
and computes it from scratch if not.

__High Resolution Dataset__

A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts large files which
are too big to include in the main `autolens_workspace` repository:

https://github.com/Jammy2211/autolens_workspace_large_files

After downloading the file, place it in the directory:

`autolens_workspace/dataset/interferometer/alma`

You can then compute the `curvature_preload` matrix for this dataset by uncommenting
the line `dataset_name = "alma"` below.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import time

import autolens as al

"""
__Dataset + Masking__ 

Load the `Interferometer` data, define the visibility and real-space masks.
"""
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256), pixel_scales=0.1, radius=mask_radius
)

dataset_name = "simple"
# dataset_name = "alma"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

"""
__Profiling Dataset__

The code above loads a dataset with very few visibilities and a low resolution real space mask, so the 
`curvature_preload` computation is fast.

Real datasets often have 100,000+ visibilities, and a high resolution real space mask, which makes the 
`curvature_preload` computation much slower.

It may therefore be useful to profile the run times for different dataset sizes using the code below, which overwrites 
the dataset above. This will allow you to plan ahead how long the `curvature_preload` computation will take for your 
dataset, and whether doing it on a HPC is necessary.

This code is commented out by default, so your dataset is used instead, but you can uncomment it to run the profiling.
"""
# ### Key run time parameters ###
#
# mask_radius = 3.0
# total_visibilities = 1000000
#
# ### Setup Data ###
#
# real_space_mask = al.Mask2D.circular(
#     shape_native=(800, 800), pixel_scales=0.05, radius=mask_radius
# )
#
# data = al.Visibilities(np.random.normal(loc=0.0, scale=1.0, size=total_visibilities) + 1j * np.random.normal(
#     loc=0.0, scale=1.0, size=total_visibilities
# ))
#
# noise_map = al.VisibilitiesNoiseMap(np.ones(total_visibilities) + 1j * np.ones(total_visibilities))
#
# uv_wavelengths = np.random.uniform(
#     low=-300.0, high=300.0, size=(total_visibilities, 2)
# )
#
# dataset = al.Interferometer(
#     data=data,
#     noise_map=noise_map,
#     uv_wavelengths=uv_wavelengths,
#     real_space_mask=real_space_mask,
#     transformer_class=al.TransformerNUFFT,
# )

"""
__W_Tilde__

Pixelized source modeling requires heavy linear algebra operations. These calculations are greatly accelerated
using an alternative mathematical approach called the **w_tilde formalism**.

You do not need to understand the full details of the method, but the key point is:

- `w_tilde` exploits the **sparsity** of the matrices used in pixelized source reconstruction.
- This leads to a **significant speed-up on GPU or CPU**, using JAX to perform the linear algebra calculations.

To enable this feature, we call `apply_w_tilde()` on the dataset. This computes and stores a `w_tilde_preload` matrix,
which reused in all subsequent pixelized source fits.

For datasets with over 100000 visibilities and many pixels in their real-space mask, this computation
can take 10 minutes or hours. Computing it once, in this script, saving it to hard-disk, and loading it
for modeling is therefore recommended. The `show_progress` input outputs a progress bar to the terminal
so you can monitor the computation.

We comment out the w_tilde calculation below as we are going to illustrate how you can compute it on CPU.
"""
# dataset = dataset.apply_w_tilde(use_jax=True, show_progress=True)

"""
__CPU__

The code below computes the w_tilde matrix on a CPU using a pure numpy implementation, including inputs:

- `chunk_k`: The chunk size of visibilities to process at a time. Decreasing this value decreases the memory
  requirements of the computation, but increases the run time. You should set this as high as your system's
  memory allows.
  
- `show_progress`: Whether to output a progress bar to the terminal, which is on here as for runs which take over
  an hour this is useful to monitor.

- `show_memory`: Whether to output memory usage to the terminal, which is useful to ensure your system has enough
  memory to complete the computation.
"""
curvature_preload = (
    al.util.inversion_interferometer.w_tilde_curvature_preload_interferometer_from(
        noise_map_real=dataset.noise_map.array.real,
        uv_wavelengths=dataset.uv_wavelengths,
        shape_masked_pixels_2d=dataset.grid.mask.shape_native_masked_pixels,
        grid_radians_2d=dataset.grid.mask.derive_grid.all_false.in_radians.native.array,
        chunk_k=2048,
        show_progress=True,
        show_memory=True,
    )
)

"""
__W Tilde Preload Output__

We now output the `curvature_preload` object to hard-disk, so it can be loaded quickly in the 
`cpu_fast_modeling` example.

We save it using a numpy `npz` file, which compresses the data to save hard-disk space, and put it in the 
dataset folder so it can be easily found.
"""
np.save(file=dataset_path / "curvature_preload", arr=curvature_preload)

"""
__Curvature Matrix Time__

The main bottleneck the w-tilde method speeds up is the calculation of the curvature matrix in the source
reconstruction, which is performed for every likelihood evaluation.

If this takes a long time, then lens modeling the dataset will be slow. We therefore time how long it takes
to compute the curvature matrix using the `w_tilde` method on the CPU with numba, you need this to be 
under a second to get reasonable performance.

The **time the curvature matrix calculation takes is independent of the number of visibilities in the dataset**. 
This is because the w-tilde method compresses all the visibility information into the `curvature_preload` matrix, 
whose size depends only on the number of pixels in the real-space mask.

Therefore, the main driver of the curvature matrix calculation time is the number of pixels in the real-space mask,
not the number of visibilities in the dataset. The calculation also runs the same speed irrespective of whether
the real space mask is circular, or irregularly shaped, therefore using a circlular mask is recommended as it is
simpler to set up.

Here are some estimated run times for different mask sizes, where the GPU is a 2021 laptop GPU with 4GB VRAM and 
therefore not too modern

- Mask Radius 3.0", pixel_scale=0.05 (e.g. high resolution ALMA): 1.7 seconds on CPU, < 0.2 seconds on GPU.
- Mask Radius 3.0", pixel_scale=0.025 (e.g. highest resolution ALMA):
- Mask Radius 3.0", pixel_scale=0.01 (e.g. JVLA extreme resolution):

__Wrap Up__

This example has demonstrated how to set up the linear algebra to perform fast pixelized source modeling on
interferometer datasets with many visibilities.
"""
