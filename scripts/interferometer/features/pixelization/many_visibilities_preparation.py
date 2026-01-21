"""
Pixelization: Many Visibilities Preparation
===========================================

To perform many visibility modeling, a matrix called `curvature_preload` is created and used, which encodes information
and symmetries into the Fourier transform operation performed when modeling interferometer datasets, in a way that
exploits the sparsity of the pixelized source reconstructions and means a very small amount of memory or VRAM is used.

The details can be found in the source code, but you do not need to know them to do science with the code,
nevertheless this ultimately means datasets exceeding millions of visibilities can be modeled in under an hour on a GPU.

The time to compute this matrix can vary between seconds and hours, depending on the number of visibilities in the
dataset, the number of image pixels in the real space mask and if a CPU or GPU is used. If this matrix is not saved
and loaded from hard disk, this recalculation would need to be performed before every model-fit, which if for your
setup and hardware takes hours would be prohibitive.

On HPC GPUs via JAX, this computation is fast even for large datasets with many visibilities, with profiling
of high resolution datasets with over 1 million visibilities showing that computation takes under 20 seconds. For
10s or 100s of millions of visibilities computation on a GPU may stretch to minutes, but this is still very fast.
If you are lucky enough to have a modern enough GPU, you can therefore compute this matrix on-the-fly during modeling.

On consumer laptop GPUs or CPU, for datasets with over 100000 visibilities and many pixels in their real-space mask, this
computation may take 10 minutes or hours (for the small dataset loaded above its miliseconds). Computing it once,
in this script, saving it to hard-disk, and loading it for modeling is therefore recommended.

On CPU, the `show_progress` input outputs  a progress bar to the terminal so you can monitor the computation,
which is useful when it is slow.

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

As discussed above, the computation of this matrix can take a long time for datasets with many visibilities
and high resolution real-space masks, unless a modern GPU is used.

We comment out the w_tilde calculation below as we are going to illustrate how you can compute it on CPU.

The code has the following inputs:

- `chunk_k`: The chunk size of visibilities to process at a time. Decreasing this value decreases the memory
  requirements of the computation, but increases the run time. You should set this as high as your system's
  memory allows.
  
- `show_progress`: Whether to output a progress bar to the terminal, which is on here as for runs which take over
  an hour this is useful to monitor.

- `show_memory`: Whether to output memory usage to the terminal, which is useful to ensure your system has enough
  memory to complete the computation.
"""
dataset = dataset.apply_w_tilde(
    use_jax=True,
    chunk_k=2048,
    show_progress=True,
    show_memory=True,
)

"""
__W Tilde Preload Output__

We now output the `curvature_preload` object to hard-disk, so it can be loaded quickly in the 
`cpu_fast_modeling` example.

We save it using a numpy `npz` file, which compresses the data to save hard-disk space, and put it in the 
dataset folder so it can be easily found. Note that metadata describing the dataset and its real space
mask is also saved in the file, so that when we load it (shown below) the code can verify it matches the dataset
being modeled.
"""
dataset.w_tilde.save_curvature_preload(
    file=dataset_path / f"curvature_preload_{mask_radius}", overwrite=True
)

"""
To load the `curvature_preload` matrix from hard-disk in your model-fit, you can use the code:
"""
curvature_preload = al.load_curvature_preload_if_compatible(
    file=dataset_path / f"curvature_preload_{mask_radius}",
    real_space_mask=real_space_mask,
)

"""
__Wrap Up__

This example has demonstrated how to set up the linear algebra to perform fast pixelized source modeling on
interferometer datasets with many visibilities.
"""
