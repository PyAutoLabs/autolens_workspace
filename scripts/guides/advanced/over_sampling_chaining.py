"""
Chaining: Over Sample
=====================

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

If you are reading this example, you should be familiar with over sampling already. If this is not the case,
checkout the over sampling guide at `autolens_workspace/*/guides/over_sampling.py`.

The guide illustrated adaptive over sampling, where the over sampling sub grid used high resolution pixels in the
centre of a light profile and lower resolution pixels further out. This reached high levels of numerical accuracy
with efficient run times.

However, the adaptive sub grid only works for uniform grids which have not been deflected or ray-traced by the lens
mass model. This criteria is met for the lens galaxy's light, but not for the emission of the lensed source. There
is no automatic adaptive method for the lensed source, which is why the autolens workspace uses cored light profiles
throughout.

An efficient and adaptive over sampling grid is possible. However, it requires using search chaining, where between
searches the over sampling grid is updated.

This example shows how to combine lensed source adaptive over sampling with search chaining.

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/chaining.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
import sys
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.

The data is simulated using a Sersic without a core, unlike most datasets fitted throughout the workspace.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = Path("imaging") / "chaining"

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 - The source galaxy's light is an MGE with 1 x 20 Gaussians [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.

Faint residuals around the multiple images will be present, because the simulated data used a non-cored Sersic
whereas the model fitted is a cored Sersic.
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__sersic_core",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarised in the `info` attribute.
"""
print(result_1.info)

"""
__Over Sampling (Search 2)__

We now create an over sampling grid which applies high levels of over sampling to the brightest regions of the
lensed source in the image plane.

This uses the result of the first lens model in the following way:

 1) Use the lens mass model to ray-trace every deflected image pixel to the source plane, computed the traced grid.
  
 2) Use the traced grid and the centre of the source light profile to compute the distance of every traced image pixel 
    to the source centre. 
    
 3) For all pixels with a distance below a threshold value of 0.1", we set the over sampling factor to a high value of 
    32, which will ensure accuracy in the evaluated of the source's light profile, even after lensing. Pixels 0.1" to
    0.3" from the centre use an over sampling factor of 4, and all other pixels use an over sampling factor of 2.  
"""

tracer = result_1.max_log_likelihood_tracer

traced_grid = tracer.traced_grid_2d_list_from(
    grid=dataset.grid,
)[-1]

source_centre = tracer.galaxies[1].bulge.centre

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=traced_grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.1, 0.3],
    centre_list=[source_centre],
)

"""
One thing to note is this over sampling grid, although specific to the lensed source, is also applied to the lens
galaxy's light. Other than slower computation times, this is not a problem, as the lens galaxy's light in these
pixels will still be evaluated accurately.

The data fitted in this example omits the lens light for simplicity, however if it were present we would want
the lens galaxy's light to be over sampled in the same way as the source galaxy's light because we still require high 
levels of over sampling in the lens galaxy's light to evaluate it correctly. 

We therefore create an over sampling grid which is centred on the lens galaxy's light and combine these values with 
those found for the source galaxy's light, to ensure over sampling is centred on the brightest regions of both galaxies.
"""
over_sample_size_lens = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.1, 0.3],
    centre_list=[(0.0, 0.0)],
)

over_sample_size = np.where(
    over_sample_size > over_sample_size_lens, over_sample_size, over_sample_size_lens
)
over_sample_size = al.Array2D(values=over_sample_size, mask=mask)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 - The source galaxy's light is an MGE with 1 x 20 Gaussians [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.Sersic)

model_2 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_2.info)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.

Faint residuals around the multiple images will be present, because the simulated data used a non-cored Sersic
whereas the model fitted is a cored Sersic.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__sersic_over_sampled",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_2 = al.AnalysisImaging(dataset=dataset)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 1)__

The results which are used for prior passing are summarised in the `info` attribute.
"""
print(result_2.info)

"""
Fin.
"""
