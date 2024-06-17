"""
Settings: Over Sampling
=======================

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

For lensing calculations, the high magnification regions of a lensed source galaxy require especially high levels of
over sampling to ensure the lensed images are evaluated accurately.

This is why throughout the workspace the cored Sersic profile is used, instead of the regular Sersic profile which
you may be more familiar with from the literature. In this example we will increase the over sampling level and
therefore fit a regular Sersic profile to the data, instead of a cored Sersic profile.

This example demonstrates how to change the over sampling used to compute the surface brightness of every image-pixel,
whereby a higher sub-grid resolution better oversamples the image of the light profile so as to provide a more accurate
model of its image.

**Benefit**: Higher level of over sampling provide a more accurate estimate of the surface brightness in every image-pixel.
**Downside**: Higher levels of over sampling require longer calculations and higher memory usage.

You should read up on over-sampling in more detail via  the `autolens_workspace/*/guides/over_sampling.ipynb`
notebook before using this example to customize the over sampling of your model-fits.

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
__Over Sampling API__

To customize the sub-grid used by the model-fit, we create a `OverSamplingUniform` object and specify that the 
`sub_size=4`. 

This increases the sub grid size of the `Grid2D` used to evaluate the galaxy  galaxy `LightProfiles` 
from the default value of 2 to 4.
"""
over_sampling = al.OverSamplingUniform(sub_size=4)

"""
We can alternatively use `OverSamplingIterate` object, where the sub-size of the grid is iteratively increased (in steps 
of 2, 4, 8, 16, 24) until the input fractional accuracy of 99.99% is met.

We will use these settings for the model-fit performed in this script.
"""
over_sampling = al.OverSamplingIterate(
    fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
)

"""
__Dataset + Masking__ 

For this sub-grid to be used in the model-fit, we must pass the `settings_dataset` to the `Imaging` object,
which will be created using a `Grid2D` with a `sub-size value` of 4 (instead of the default of 2).
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

dataset = dataset.apply_over_sampling(
    over_sampling=over_sampling,
)

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model + Search + Analysis__ 

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Nautilus(
    path_prefix=path.join("imaging", "settings"),
    name="sub_grid_size",
    unique_tag=dataset_name,
)

analysis = al.AnalysisImaging(dataset=dataset)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Because the `AnalysisImaging` was passed a `Imaging` with a `sub_size=4` it uses a higher level of sub-gridding
to fit each model `LightProfile` to the data.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can confirm that the `Result`'s grid used a sub-size of 4.
"""
print(result.grid.sub_size)

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Finish.
"""
