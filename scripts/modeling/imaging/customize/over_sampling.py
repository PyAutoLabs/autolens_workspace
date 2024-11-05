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

Over sampling is applied separately to the light profiles which compute the surface brightness of the lens galaxy,
which are on a `uniform` grid, and the light profiles which compute the surface brightness of the source galaxy,
which are on a `non-uniform` grid.

__Prequisites__

You should read `autolens_workspace/*/guides/advanced/over_sampling.ipynb` before running this script, which
introduces the concept of over sampling in PyAutoLens and explains why the lens and source galaxy are evaluated
on different grids.

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

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling Lens Galaxy (Uniform)__

The over sampling of the lens galaxy is controlled using the `OverSamplingUniform` object, where an adaptive
over sampling grid is used to compute the surface brightness of the lens galaxy such that high levels of over sampling
are used in the central regions of the lens galaxy at (0.0", 0.0").
"""
over_sampling_lens = al.OverSamplingUniform.from_radial_bins(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.1, 0.3],
    centre_list=[(0.0, 0.0)],
)

"""
__Over Sampling Source Galaxy__

To customize the sub-grid used by the model-fit, we create a `OverSamplingUniform` object and specify that the 
`sub_size=4`. 

This increases the sub grid size of the `Grid2D` used to evaluate the source galaxy light profiles from the default 
value of 2 to 8.

For many reasons, this uniform grid is not ideal, as we will use high levels of over sampling over the whole mask,
including the regions where the lensed source is not located. This is inefficient and can lead to longer run times
and higher memory usage.

Checkout `autolens_workspace/*/guides/over_sampling.ipynb`
and `autolens_workspace/*/advanced/chaining/examples/over_sample.py` for a discussion of how to use an adaptive
over sampling grid to compute the surface brightness of the source galaxy.
"""
over_sampling_source = al.OverSamplingUniform(sub_size=8)

"""
__Over Sampling__

We now apply the over sampling to the `Imaging` dataset.
"""
dataset = dataset.apply_over_sampling(
    over_sampling=al.OverSamplingDataset(
        uniform=over_sampling_lens, non_uniform=over_sampling_source
    )
)

"""
__Model + Search + Analysis__ 

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Nautilus(
    path_prefix=path.join("imaging", "settings"),
    name="over_sampling",
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

We can confirm that the `Result`'s grid used an over sampling iterate object.
"""
print(result.grids.uniform.over_sampling)

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Finish.
"""
