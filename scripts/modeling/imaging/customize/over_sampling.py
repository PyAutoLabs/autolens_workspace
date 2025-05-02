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
__Dataset__ 

We load the example dataset which will be used for customizing over sampling.
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
__Over Sampling Uniform__

The over sampling used to fit the data is customized using the `apply_over_sampling` method, which you may have
seen in example `modeling` scripts.

To apply uniform over sampling of degree 4x4, we simply input the integer 4.

The grid this is applied to is called `lp`, to indicate that it is the grid used to evaluate the emission of light
profiles for which this over sampling scheme is applied.
"""
dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

"""
__Over Sampling Adaptive Lens Galaxy__

Above, the `over_sample_size` input has been an integer, however it can also be an `ndarray` of values corresponding
to each pixel. 

Below, we create an `ndarray` of values which are high in the centre, but reduce to 2 at the outskirts, therefore 
providing high levels of over sampling where we need it whilst using lower values which are computationally fast to 
evaluate at the outskirts.

Specifically, we define a 24 x 24 sub-grid within the central 0.3" of pixels, uses a 8 x 8 grid between
0.3" and 0.6" and a 2 x 2 grid beyond that. 

This will provide high levels of over sampling for the lens galaxy, whose emission peaks at the centre of the
image near (0.0", 0.0"), but will not produce high levels of over sampling for the lensed source.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[24, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)


"""
__Over Sampling Source Galaxy__

By assuming the lens galaxy is near (0.0", 0.0"), it was simple to set up an adaptive over sampling grid which is
applicable to all strong lens dataset.

There is no analogous define an adaptive over sampling grid for the lensed source, because for every dataset the
source's light will appear in different regions of the image plane.

This is why the majority of workspace examples use cored light profiles for the source galaxy. A cored light profile
does not rapidly change in its central regions, and therefore can be evaluated accurately without over-sampling.

There is a way to set up an adaptive over sampling grid for a lensed source, however it requries one to use and
understanding the advanced lens modeling feature search chaining.

An example of how to use search chaining to over sample sources efficient is provided in 
the `autolens_workspace/*/imaging/advanced/chaining/over_sampling.ipynb` example.

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
print(result.grids.lp.over_sampled)

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Finish.
"""
