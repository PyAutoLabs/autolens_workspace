"""
Modeling: Customize
===================

This script gives a run through of all the different ways the analysis can be customized for lens modeling, with
reasons explaining why each customization is useful.

__Contents__

**Dataset**: Load a dataset which is used to illustrate the customizations.
**Mask:** Apply a custom mask to the dataset, which can be used to remove regions of the image that contain no emission.
**Over Sampling:** Change the over sampling used to compute the surface brightness of every image-pixel.
**Positions:** Specify the positions of the lensed source's multiple images, which can be used to discard unphysical mass models.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

All customizations in this script are applied to the strong lens dataset `simple__no_lens_light`, which is a
simple strong lens with no lens light emission.

We therefore load and plot the strong lens dataset `simple__no_lens_light` via .fits files.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
__Mask__

All example default scritps use a circular mask to model a lens, which contains the lens and source emission.
However, the mask can be customized to better suit the lens and source emission, for example by using an annular
mask to only contain the emission of the Einstein ring itelf.

Advantages: Strong lenses with complex and difficult-to-subtract foreground lens galaxies can leave residuals that
bias the mass and source models, which this custom mask can remove from the model-fit. The custom mask can also provide
faster run times, as the removal of large large regions of the image (which contain no signal) no longer need to be
processed and fitted.

Disadvantages: Pixels containing no source emission may still constrain the lens model, if a mass model incorrectly
predicts that flux will appear in these image pixels. By using a custom mask, the model-fit will not be penalized for
incorrectly predicting flux in these image-pixels (As the mask has removed them from the fit).

We first show an example using an annular masks, which because the data does not contain lens light can be
used to remove the central pixels containing no emission and thus only fit the source.
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.5,
    outer_radius=2.5,
)

dataset = dataset.apply_mask(mask=mask)  # <----- The custom mask is used here!

visuals = aplt.Visuals2D(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
We can also load the mask from a .fits file, which could have been produced in a way which is even more customized
to the source emission than the annular masks above.. 

To create the .fits file of a mask, we use a GUI tool which is described in the following script:

 `autolens_workspace/*/imaging/data_preparation/gui/mask.py`
"""
mask = al.Mask2D.from_fits(
    file_path=Path(dataset_path, "mask_gui.fits"),
    hdu=0,
    pixel_scales=dataset.pixel_scales,
)

dataset = dataset.apply_mask(mask=mask)  # <----- The custom mask is used here!

visuals = aplt.Visuals2D(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

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

Prequisites: You should read `autolens_workspace/*/guides/advanced/over_sampling.ipynb` before running this script, which
introduces the concept of over sampling in PyAutoLens and explains why the lens and source galaxy are evaluated
on different grids.



The over sampling used to fit the data is customized using the `apply_over_sampling` method, which you may have
seen in example `modeling` scripts.

To apply uniform over sampling of degree 4x4, we simply input the integer 4.

The grid this is applied to is called `lp`, to indicate that it is the grid used to evaluate the emission of light
profiles for which this over sampling scheme is applied.
"""
dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

"""
Above, the `over_sample_size` input has been an integer, however it can also be an `ndarray` of values corresponding
to each pixel. 

We create an `ndarray` of values which are high in the centre, but reduce to 2 at the outskirts, therefore 
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
"""

"""
__Positions__

Before fitting a strong lens, we can manually specify a grid of image-plane coordinates corresponding to the multiple
images of the lensed source-galaxy(s). During the model-fit, **PyAutoLens** will check that these coordinates trace
within a specified arc-second threshold of one another in the source-plane. If they do not meet this threshold, the
mass model is discarded and a new sample is generated by the non-linear search.

Advantages: The model-fit is faster, as the non-linear search avoids regions of parameter space where the mass-model
is clearly not accurate. Removing these unphysical solutions may also mean that the global-maximum solution is inferred
instead of a local-maxima, given that removing unphysical mass models makes non-linear parameter space less complex.

Disadvantages: If the positions are inaccurate or threshold is set too low, one may inadvertantly remove the correct
mass model!

The positions are associated with the and they are loaded from a `positions.json` file which is in the same folder as 
the dataset itself. 

To create this file, we used a GUI to `draw on` the positions with our mouse. This GUI can be found in the script:

 `autolens_workspace/*/imaging/data_preparation/gui/positions.py`

If you wish to use positions for modeling your own lens data, you should use this script to draw on the positions of
every lens in you dataset.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "positions.json"))
)

visuals = aplt.Visuals2D(mask=mask, positions=positions)
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
Alternatively, the positions can be specified manually in the modeling script script using the `Grid2DIrregular`object.

Below, we specify a list of (y,x) coordinates (that are not on a uniform or regular grid) which correspond to the 
arc-second (y,x) coordinates ot he lensed source's brightest pixels.
"""
positions = al.Grid2DIrregular(
    [(0.4, 1.6), (1.58, -0.35), (-0.43, -1.59), (-1.45, 0.2)]
)

visuals = aplt.Visuals2D(mask=mask, positions=positions)
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
To use positions in lens modeling, we pass the `AnalysisImaging` object a `PositionsLH` object, which includes the 
positions we loaded above, alongside a `threshold`.

The threshold of 0.3" is large. For an accurate lens model we would anticipate the positions trace within < 0.01" of
one another. The high threshold ensures only the initial mass models at the start of the fit are penalized.
"""
positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
The analysis receives a list of `PositionsLH` objects, which allows us to use multiple sets of positions to apply this 
penalty, for example if there source has multiple sets of multiple images which we know how they map to one another.
"""
analysis = al.AnalysisImaging(
    dataset=dataset, positions_likelihood_list=[positions_likelihood]
)

"""
Finish.
"""
