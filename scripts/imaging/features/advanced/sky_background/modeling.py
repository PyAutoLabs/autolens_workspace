"""
Modeling Features: Sky Background
=================================

The background of an image is the light that is not associated with the strong lens we are interested in. This is due to
light from the sky, zodiacal light, and light from other galaxies in the field of view.

The background sky is often subtracted from image data during the data reduction procedure. If this subtraction is
perfect, there is then no need to include the sky in the model-fitting. However, it is difficult to achieve a perfect
subtraction and there is some uncertainty in the procedure.

The residuals of an imperfect back sky subtraction can leave a signal in the image which is degenerate with the
light profile of the lens galaxy. This is especially true for low surface brightness features, such as the faint
outskirts of the galaxy.

Fitting the sky can therefore ensure errors on light profile parameters which fit the low surface brightness features
further out, like the effective radius and Sersic index, fully account for the uncertainties in the sky background.

This example script illustrates how to include the sky background in the model-fitting of an `Imaging` dataset as
a non-linear free parameter (e.g. an extra dimension in the non-linear parameter space).

__Model__

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The sky background is included as part of a `DatasetModel`.
 - The lens galaxy's light is a linear `Sersic` bulge.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear `SersicCore`.

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

Load and plot the galaxy dataset `sky_background` via .fits files, which we will fit with the model.

This dataset has not had the sky background subtracted from it, therefore the sky background is included in the
image data when we fit it. 

This is seen clearly in the plot, where the outskirts of the image do not go to values near 0.0 electrons per second
like other datasets but instead have values of 5.0 electrons per second, the sky background level used to simulate
the image.
"""
dataset_name = "sky_background"
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

Define a 3.0" circular mask, which includes the emission of the galaxy.
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

Apply adaptive over sampling to ensure the lens galaxy light calculation is accurate, you can read up on over-sampling 
in more detail via the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
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
__Model__

In this example we compose a lens model where:

 - The sky background is included as a `DatasetModel` [1 parameter].

 - The lens galaxy's light is a linear `Sersic` bulge [6 parameters].

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's light is a linear `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=22.

The sky is not included in the `galaxies` collection, but is its own separate component in the overall model.

We update the prior on the `background_sky_level` manually, such that it surrounds the true value of 5.0 electrons
per second. 

You must always update the prior on the sky's intensity manually (unlike light profile priors), because the appropriate
prior depends on the dataset being fitted.
"""
# Lens:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

dataset_model = af.Model(al.DatasetModel)
dataset_model.background_sky_level = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)

# Overall Lens Model:

model = af.Collection(
    dataset_model=dataset_model, galaxies=af.Collection(lens=lens, source=source)
)

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the sky is a model component that is not part of the `galaxies` collection.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=Path("imaging") / "features",
    name="sky_background",
    unique_tag=dataset_name,
    n_live=125,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
)

"""
__Run Time__

For standard light profiles, the log likelihood evaluation time is of order ~0.01 seconds for this dataset.

Adding the background sky model to the analysis has a negligible impact on the run time, as it requires simply adding
a constant value to the data. The run time is therefore still of order ~0.01 seconds.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that a `background_sky_level` of approximately 5.0 electrons per second was inferred, as expected.
"""
print(result.info)

"""
To print the exact value, the `sky` attribute of the result contains the `intensity` of the sky.
"""
print(result.instance.dataset_model.background_sky_level)

"""
Checkout `autogalaxy_workspace/*/guides/result` for a full description of the result object.
"""
