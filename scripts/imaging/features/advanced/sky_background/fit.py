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
 - The lens galaxy's light is a linear parametric `Sersic` bulge.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear parametric `SersicCore`.

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
__Fit__

We first show how to use a `DatasetModel` object to fit the sky background in the data.

This illustrates the API for performing a sky background fit using standard objects like a `Galaxy` and `FitImaging` .

This does not perform a model-fit via a non-linear search, and therefore requires us to manually specify and guess
suitable parameter values for the sky. We will use the true value of 5.0 electrons per second.

For the galaxies, we will use the true parameters used to simulate the data, for illustrative purposes.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

dataset_model = al.DatasetModel(background_sky_level=5.0)

fit = al.FitImaging(dataset=dataset, tracer=tracer, dataset_model=dataset_model)

"""
By plotting the fit, we see that the sky is subtracted from the data such that the outskirts are zero.

There are few residuals, except for perhaps some central regions where the light profile is not perfectly fitted.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
Finish.
"""
