"""
Modeling Features: Deblending
=============================

The image-plane multiple-image positions of a lensed point source (e.g. a quasar or supernova) are used as the
dataset in point-source modeling. For example, simulated values were input into the `PointDataset` object in the
`point_source/modeling/start_here.ipynb` example.

These positions must first be measured from imaging data of the lensed point-source. A simple way to do this is
to locate the brightest 2 or 4 pixels of the lensed point-source (e.g. via a GUI or ds9) and use these values
as the positions. 

For many users this will be sufficient, however it has downsides:

- It does not provide sub-pixel precision on the positions.

- It does not account for the Point Spread Function.

It also does not measure the following quantities at all:

- The flux of each lensed point source (it provides an estimate via the brightest pixel fluxes, but proper deblending
of the PSF is key for accurate flux measurements).

- Any properties of the lens galaxy's light, which is blended with the lensed point source images.

In this example, we perform this deblending so that we can accurately measure the point-source positions, fluxes and
properties of the lens galaxy's light.

__Image Plane Multiple Images__

When fitting the `Imaging` dataset in order to deblend the lensed point-source images and lens galaxies, the four
multiple images of the lensed point source are modeled in the image-plane, using four independent light profiles.

This means the model does not place a point-source in the source-plane, and does not use ray-tracing to determine its
image-plane multiple images and fluxes.

The reason for this is due to a phenomenon called 'micro lensing'. In brief, each multiple image of the lensed
point source will have its magnification boosted or reduced by stars in the lens galaxy. This lensing effect is
extremely difficult to model accurately in the lens galaxy's mass model. This means that if we modeled the lensed point
source in the source-plane, we would not be able to accurately measure its fluxes.

By fitting each multiple image in the image-plane, the effects of micro lensing on each multiple image are accounted
for in the deblending process by the `intensity` of each light profile being free parameters in the model. Micro
lensing is also why `fluxes` are typically not used to fit point source lens models.

__Point Source Host Galaxy__

For high quality imaging of a lensed point source, the light from the point source's host galaxy may also be visible.
The deblending procedure illustrated in this script can also therefore be used to extract and model the host galaxy's
light.

We do not perform any deblending of the lensed point source's host source galaxy, because it requires a more
sophisticated analysis. An example script for doing this is not currently available, but if it would be useful for
you please contact me on SLACK and I can write it!

__Imaging__

This example script fits `Imaging` data, using many of the features described in the `modeling/imaging` workspace
examples.

It also uses the following features described in the `modeling/imaging/features` workspace examples:

- `linear_light_profiles.py`: The model includes light profiles which use linear algebra to solve for their
   intensity, reducing model complexity.

- `advanced/operated_light_profiles.py`: There are light profiles which are assumed to already be convolved with
  the instrumental PSF (e.g. point sources), commonly used for modeling bright AGN in the centre of a galaxy.

It is recommended you are familiar with imaging modeling and these features before reading this example.

However, you would probably be able to use and adapt this script to your use-case even if you are not.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' point-source strong lens with a model where:

 - The lens galaxy's light is a parametric linear `Sersic` bulge.
 - The multiple images of the lensed source are each fitted with a `Gaussian` operated linear light profile.

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

Load and plot the strong lens dataset `deblending` via .fits files.
"""
dataset_name = "deblending"
dataset_path = path.join("dataset", "point_source", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and lensed point-sources.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

We compose a lens model where:

 - The lens galaxy's light is a linear parametric `Sersic` bulge [6 parameters].
 - The four image-plane multiple images of the lensed source are each fitted with a `Gaussian` operated linear light 
 profile [4 x 5 = 20 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=26.

Note how all light profiles are linear light profiles, meaning that the `intensity` parameter of all profiles are
not free parameters in the fit but instead are solved via linear algebra. This reduces the dimensionality of the
non-linear parameter space by N=5.

We note that our lens model therefore does not include:

 - A lens galaxy with a total mass distribution.
 - A source galaxy's with a light profile or point source.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)

# Lensed Source Multiple Images (Image-Plane):

multiple_image_0 = af.Model(al.lp_linear_operated.Gaussian)
multiple_image_1 = af.Model(al.lp_linear_operated.Gaussian)
multiple_image_2 = af.Model(al.lp_linear_operated.Gaussian)
multiple_image_3 = af.Model(al.lp_linear_operated.Gaussian)

"""
The model has N=26 free parameters, and is quite a complex model to fit. 

The different multiple image light profiles can also produce identical solutions, because each `Gaussian` point source 
can change its `centre` to fit different lensed point source images.

To simplify the model and remove identical solutions, we manually set the priors on the `centre` of the lens galaxy 
light profile and each multiple image light profile to narrow uniform priors. The values of these priors are based on
where the peak fluxes of each image appear to be located in the image plotted above.
"""
bulge.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

multiple_image_0_estimate = (1.148, -1.148)
multiple_image_1_estimate = (1.190, 1.190)
multiple_image_2_estimate = (-1.190, -1.190)
multiple_image_3_estimate = (-1.148, 1.148)

multiple_image_width = dataset.pixel_scales[0]

multiple_image_0.centre_0 = af.UniformPrior(
    lower_limit=multiple_image_0_estimate[0] - multiple_image_width,
    upper_limit=multiple_image_0_estimate[0] + multiple_image_width,
)
multiple_image_0.centre_1 = af.UniformPrior(
    lower_limit=multiple_image_0_estimate[1] - multiple_image_width,
    upper_limit=multiple_image_0_estimate[1] + multiple_image_width,
)

multiple_image_1.centre_0 = af.UniformPrior(
    lower_limit=multiple_image_1_estimate[0] - multiple_image_width,
    upper_limit=multiple_image_1_estimate[0] + multiple_image_width,
)
multiple_image_1.centre_1 = af.UniformPrior(
    lower_limit=multiple_image_1_estimate[1] - multiple_image_width,
    upper_limit=multiple_image_1_estimate[1] + multiple_image_width,
)

multiple_image_2.centre_0 = af.UniformPrior(
    lower_limit=multiple_image_2_estimate[0] - multiple_image_width,
    upper_limit=multiple_image_2_estimate[0] + multiple_image_width,
)
multiple_image_2.centre_1 = af.UniformPrior(
    lower_limit=multiple_image_2_estimate[1] - multiple_image_width,
    upper_limit=multiple_image_2_estimate[1] + multiple_image_width,
)

multiple_image_3.centre_0 = af.UniformPrior(
    lower_limit=multiple_image_3_estimate[0] - multiple_image_width,
    upper_limit=multiple_image_3_estimate[0] + multiple_image_width,
)
multiple_image_3.centre_1 = af.UniformPrior(
    lower_limit=multiple_image_3_estimate[1] - multiple_image_width,
    upper_limit=multiple_image_3_estimate[1] + multiple_image_width,
)

"""
We now create the model, using the standard model composition API.

Note that we put each multiple image light profile inside the `lens`. This is a bit of a strange syntax, but 
functionally it works.

Future versions of PyAutoLens will have a more intuitive API for this, but for now we have to do it this way!
"""
# Lens

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    multiple_image_0=multiple_image_0,
    multiple_image_1=multiple_image_1,
    multiple_image_2=multiple_image_2,
    multiple_image_3=multiple_image_3,
)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the model only has a `lens` and that it has different components for each multiple image.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).

In the `start_here.py` example 150 live points (`n_live=150`) were used to sample parameter space. For this fit
we have a much more complex parameter space with N=26 free parameters, therefore we use 400 live points to ensure
we thoroughly sample parameter space.
"""
search = af.Nautilus(
    path_prefix=path.join("point_source", "modeling"),
    name="deblending",
    unique_tag=dataset_name,
    n_live=400,
    number_of_cores=4,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Run Time__

For standard light profiles, the log likelihood evaluation time is of order ~0.01 seconds for this dataset.

For linear light profiles, the log likelihood evaluation increases to around ~0.05 seconds per likelihood evaluation.
This is still fast, but it does mean that the fit may take around five times longer to run.

The run time to perform deblending are therefore still relatively fast.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that `intensity` parameters are not inferred by the model-fit.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

The lens and source galaxies appear similar to those in the data, confirming that the `intensity` values inferred by
the inversion process are accurate.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grids.uniform
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
__Point Source__

After the analysis above is complete, the lens model infers the following information: 

- The `centre` parameter of each multiple image `Gaussian` is the (y,x) image-plane coordinate of each lensed point 
  source. These values provide sub-pixel precision, because they fully account for the shape and blending of the PSF. 
  Using these as a `positions` of point-source mass modeling will also produce a more accurate lens model.

- The `intensity` value of each `Gaussian` estimates the flux of each point source. The `fluxes` are typically not
  used in point-source modeling as they are subject to microlensing, but this analysis nevertheless does measure them.
  
 - The lens galaxy's properties are measured via this analysis.

The lensed source image-plane positions, inferred to sub-pixel precision, are printed below and output to a 
`PointDataset` object and .json file.

They can be used as input positions in a point-source model-fit, using an identical API to 
the `point_source/modeling/start_here.ipynb` example, to perform mass modeling of the point source dataset.
"""
