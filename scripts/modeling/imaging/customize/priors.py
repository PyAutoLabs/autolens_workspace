"""
Customize: Priors
=================

This example demonstrates how to customize the priors of a model-fit, for example if you are modeling a lens where
certain parameters are known before.

__Advantages__

If you are having difficulty fitting an accurate lens model, but have a previous estimate of the model from another
analysis or strong intiution from inspecting the data what values certain parameters should be, customizing the priors
of the model-fit may ensure you infer an accurate model-fit.

Custom Priors result in a computationally faster model-fit, provided the priors are sufficiently tight.

__Disadvantages__

The priors on your model determine the errors you infer. Overly tight priors may lead to over-confidence in the
inferred parameters.

If you are using your intuition to customize the priors, the priors you manually input may not be accurate.

__Start Point__

The `autolens_workspace/*/modeling/imaging/customize/start_point.ipynb` shows an alternative API, which
customizes where the non-linear search starts its search of parameter space.

This cannot be used for a nested sampling method like `Nautilus` (whose parameter space search is dictated by priors)
but can be used for the maximum likelihood estimator / MCMC methods PyAutoGalaxy supports.

The benefit of the starting point API is that one can tell the non-linear search where to look in parameter space
(ensuring a fast and robust fit) but retain uniformative priors which will not lead to over-confident errors.

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

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

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

The model-fit requires a `Mask2D` defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
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

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a linear parametric `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
 
__Prior Customization__
 
We customize the parameter of every prior to values near the true values, using the following priors:

- UniformPrior: The values of a parameter are randomly drawn between a `lower_limit` and `upper_limit`. For example,
the effective radius of ellipitical Sersic profiles typically assumes a uniform prior between 0.0" and 30.0".

- LogUniformPrior: Like a `UniformPrior` this randomly draws values between a `limit_limit` and `upper_limit`, but the
values are drawn from a distribution with base 10. This is used for the `intensity` of a light profile, as the
luminosity of galaxies follows a log10 distribution.

- GaussianPrior: The values of a parameter are randomly drawn from a Gaussian distribution with a `mean` and width
 `sigma`. For example, the $y$ and $x$ centre values in a light profile typically assume a mean of 0.0" and a
 sigma of 0.3", indicating that we expect the profile centre to be located near the centre of the image.
 
The API below can easily be adapted to customize the priors on a `disk` component, for example by simply making it
a `Model`. 
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

mass.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
mass.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
mass.einstein_radius = af.UniformPrior(lower_limit=1.4, upper_limit=1.8)


shear = af.Model(al.mp.ExternalShear)

shear.gamma_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
shear.gamma_2 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)

lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

# Source:

bulge = af.Model(al.lp_linear.SersicCore)

bulge.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.effective_radius = af.UniformPrior(lower_limit=0.05, upper_limit=0.15)
bulge.sersic_index = af.GaussianPrior(mean=1.0, sigma=0.5)


source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format, including the customized priors above.
"""
print(model.info)

"""
The info of individual model components can also be printed.
"""
print(bulge.info)

"""
__Alternative API__

The priors can also be customized after the `lens` and `source` model object are created instead.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)


lens.mass.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
lens.mass.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
lens.mass.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
lens.mass.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
lens.mass.einstein_radius = af.UniformPrior(lower_limit=1.4, upper_limit=1.8)


lens.shear.gamma_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
lens.shear.gamma_2 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)


source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

source.bulge.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
source.bulge.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
source.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
source.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
source.bulge.effective_radius = af.UniformPrior(lower_limit=0.05, upper_limit=0.15)
source.bulge.sersic_index = af.GaussianPrior(mean=1.0, sigma=0.5)

"""
The `info` attribute shows the model in a readable format, including the customized priors above.
"""
print(lens.info)
print(source.info)

"""
We could also customize the priors after the creation of the whole model.

Note that you can mix and match any of the API's above, and different styles will lead to concise and readable
code in different circumstances.
"""
# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


model.galaxies.lens.mass.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
model.galaxies.lens.mass.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
model.galaxies.lens.mass.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
model.galaxies.lens.mass.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
model.galaxies.lens.mass.einstein_radius = af.UniformPrior(
    lower_limit=1.4, upper_limit=1.8
)


model.galaxies.lens.shear.gamma_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
model.galaxies.lens.shear.gamma_2 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)


model.galaxies.source.bulge.centre_0 = af.UniformPrior(
    lower_limit=-0.1, upper_limit=0.1
)
model.galaxies.source.bulge.centre_1 = af.UniformPrior(
    lower_limit=-0.1, upper_limit=0.1
)
model.galaxies.source.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
model.galaxies.source.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
model.galaxies.source.bulge.effective_radius = af.UniformPrior(
    lower_limit=0.05, upper_limit=0.15
)
model.galaxies.source.bulge.sersic_index = af.GaussianPrior(mean=1.0, sigma=0.5)

"""
The `info` attribute shows the model in a readable format, including the customized priors above.
"""
print(model.info)

"""
__Search + Analysis + Model-Fit__

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "customize"),
    name="priors",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)


analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
By inspecting the `model.info` file of this fit we can confirm the above priors were used. 
"""
