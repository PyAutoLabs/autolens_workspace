"""
Modeling Features: Extra Galaxies
=================================


There may be extra galaxies nearby the lens and source galaxies, whose mass may contribute to the ray-tracing and
lens model.

They may also emit  luminous emission, but for interferometer datasets they are rarely detected at the long wavelengths
probed. The extra galaxies examples for interferometer throughout the workspace therefore do not include their light
(unlike the CCD image examples).

This example shows how to perform lens modeling which accounts for the mass of these extra galaxies. The centres of
each galaxy (e.g. their brightest pixels observed from imaging data) are used as the centre of the mass profiles of
these galaxies, in order to reduce model complexity.

__Data Preparation__

To perform modeling which accounts for extra galaxies, a list of the centre of each extra galaxy are used to set up
the model-fit. For the example dataset used here, these tasks have already been performed and the
metadata (`mask_extra_galaxies.fits` and `extra_galaxies_centres.json` are already included in results folder.

The tutorial `autolens_workspace/*/imaging/data_preparation/optional/extra_galaxies_centres.py`
describes how to create these centres and output them to a `.json` file. You will need to use imaging data
to do not, as interferometer data rarely detects the light of these extra galaxies. If this data is not available,
you probably dont have any evidence of there being multiple galaxies in the system!

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

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
__Mask__

Define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `simple` from .fits files, which we will fit 
with the lens model.

We load the `extra_galaxies` dataset, which includes two extra galaxies either side of the main lens galaxy.

Load and plot the strong lens dataset `extra_galaxies` via .fits files.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()


"""
We do not perform a model-fit using this dataset, as using a mask like this requires that we use a pixelization
to fit the lensed source, which you may not be familiar with yet.

In the `features/pixelization` example we perform a fit using this noise scaling scheme and a pixelization,
so check this out if you are interested in how to do this.

__Extra Galaxies Dataset__

We are now going to model the dataset with extra galaxies included in the model, where these galaxies include
the mass profiles of the extra galaxies.

__Extra Galaxies Centres__

To set up a lens model including each extra galaxy with a mass profile, we input manually the centres of
the extra galaxies.

In principle, a lens model including the extra galaxies could be composed without these centres. For example, if 
there were two extra galaxies in the data, we could simply add two additional mass profiles into the model. 
The modeling API does support this, but we will not use it in this example.

This is because models where the extra galaxies have free centres are often too complex to fit. It is likely the fit 
will infer an inaccurate lens model and local maxima, because the parameter space is too complex.

For example, a common problem is that one of the extra galaxy light profiles intended to model a nearby galaxy instead 
fits one of the lensed source's multiple images. Alternatively, an extra galaxy's mass profile may recenter itself and 
act as part of the main lens galaxy's mass distribution.

Therefore, when modeling extra galaxies we input the centre of each, in order to fix their mass profile 
centres or set up priors centre around these values.

The `data_preparation` tutorial `autolens_workspace/*/imaging/data_preparation/examples/optional/extra_galaxies_centres.py` 
describes how to create these centres. Using this script they have been output to the `.json` file we load below.
"""
extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "extra_galaxies_centres.json"))
)

print(extra_galaxies_centres)

"""
__Model__ 

Perform the normal steps to set up the main model of the lens galaxy and source.

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)
mass = af.Model(al.mp.Isothermal)
lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

"""
__Extra Galaxies Model__ 

We now use the modeling API to create the model for the extra galaxies.

Currently, the extra galaxies API require that the centres of the mass profiles are fixed to the input centres
(but the other parameters of the mass profiles remain free). 

Therefore, in this example fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].

 - The source galaxy's light is a Multi Gaussian Expansion [4 parameters].

 - Each extra galaxy's total mass distribution is a `IsothermalSph` profile with fixed 
 centre [2 extra galaxies x 1 parameters = 2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.

Extra galaxy mass profiles often to go unphysically high `einstein_radius` values, degrading the fit. The 
`einstein_radius` parameter is set a `UniformPrior` with an upper limit of 0.1" to prevent this.
"""
# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    # Extra Galaxy

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), extra_galaxies=extra_galaxies
)

"""
The `info` attribute confirms the model includes extra galaxies that we defined above.
"""
print(model.info)

"""
__Search + Analysis__ 

The code below performs the normal steps to set up a model-fit.

Given the extra model parameters due to the extra galaxies, we increase the number of live points to 200.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "features",
    name="extra_galaxies_model",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=20000,
)

analysis = al.AnalysisInterferometer(dataset=dataset)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM 
required by a model.

Adding extra galaxies with just mass profiles has a negligible effect on VRAM, because mass profiles are fast to
compute and do not require large images to be stored in VRAM.

__Run Time__

Adding extra galaxies to the model increases the likelihood evaluation times, because their mass profiles need their 
deflection angles computed. These calculations are pretty fast, so only a small increase in time is expected.

The bigger hit on run time is due to the extra free parameters, 1 `einstein_radius` per extra galaxy for its mass. 
This increases the dimensionality  of non-linear parameter space.  This means Nautilus takes longer to converge on 
the highest likelihood regions of  parameter space.

The Source, Light and Mass (SLaM) pipelines support extra galaxies but in a way that reduces the number of free
parameters they add to the model. This is described in the `slam` examples. The `group` package, which models systems
with 10+ extra galaxies, introduces even more clever parameterizations which add 0 free parameters per extra galaxy,
so if your model has many extra galaxies you should check out the `group` package.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

By plotting the maximum log likelihood `FitInterferometer` object we can confirm the extra galaxies contribute to the fit.
"""
fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

These examples show how the results API can be extended to investigate extra galaxies in the results.

__Scaling Relations__

The modeling API has full support for composing the extra galaxies such that their mass follows scaling
relations. For example, you could assume that the mass of the extra galaxies is related to their luminosity via a
constant mass-to-light ratio.

This is documented in the `autolens_workspace/*/imaging/features/scaling_relation` example.

__Wrap Up__

The extra galaxies API makes it straight forward for us to model galaxy-scale strong lenses with additional components
for mass of nearby objects.

The `autolens_workspace` includes a `group` package, for modeling group scale strong lenses which have multiple lens 
galaxies. When you should use the extra galaxies API as shown here, and when you should use the group package? 

The distinction is as follows:

 - A galaxy scale lens is a system which can be modeled to a high level of accuracy using a single light and mass 
 distribution for the main lens galaxy. Including additional galaxies in the model via the extra galaxies API makes small 
 improvements on the lens model, but a good fit is possible without them. 

 - A group scale lens is a system which cannot be modeled to a high level of accuracy using a single light and mass 
 distribution. Defining a 'main' lens galaxy is unclear and two or more main lens galaxies are required to fit an 
 accurate model. 

The `group` package also uses the extra galaxies API for model composition, but does so to compose and fit more 
complex lens models.
"""
