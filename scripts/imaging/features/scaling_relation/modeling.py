"""
Misc: Scaling Relations
=======================

Strong lenses often have many galaxies surrounding the lens galaxy.

For galaxy-scale systems these are often far from the lensed source, meaning they individually contribute little to the
overall lensing but may have a measurable impact when considered collectively.

In group and cluster lenses these objects are often closer to the lensed source and can therefore individually have
a significant impact on the lensing.

In both cases, it is desirable to include these objects in the lens mass model. However, the number of parameters
required to model each galaxy individually can be prohibitively large. For example, with 10 galaxies each modeled
using a `IsothermalSph` profile, the lens model would have 30 parameters!

It is therefore common practice to model the lensing contribution of these galaxies using a scaling relation,
whereby easier to measure properties of the galaxy (e.g. its luminosity, stellar mass, velocity dispersion) are related
to the mass profile's quantities.

The free parameters are now only those related to the scaling relation, for example is normalization and gradient.

__Mass Model And Scaling Relation__

This example shows how to compose a scaling-relation lens model using the dual Pseudo-Isothermal Elliptical (dPIE)
mass distribution introduced in Eliasdottir 2007: https://arxiv.org/abs/0710.5636.

It relates the luminosity of every galaxy to a cut radius (r_cut), a core radius (r_core) and a mass normaliaton b0:

$r_cut = r_cut^* (L/L^*)^{0.5}$

$r_core = r_core^* (L/L^*)^{0.5}$

$b0 = b0^* (L/L^*)^{0.25}$

The free parameters are therefore L^*, r_cut^*, r_core^* and b0^*.

This mass model differs from the `Isothermal` profile used commonly throughout the **PyAutoLens** examples. The dPIE
is more commonly used in strong lens cluster studies where scaling relations are used to model the lensing contribution
of many cluster galaxies.

The API provided in this example is general and can be used to compose any scaling relation mass model (or
light model, or anything else!).

__Centres__

Scaling relations parameterize the mass of each galaxy, but not their centres. If the centres of the galaxies are
treated as free parameters, one again runs into the problem of having too many parameters and a model which
cannot be fitted efficiently.

Scaling relation modeling therefore always inputs the centres of the galaxies as fixed values. In this example, we
use a simulated dataset where the centres of the galaxies are known perfectly.

In a real analysis, one must determine the centres of the galaxies before modeling them with a scaling relation.
There are a number of ways to do this:

 - Use image processing software like Source Extractor (https://sextractor.readthedocs.io/en/latest/).

 - Fit every galaxy individually with a light profile (e.g. an `Sersic`).

 - Use a moment's based analysis of the data.

For certain strong lenses all of the above approaches may be challenging, because the light of each galaxy may be
blended with the lensed source's emission. This may motivate simultaneous fitting of the lensed source and galaxies.

__Redshifts__

In this example all line of sight galaxies are at the same redshift as the lens galaxy, meaning multi-plane lensing
is not used.

If you have redshift information on the line of sight galaxies and some of their redshifts are different to the lens
galaxy, you can easily extend this example below to perform multi-plane lensing.

You would simply define a `redshift_list` and use this to set up the extra `Galaxy` redshifts.

__Extra Galaxies API__

**PyAutoLens** refers to all galaxies surrounded the strong lens as `extra_galaxies`, with the modeling API extended
to model them.

The galaxies (and their parameters) included via a scaling relation are therefore prefixed with `extra_galaxy_` to
distinguish them from the lens galaxy and source galaxy, and in the model they are separate from the `galaxies` and
use their own `extra_galaxies` collection.
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

output = aplt.Output(path=".", format="png")

"""
__Dataset__

First, lets load a strong lens dataset, which is a simulated group scale lens with 2 galaxies surrounding the
lensed source.

These three galaxies will be modeled using a scaling relation.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Centres__

Before composing our scaling relation model, we need to define the centres of the galaxies. 

In this example, we know these centres perfectly from the simulated dataset. In a real analysis, we would have to
determine these centres beforehand (see discussion above).
"""
extra_galaxies_centre_list = [(3.5, 2.5), (-4.4, -5.0)]

"""
We can plot the centres over the strong lens dataset to check that they look like reasonable values.
"""
visuals = aplt.Visuals2D(
    light_profile_centres=al.Grid2DIrregular(values=extra_galaxies_centre_list)
)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=visuals, mat_plot_2d=aplt.MatPlot2D(output=output)
)
dataset_plotter.subplot_dataset()

"""
__Luminosities__

We also need the luminosity of each galaxy, which in this example is the measured property we relate to mass via
the scaling relation.

We again uses the true values of the luminosities from the simulated dataset, but in a real analysis we would have
to determine these luminosities beforehand (see discussion above).

This could be other measured properties, like stellar mass or velocity dispersion.
"""
extra_galaxies_luminosity_list = [0.9, 0.9]

"""
__Scaling Relation__

We now compose our scaling relation models, using **PyAutoFits** relational model API, which works as follows:

- Define the free parameters of the scaling relation using priors (note how the priors below are outside the for loop,
  meaning that every extra galaxy is associated with the same scailng relation prior and therefore parameters).

- For every extra galaxy centre and lumnosity, create a model mass profile (using `af.Model(dPIEPotentialSph)`), where 
  the centre of the mass profile is the extra galaxy centres and its other parameters are set via the scaling relation 
  priors.

- Make each extra galaxy a model galaxy (via `af.Model(Galaxy)`) and associate it with the model mass profile, where the
  redshifts of the extra galaxies are set to the same values as the lens galaxy.
"""
ra_star = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e11)
rs_star = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
b0_star = af.LogUniformPrior(lower_limit=1e5, upper_limit=1e7)
luminosity_star = 1e9

extra_galaxies_list = []

for extra_galaxy_centre, extra_galaxy_luminosity in zip(
    extra_galaxies_centre_list, extra_galaxies_luminosity_list
):
    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = extra_galaxy_centre
    mass.ra = ra_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.rs = rs_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.b0 = b0_star * (extra_galaxy_luminosity / luminosity_star) ** 0.25

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

"""
__Model__

We compose the overall lens model using the normal API.
"""
mask_radius = 9.0

# Lens:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.IsothermalSph)

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
When creating the overall model, we include the extra galaxies as a separate collection of galaxies.

This is not strictly necessary (e.g. if we input them into the `galaxies` attribute of the model the code would still
function correctly).

However, to ensure results are easier to interpret we keep them separate.
"""
model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source)
    + af.Collection(extra_galaxies_list),
)

"""
The `model.info` shows the model we have composed.

The priors and values of parameters that are set via scaling relations can be seen in the printed info.

The number of free parameters is N=16, which breaks down as follows:

 - 4 for the lens galaxy's `SersicSph` bulge.
 - 3 for the lens galaxy's `IsothermalSph` mass.
 - 6 for the source galaxy's `Sersic` bulge.
 - 3 for the scaling relation parameters.

Had we modeled both extra galaxies independently as dPIE profiles, we would of had 6 parameters per extra galaxy, 
giving N=19. Furthermore, by using scaling relations we can add more extra galaxies to the model without increasing the 
number of free parameters. 
"""
print(model.info)

"""
__Model Fit__

We now perform the usual steps to perform a model-fit, to see our scaling relation based fit in action!
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

search = af.Nautilus(
    path_prefix=Path("features"),
    name="scaling_relation",
    unique_tag=dataset_name,
    n_live=150,
    iterations_per_quick_update=10000,
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

This example has shown how to use **PyAutoLens**'s scaling relation API to model a strong lens. 

We have seen how by measuring the centres and luminosities of galaxies (referred to as extra galaxies) surrounding the 
lens galaxy, we can use scaling relations to define their mass profiles. This reduces the number of free parameters in 
the lens model, because we only need to infer the scaling relation parameters, rather than the individual parameters of
each extra galaxy.

The API shown in this script is highly flexible and you should have no problem adapting it use any scaling relation
you wish to use in your own strong lens models! 
"""
