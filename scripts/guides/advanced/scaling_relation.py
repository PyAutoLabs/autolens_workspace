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

It relates the luminosity of every galaxy to a cut radius (r_cut), a core radius (r_core) and a velocity dispersion
(sigma):

$r_cut = r_cut^* (L/L^*)^{0.5}$

$r_core = r_core^* (L/L^*)^{0.5}$

$\sigma = \sigma^* (L/L^*)^{0.25}$

The free parameters are therefore L^*, r_cut^*, r_core^* and \sigma^*.

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

 - Fit every galaxy individually with a parametric light profile (e.g. an `Sersic`).

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

from os import path
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
dataset_name = "lens_x3__source_x1"
dataset_path = path.join("dataset", "group", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
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
__dPIE__

The dPIE is not yet implemented in the source code so I am copy and pasting it in here below.

This part of the example will be removed, once its in the source code.
"""
from typing import Tuple
import numpy as np

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class dPIESph(MassProfile):
    """
    The dual Pseudo-Isothermal Elliptical mass distribution introduced in
    Eliasdottir 2007: https://arxiv.org/abs/0710.5636

    This version is without ellipticity, so perhaps the "E" is a misnomer.

    Corresponds to a projected density profile that looks like:

        \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                      (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

    (c.f. Eliasdottir '07 eqn. A3)

    In this parameterization, ra and rs are the scale radii above in angular
    units (arcsec). The parameter is \\Sigma_0 / \\Sigma_crit.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        sigma_scale: float = 0.1,
    ):
        super(MassProfile, self).__init__(centre, (0.0, 0.0))
        self.ra = ra
        self.rs = rs
        self.sigma_scale = sigma_scale

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        radii = np.sqrt(xoff**2 + yoff**2)

        r_ra = radii / self.ra
        r_rs = radii / self.rs
        # c.f. Eliasdottir '07 eq. A20
        f = r_ra / (1 + np.sqrt(1 + r_ra * r_ra)) - r_rs / (
            1 + np.sqrt(1 + r_rs * r_rs)
        )

        ra, rs = self.ra, self.rs
        # c.f. Eliasdottir '07 eq. A19
        # magnitude of deflection
        alpha = 2 * self.sigma_scale * ra * rs / (rs - ra) * f

        # now we decompose the deflection into y/x components
        defl_y = alpha * yoff / radii
        defl_x = alpha * xoff / radii
        return aa.Grid2DIrregular.from_yx_1d(defl_y, defl_x)

    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        # already transformed to center on profile centre so this works
        radsq = grid[:, 0] ** 2 + grid[:, 1] ** 2
        a, s = self.ra, self.rs
        # c.f. Eliasdottir '07 eqn (A3)
        return (
            self.sigma_scale
            * (a * s)
            / (s - a)
            * (1 / np.sqrt(a**2 + radsq) - 1 / np.sqrt(s**2 + radsq))
        )

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        return np.zeros(shape=grid.shape[0])


"""
__Scaling Relation__

We now compose our scaling relation models, using **PyAutoFits** relational model API, which works as follows:

- Define the free parameters of the scaling relation using priors (note how the priors below are outside the for loop,
  meaning that every extra galaxy is associated with the same scailng relation prior and therefore parameters).

- For every extra galaxy centre and lumnosity, create a model mass profile (using `af.Model(dPIESph)`), where the centre
  of the mass profile is the extra galaxy centres and its other parameters are set via the scaling relation priors.
  
- Make each extra galaxy a model galaxy (via `af.Model(Galaxy)`) and associate it with the model mass profile, where the
  redshifts of the extra galaxies are set to the same values as the lens galaxy.
"""
ra_star = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e11)
rs_star = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
sigma_star = af.LogUniformPrior(lower_limit=1e5, upper_limit=1e7)
luminosity_star = 1e9

extra_galaxies_list = []

for extra_galaxy_centre, extra_galaxy_luminosity in zip(
    extra_galaxies_centre_list, extra_galaxies_luminosity_list
):
    mass = af.Model(dPIESph)
    mass.centre = extra_galaxy_centre
    mass.ra = ra_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.rs = rs_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.sigma_scale = sigma_star * (extra_galaxy_luminosity / luminosity_star) ** 0.25

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

"""
__Model__

We compose the overall lens model using the normal API.
"""

# Lens:

bulge = af.Model(al.lp.SersicSph)

mass = af.Model(al.mp.IsothermalSph)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicCore)

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
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=9.0
)

dataset = dataset.apply_mask(mask=mask)

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="scaling_relation",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=4,
    iterations_per_update=10000,
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
