"""
Misc: Add A Profile
===================

**PyAutoLens** support many different mass and light profiles, for modeling the lens and source galaxies in
strong lensing systems.

For your science case, you may need to add your own custom profiles. This could be a new profile which is not
supported by **PyAutoLens**, or a new parameterization of an existing profile. We cover both of these possibilities
in this example script.

We will first explain how to add a new mass profile. This covers the majority of key concepts required for adding
light profiles, which are explained afterwards.

__Source Code__

This example contains URLs to the locations of the source code of the classes used when creating light and mass
profiles.

The example itself is standalone and should by the end allow you to implement a custom profile without diving into
the **PyAutoLens** source code.

We still recommend you take a look to see how things are structured!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import autolens as al

"""
__Example Mass Profile__

The mass profiles available in **PyAutoLens** are actually located in its parent package, **PyAutoGalaxy**: 

 https://github.com/Jammy2211/PyAutoGalaxy

All light and mass profiles are found in the following python package:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/profiles

Mass profiles are in the following package:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/profiles/mass

Lets look at an example mass profile. We'll use the `Isothermal` profile, which is located in the `total` package
because it represents a total (stars + dark matter) mass distribution:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/profiles/mass/total

 https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/mass/total/isothermal.py

For simplicity, a shortened version of the `Isothermal` profile is shown below. 

This has docstrings updated to focus on the key aspects of implementing a new profiles and simplifies the 
inheritance structure of the profile.
"""
from typing import Tuple

import autoarray as aa
from autogalaxy.profiles.mass.total.isothermal import psi_from
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class Isothermal(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
    ):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope = 2.0.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
        )

        self.einstein_radius = einstein_radius
        self.slope = 2.0

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        The input grid of (y,x) coordinates are transformed to a coordinate system centred on the profile centre with
        and rotated based on the position angle defined from its `ell_comps` (this is described fully below).

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore,
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        factor = (
            2.0
            * self.einstein_radius
            * self.axis_ratio
            / np.sqrt(1 - self.axis_ratio**2)
        )

        psi = psi_from(grid=grid, axis_ratio=self.axis_ratio, core_radius=0.0)

        deflection_y = np.arctanh(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio**2), grid[:, 0]), psi)
        )
        deflection_x = np.arctan(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio**2), grid[:, 1]), psi)
        )
        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(factor, np.vstack((deflection_y, deflection_x)).T)
        )


"""
Lets first consider the inheritance structure of the `Isothermal` profile, defined by the first line of the 
profile `class Isothermal(MassProfile):`

I will emphasise again you do not need to fully understand the inheritance structure of the **PyAutoLens** profiles
or the source code structure. The text below is simply to make you aware that all calculations one needs to do
with mass and light profiles is built on a framework of abstract base classes, which your custom profiles will
inherit from.

__Inheritance (MassProfile)__

All mass profiles in **PyAutoLens** inherit from the `MassProfile` abstract base class, which is located in the
following package:

 https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/mass/abstract/abstract.py

This contains generic functions which are useful for any mass profile, which your custom mass profile will inherit.

For example, it includes the function `mass_angular_within_circle_from`, which computes the mass of the profile
within an input circle of radius `radius`.

__Inheritance (GeometryProfile)__

The `MassProfile` class inherits from the `GeometryProfile` abstract base class, which is located here:

 https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/geometry_profiles.py

This contains generic functions which are useful for any elliptical (and spherical) profile, which your custom 
mass profile will again inherit. 

They typically perform coordinate transforms between the profile's elliptical (spherical) coordinate system and the 
input 2D grid of (y,x) coordinates. For example, the function `transformed_to_reference_frame_grid_from` transforms
the (y,x) coordinates to the profile's elliptical coordinate system.

The convention of this calculation is key for ensuring you implement your custom profile correctly. We illustrate
it fully below.

__Inheritance (OperateDeflections)__

Nearly all lensing quantities (e.g. `convergence`, `potential`, `magnification`) can be derived from the deflection
angles of a mass profile. 

Mass profiles therefore also inherit from the `OperateDeflections` abstract base class, which contains numerous 
functions for computing these lensing quantities from the deflection angles. This is located here:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/operate/deflections.py

This means that once you've implemented a deflections angles calculation for your mass profile, you can compute all
lensing quantities from it without having to write any additional code!

__Whats Going On With Those Decorators?__

Before considering how to implement your own `deflections_yx_2d_from` function, lets first consider the 4 decorators
that are used to decorate it in the `Isothermal` profile above.

There are 3 decorators which you don't really need to worry about, all `deflections_yx_2d_from` methods have them
and you will pretty much just copy and paste them into your own profile:

`@aa.grid_dec.to_array`: 
 
Different grids can be input into each mass profile function (e.g. `Grid2D`, `Grid2DIrregular`). Depending on the input 
grid, this changes the structure of the output array. 
 
For example, a `Grid2D` objected in defined on a natively uniform grid, meaning the output should also be uniform structure 
(e.g. an `Array2D`) object. A `Grid2DIrregular` object is defined on an irregular grid, meaning the output should 
also be irregular (e.g. `ArrayIrregular`). This decorator simply ensures the output structure matches the input.

 
`@aa.grid_dec.to_vector_yx`: 

The decorator above handles all structure conversions for scalar quantities (e.g. `convergence`, `potential`). The 
deflection angles are a 2D vector and this decorator simple handles  structure conversions for vector quantities (
another vector quantity which uses this decorator is the `shear`).

For example, if a `Grid2D` is input, the output deflection angles will be an `VectorYX2D` object. If 
a `Grid2DIrregular` is input, the output deflection angles will be an `VectorYX2DIrregular` object.
 
 
`@aa.grid_dec.relocate_to_radial_minimum`: 

For certain mass profiles, if a (y,x) coordinate is numerically (0.0, 0.0) this can lead the code to crash (e.g. a 
divide by zero). 

This decorator relocates any coordinate near (0.0, 0.0) to a small offset from (0.0, 0.0), ensuring this does not 
happen. 

The size of the offset is provided in the `autolens_workspace/config/grids.yaml` file. You should add your mass 
profile to this file, and set the offset to a value that ensures your code does not crash.
 
__Transform Decorator__
  
The fourth decorator is the `@aa.grid_dec.transform` decorator. This one we will have a closer look at, as it
will influence how you implement your mass profile functions.

The `transform` decorator is used to transform the input grid of (y,x) coordinates to the mass profile's elliptical 
coordinate system. It does this by calling the function `transform_grid_2d_to_reference_frame`, which I have
provided below for convenience:
"""


def transform_grid_2d_to_reference_frame(
    grid_2d: np.ndarray, centre: Tuple[float, float], angle: float
) -> np.ndarray:
    """
    Transform a 2D grid of (y,x) coordinates to a new reference frame.

    This transformation includes:

     1) A translation to a new (y,x) centre value, by subtracting the centre from every coordinate on the grid.
     2) A rotation of the grid around this new centre, which is performed clockwise from an input angle.

    Parameters
    ----------
    grid
        The 2d grid of (y, x) coordinates which are transformed to a new reference frame.
    """
    shifted_grid_2d = np.subtract(grid_2d, centre)
    radius = np.sqrt(np.sum(shifted_grid_2d**2.0, 1))
    theta_coordinate_to_profile = np.arctan2(
        shifted_grid_2d[:, 0], shifted_grid_2d[:, 1]
    ) - np.radians(angle)
    return np.vstack(
        radius
        * (np.sin(theta_coordinate_to_profile), np.cos(theta_coordinate_to_profile))
    ).T


"""
A simple example of this function is shifting a grid to a mass profile's centre (which simply subtracts the centre
coordinates from every coordinate on the grid):
"""
grid = al.Grid2DIrregular(values=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])

mass_profile_centre = (0.5, 0.5)

transformed_grid = transform_grid_2d_to_reference_frame(
    grid_2d=grid, centre=mass_profile_centre, angle=0.0
)

print(f"Grid Coordinates Before: {grid}")
print(f"Grid Coordinates After: {transformed_grid}")

"""
The `angle` input is the rotation angle of the mass profile's ellipse counter-clockwise from the positive x-axis.

It is computed from the `ell_comps` of the mass profile, which are the elliptical components of the mass profile's
ellipse.
"""


def axis_ratio_and_angle_from(ell_comps):
    """
    Convert the ellipitical components e1 and e2 to an axis ratio (0.0 > q > 1.0) and rotation position angle
    defined counter clockwise from the positive x-axis(0.0 > angle > 180) to .

    Parameters
    ----------
    ell_comps : (float, float)
        The first and second ellipticity components of the elliptical coordinate system.
    """
    angle = np.arctan2(ell_comps[0], ell_comps[1]) / 2
    angle *= 180.0 / np.pi
    fac = np.sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2)
    if fac > 0.999:
        fac = 0.999  # avoid unphysical solution
    # if fac > 1: print('unphysical e1,e2')
    axis_ratio = (1 - fac) / (1 + fac)
    return axis_ratio, angle


mass_profile_ell_comps = (0.5, 0.5)
mass_profile_angle = axis_ratio_and_angle_from(ell_comps=mass_profile_ell_comps)[1]

print(f"\nMass Profile Angle (degrees) {mass_profile_angle}")

transformed_grid = transform_grid_2d_to_reference_frame(
    grid_2d=grid, centre=mass_profile_centre, angle=mass_profile_angle
)

print(f"Grid Coordinates Before: {grid}")
print(f"Grid Coordinates After: {transformed_grid}")

"""
The `@aa.grid_dec.transform` packages all the above calculations up and uses the mass profile `centre` and `ell_comps` 
to perform them before your function is called.

The class below demonstrates this.
"""

from autogalaxy.profiles.geometry_profiles import EllProfile


class ExampleMass(EllProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(centre=centre, ell_comps=ell_comps)

    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.Grid2D, **kwargs):
        print(
            f"\n Grid In Deflections After Transform "
            f"Which is Same As Transformed Grid Above: {grid}"
        )


mass = ExampleMass(centre=(0.5, 0.5), ell_comps=(0.5, 0.5))
mass.deflections_yx_2d_from(grid=grid)

"""
__Do I Rotate Back?__

The `aa.grid_dec.transform` decorator does not rotate the deflection angles back to the original reference frame
(a function in the `GeometryProfile` class exists to do this.

Deflection angles are typically applied in the same reference frame as the mass profile, so this is not necessary.
You should ensure your mass profile is implemented in a way that accounts for any rotation of the deflection angles
after they are calculated.

__Lens Modeling__

**PyAutoLens** assumes that all input parameters of the mass profile (e.g. those listed in the `__init__` constructor 
are free parameters that for lens modeling can be fitted for by the non-linear search.

If a parameter in the `__init__` constructor is a float (e.g. the `einstein_radius` of the `Isothermal` profile) it 
will associated it with one free parameter. If it is a tuple of floats (e.g. the `centre` of the `Isothermal`) it will
associate each value with a free parameter.

Below, we compose the `Isothermal` class as a model and it becomes a model with 5 free parameters based on
those in its `__init__` constructor:
"""
import autofit as af


class LensModelExample:
    def __init__(
        self,  # <-- PyAutoLens Assumes these input parameters are free.
        centre: Tuple[float, float] = (
            0.0,
            0.0,
        ),  # <-- The `centre` has two free parameters because its a tuple.
        ell_comps: Tuple[float, float] = (
            0.0,
            0.0,
        ),  # <-- The `ell_comps` also has two free parameters.
        einstein_radius: float = 1.0,  # <-- The `einstein_radius` just one because its a float.
        your_parameter_here: float = 2.0,  # <-- Add whatever parameters you need!
    ):
        pass


"""
The `Isothermal` class was shown above and had 3 `__init__` parameters, its `centre`, `ell_comps` and `einstein_radius`.

If we compose the `Isothermal` class as a model, it becomes a model with these 5 free parameters.
"""
mass = af.Model(al.mp.Isothermal)

print(mass.info)

"""
By default, **PyAutoLens** will not know what priors to place on your profile's parameters. 

The simplest way to do this is to manually specify them yourself after creating the model, as shown below.
"""
mass.centre.centre_0 = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)

"""
Therefore, provided you follow the same convention your mass profile will fully support lens modeling without any
additional code!

__Lens Modeling Configs__

In most **PyAutoLens** examples, you will notice we compose models without manually specifying priors. This is because
**PyAutoLens** uses configs to set up priors on all of the model's parameters. 

These configs are stored in the `config/priors/mass/<mass_profile_package>/<mass_profile_module>.yaml`, for 
example `config/priors/mass/total/isothermal.yaml`.

If you add your mass profile to the **PyAutoLens** source code you can add a config file for it to this folder and
**PyAutoLens** will automatically use it to set up the priors on your mass profile.

You should also add your mass profile and its parameters to the `config/notation.yaml` file, so that **PyAutoLens**
knows how to label your mass profile in plots.

__Deflections__

We are therefore ready to implement a mass profile, and the best place to start is the `deflections_2d_from` function.

In fact, this is the only function you need to implement in order for lens modeling to work. This is because pretty much
all lensing calculations can be computed from the deflection angles.

However, we recommend you also implement analytic functions for the `convergence` and `potential` of your mass profile.
They are often used for separate calculations outside of lens modeling and are commonly visualized in plots.

The template below is a good starting point for your mass profile and explains what functions you need to implement
and what are optional.
"""


class TemplateMass(EllProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        # Your parameters here.
    ):
        super().__init__(centre=centre, ell_comps=ell_comps)

        # Note that for a Spherical profile, which does not have an `ell_comps` parameter,
        # you can remove it from the __init__ constructor and pass (0.0, 0.0) below, e.g.

        # super().__init__(centre=centre, ell_comps=(0.0, 0.0))

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        REQUIRED: The function is key for all lensing calculations and must be implemented.
        """
        pass

    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """
        RECOMMENDED: The convergence is used for visualization and inspecting properties of the mass profile.
        """
        pass

    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        """
        RECOMMENDED: The gravitational potential is used for visualization and inspecting properties of the mass
        profile.
        """
        pass

    def convergence_func(self, grid_radius: float) -> float:
        """
        Optional: A 1D function which returns the convergence at a given 1D coordinate (e.g. radius). This is used
        for computing integrated mass quantities.
        """
        pass

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        _eta_u = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        return (
            (_eta_u / u)
            * ((3.0 - slope) * _eta_u) ** -1.0
            * _eta_u ** (3.0 - slope)
            / ((1 - (1 - axis_ratio**2) * u) ** 0.5)
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def shear_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        OPTIONAL: Shears are used for weak lensing calculations and inspection properties of the mass profile.

        Shears can reliably be calculated via methods inherited from the `OperateDeflections` class. Providing an
        analytic calculation here can speed this up and provide more accurate results.
        """
        pass


"""
__Spherical Template__

radiaul_grid
removal of ell_comps

__Physical Profiles__

Show how to wrap existing profiles with physical units?

__Light Profiles__

Pretty much the same but need to add text.
"""
