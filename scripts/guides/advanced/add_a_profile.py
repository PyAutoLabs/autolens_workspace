"""
Misc: Add A Profile
===================

**PyAutoLens** supports a wide range of mass and light profiles for modelling the
lens and source galaxies in strong gravitational lensing systems.

For some science cases, it may be necessary to define custom profiles. This could
involve implementing a new profile that is not currently supported by
**PyAutoLens**, or introducing a new parameterization of an existing profile. Both
of these possibilities are covered in this example tutorial.

We begin by explaining how to add a new _mass profile_, as this introduces the core
concepts required for defining custom profiles in **PyAutoLens**. These concepts
are then applied to show how custom _light profiles_ can be implemented.

__Source Code__

This example includes direct links to the source code of the classes used to define
mass and light profiles, allowing you to see exactly how they are implemented.

The tutorial is fully standalone and, by the end, should enable you to implement a
custom profile without needing to dive deeply into the **PyAutoLens** codebase.
That said, we still recommend exploring the source code to better understand how
everything fits together.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import autolens as al

"""
__Example Mass Profile__

The mass profiles available in **PyAutoLens** are located in its parent package, **PyAutoGalaxy**: 

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
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        The input grid of (y,x) coordinates are transformed to a coordinate system centred on the profile centre with
        and rotated based on the position angle defined from its `ell_comps` (this is described fully below).

        The numerical backend can be selected via the ``xp`` argument, allowing this
        method to be used with both NumPy and JAX (e.g. inside ``jax.jit``-compiled
        code). This is described fully later in this example.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        xp
            The numerical backend to use, either `numpy` or `jax.numpy`.
        """

        factor = (
            2.0
            * self.einstein_radius_rescaled(xp)
            * self.axis_ratio(xp)
            / xp.sqrt(1 - self.axis_ratio(xp) ** 2)
        )

        psi = psi_from(
            grid=grid, axis_ratio=self.axis_ratio(xp), core_radius=0.0, xp=xp
        )

        deflection_y = xp.arctanh(
            xp.divide(
                xp.multiply(xp.sqrt(1 - self.axis_ratio(xp) ** 2), grid.array[:, 0]),
                psi,
            )
        )
        deflection_x = xp.arctan(
            xp.divide(
                xp.multiply(xp.sqrt(1 - self.axis_ratio(xp) ** 2), grid.array[:, 1]),
                psi,
            )
        )
        return self.rotated_grid_from_reference_frame_from(
            grid=xp.multiply(factor, xp.vstack((deflection_y, deflection_x)).T),
            xp=xp,
            **kwargs,
        )


"""
__JAX, Numpy and xp__

Throughout this tutorial, and in the profile functions above, you will see functions and methods that accept an
argument called ``xp``. The default input value above is ``xp=np``, which sets it to the standard NumPy library,
imported using the statement ``import numpy as np``. This means all arithmetic operations use NumPy in a way
you are likely familiar with.

However, to enable mass profile calculations to run on a GPU, a library called JAX is used which mirrors the
NumPy API. Conventionally, this is imported as ``jnp`` using the statement ``import jax.numpy as jnp``. When the
source code is running in JAX mode, the input ``xp`` to the functions above will be the ``jnp`` library instead
of ``np``. This is why mass profiles support the ``xp`` argument and API: they need to be able to run using
either NumPy or JAX.

The PyAutoLens source code runs in pure NumPy by default, where ``xp`` is always set to ``np``. This only
changes if you manually call a function passing ``xp=jnp``, or when certain high-level objects, such as the
``Analysis`` class, are used. These objects automatically set ``xp=jnp`` when a likelihood is evaluated for
lens modelling.

Your final mass profile should therefore use the ``xp`` API throughout, ensuring compatibility with both NumPy
and JAX and allowing it to work seamlessly with the PyAutoLens source code. You may find it easier to first
write your functions in pure NumPy (which you are likely most familiar with), and then convert them to use the
``xp`` API and test them with JAX afterwards. While using ``xp`` makes the API slightly more verbose, it is a
small price to pay for the significant speed-ups available when running JAX on a GPU.

__Inheritance Structure__

Let us next consider the inheritance structure of the ``Isothermal`` profile,
defined by the class declaration::

    class Isothermal(MassProfile):

In Python, inheritance means that a class can reuse and extend the behaviour of
another class. By inheriting from ``MassProfile``, the ``Isothermal`` profile
automatically has access to all methods and attributes defined in
``MassProfile``. This allows ``Isothermal`` to make use of shared functionality
(such as common calculations and interfaces) without reimplementing it.

The key mechanism used to enable this inheritance is the ``super`` function. For
example, in the ``Isothermal`` initializer we see::

    super().__init__(
        centre=centre,
        ell_comps=ell_comps,
    )

This line calls the ``__init__`` method of the parent ``MassProfile`` class,
ensuring that all required base-class setup is performed before adding any
``Isothermal``-specific behaviour.

It is important to emphasize that you do not need to fully understand the full
inheritance structure of the **PyAutoLens** profiles or the layout of the source
code to define your own custom profiles. This discussion is included simply to
highlight that all calculations involving mass and light profiles are built on
a set of abstract base classes, which your custom profiles will automatically
inherit from.

__Inheritance (MassProfile)__

All mass profiles in **PyAutoLens** inherit from the `MassProfile` abstract base class, which is located in the
following package:

 https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/mass/abstract/abstract.py

This contains functions which are useful for any mass profile, which your custom mass profile will inherit.

For example, it includes the function `mass_angular_within_circle_from`, which computes the mass of the profile
within an input circle of radius `radius`.

__Inheritance (GeometryProfile)__

The `MassProfile` class inherits from the `GeometryProfile` abstract base class, which is located here:

 https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/geometry_profiles.py

This contains functions which are useful for any elliptical (and spherical) profile, which your custom 
mass profile will again inherit (e.g. `radial_grid_from`).

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

A decorator in Python is a special syntax used to modify or extend the behaviour of a function or method without 
changing its implementation. It is denoted by the `@decorator_name` syntax placed above the function definition,
with two decorators shown in the example above: `@aa.grid_dec.to_vector_yx` and `@aa.grid_dec.transform`. Lets
now consider what these do.

__Data Structure Decorators__

Different grids can be input into each mass profile function (e.g. `Grid2D`, `Grid2DIrregular`). Depending on the input 
grid, this changes the structure of the output array. 

For example, if a `Grid2D` is input, which is defined on a uniform grid of 2D coordinates, the output deflection angles
are also defined on a uniform grid and are returned as a `VectorYX2D` object. If a `Grid2DIrregular` is input, 
which is defined on an irregular grid of 2D coordinates, the output deflection angles are also defined on an irregular 
grid and are returned as a `VectorYX2DIrregular` object.

The `@aa.grid_dec.to_vector_yx` decorator handles this structure conversion for vector quantities, such that the output
vector structure matches the input grid structure.

The function `deflections_yx_2d_from` returns 2D vectors, but other mass profile methods, like `convergence_2d_from` and
`potential_2d_from`, return scalar quantities. These methods use the `@aa.grid_dec.to_array` decorator, which behaves
analogously to the `@aa.grid_dec.to_vector_yx` decorator but for scalar quantities (e.g. for an input `Grid2D`, the output
is an `Array2D` object, for an input `Grid2DIrregular`, the output is an `ArrayIrregular` object).

For your custom mass profile, you basically just need to copy and paste these decorators above your mass profile 
functions and not worry about them any further.
 
__Transform Decorator__
  
The second decorator is the `@aa.grid_dec.transform` decorator. This one we will have a closer look at, as it
will influence how you implement your mass profile functions.

The `transform` decorator is used to transform the input grid of (y,x) coordinates to the mass profile's elliptical 
coordinate system. It does this by calling the function `transform_grid_2d_to_reference_frame`, which I have
provided below for convenience:
"""


def transform_grid_2d_to_reference_frame(
    grid_2d: np.ndarray, centre: Tuple[float, float], angle: float, xp=np
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

    shifted_grid_2d = grid_2d - xp.array(centre)

    radius = xp.sqrt(xp.sum(xp.square(shifted_grid_2d), axis=1))
    theta_coordinate_to_profile = xp.arctan2(
        shifted_grid_2d[:, 0], shifted_grid_2d[:, 1]
    ) - xp.radians(angle)

    return xp.vstack(
        [
            radius * xp.sin(theta_coordinate_to_profile),
            radius * xp.cos(theta_coordinate_to_profile),
        ]
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


def axis_ratio_and_angle_from(
    ell_comps: Tuple[float, float], xp=np
) -> Tuple[float, float]:
    """
    Returns the axis-ratio and position angle in degrees (-45 < angle < 135.0) from input elliptical components e1
    and e2 of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    The axis-ratio and angle are therefore given by:

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the axis-ratio and angle in degrees.

    An additional check is performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `ell_comps` the angle does not jump from one boundary to another (e.g. without this check
    certain values of `ell_comps` return -1.0 degrees and others 179.0 degrees). This ensures that when error
    estimates are computed from samples of a lens model via marginalization, the calculation is not biased by the
    angle jumping between these two values.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    angle = xp.arctan2(ell_comps[0], ell_comps[1]) / 2
    angle *= 180.0 / xp.pi

    angle = xp.where(angle < -45, angle + 180, angle)

    fac = xp.sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2)
    if xp.__name__.startswith("jax"):
        import jax

        fac = jax.lax.min(fac, 0.999)
    else:  # NumPy
        fac = np.minimum(fac, 0.999)

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
    def deflections_yx_2d_from(self, grid: aa.Grid2D, xp=np, **kwargs):
        print(
            f"\n Grid In Deflections After Transform "
            f"Which is Same As Transformed Grid Above: {grid}"
        )


mass = ExampleMass(centre=(0.5, 0.5), ell_comps=(0.5, 0.5))
mass.deflections_yx_2d_from(grid=grid)

"""
__Do I Rotate Back?__

The ``aa.grid_dec.transform`` decorator does not rotate the deflection angles back to the original reference frame.

This is because deflection angles are typically applied in the same reference frame as the mass profile itself, 
making such a rotation unnecessary. When implementing a custom mass profile, you should therefore ensure that any
required rotation of the deflection angles is correctly accounted for after they are calculated.

__Lens Modeling__

**PyAutoLens** assumes that all input parameters of a mass profile (for example,
those listed in its ``__init__`` constructor) are free parameters that can be
fitted during lens modelling using a non-linear search.

If a parameter in the ``__init__`` constructor is a float (e.g. the
``einstein_radius`` of the ``Isothermal`` profile), it is treated as a single
free parameter. If a parameter is a tuple of floats (e.g. the ``centre`` of the
``Isothermal`` profile), each element of the tuple is treated as a separate free
parameter.

We demonstrate this behaviour using a simple example mass profile defined
below. We compose it as a model using ``af.Model`` and print its ``info``,
which summarizes its free parameters and shows that no priors have yet been
assigned.
"""
import autofit as af
from typing import Tuple


class LensModelExample:
    def __init__(
        self,  # <-- **PyAutoLens** assumes these input parameters are free.
        centre: Tuple[float, float] = (
            0.0,
            0.0,
        ),  # <-- Two free parameters because this is a tuple.
        ell_comps: Tuple[float, float] = (
            0.0,
            0.0,
        ),  # <-- Also two free parameters.
        einstein_radius: float = 1.0,  # <-- A single free parameter.
        your_parameter_here: float = 2.0,  # <-- Add any custom parameters you need.
    ):
        pass


lens_model_example = af.Model(LensModelExample)

print(lens_model_example.info)

"""
For this example model, we can manually assign priors to its parameters as shown
below.
"""
lens_model_example.centre.centre_0 = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
lens_model_example.centre.centre_1 = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
lens_model_example.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
lens_model_example.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
lens_model_example.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)
lens_model_example.your_parameter_here = af.UniformPrior(
    lower_limit=0.0, upper_limit=5.0
)

print(lens_model_example.info)

"""
The exact same API applies to the ``Isothermal`` class defined above, which has
three ``__init__`` parameters: ``centre``, ``ell_comps``, and
``einstein_radius``.

When composed as a model, the ``Isothermal`` profile therefore has five free
parameters in total.
"""
mass = af.Model(Isothermal)

print(mass.info)

"""
As before, we must manually assign priors to these parameters for lens modelling.
"""
mass.centre.centre_0 = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)

"""
Provided you follow this same convention when defining your own mass profile,
it will fully support lens modelling in **PyAutoLens** without requiring any
additional code.

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

We are therefore ready to implement a mass profile, and the best place to start is the `deflections_yx_2d_from` function.

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
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        REQUIRED: The function is key for all lensing calculations and must be implemented.
        """
        pass

    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        RECOMMENDED: The convergence is used for visualization and inspecting properties of the mass profile.
        """
        pass

    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
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
    def shear_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        OPTIONAL: Shears are used for weak lensing calculations and inspection properties of the mass profile.

        Shears can reliably be calculated via methods inherited from the `OperateDeflections` class. Providing an
        analytic calculation here can speed this up and provide more accurate results.
        """
        pass


"""
__Spherical Template__

radial_grid
removal of ell_comps

__Physical Profiles__

Show how to wrap existing profiles with physical units?

__Light Profiles__

Pretty much the same but need to add text.
"""
