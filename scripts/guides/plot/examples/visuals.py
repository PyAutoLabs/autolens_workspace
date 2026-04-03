"""
Plots: Visuals (Overlays)
=========================

This example illustrates how to add overlays to plots using the new API.

Overlays are specified via two keyword arguments on `aplt.plot_array()` and `aplt.plot_grid()`:

 - `lines=`: A list of `Grid2DIrregular` objects drawn as lines (e.g. critical curves, caustics).
 - `positions=`: A `Grid2DIrregular` object drawn as scatter points (e.g. image positions).

The old `Visuals2D` and `MatPlot2D` objects that configured overlays have been removed.

__Start Here Notebook__

Refer to `plots/start_here.ipynb` for a general introduction to the new plotting API.

__Contents__

- **Setup**: Set up all objects used to illustrate overlays.
- **Critical Curves**: Plot tangential and radial critical curves using `lines=`.
- **Caustics**: Plot tangential and radial caustics using `lines=`.
- **Multiple Critical Curves**: Plot critical curves for a multi-galaxy system.
- **Image Positions**: Plot image positions using `positions=`.
- **Light Profile Centres**: Overlay centre positions on an image.
- **Mass Profile Centres**: Overlay mass profile centres on an image.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Setup__

Create the standard objects used to illustrate overlays.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.2, 0.2)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCoreSph(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

lens_calc = al.LensCalc.from_tracer(tracer=tracer)

lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-1.0, 0.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-1.0, 0.0), einstein_radius=0.8, ell_comps=(0.2, 0.2)
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCoreSph(
        centre=(0.2, 0.2), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

tracer_x2 = al.Tracer(
    galaxies=[lens_galaxy, lens_galaxy_1, source_galaxy, source_galaxy_1]
)

lens_calc_x2 = al.LensCalc.from_tracer(tracer=tracer_x2)

dataset_path = Path("dataset") / "imaging" / "slacs1430+4105"
data_path = dataset_path / "data.fits"
data = al.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.03)

"""
__Critical Curves__

Critical curves are plotted as lines over the image using the `lines=` keyword argument.

`tangential_critical_curve_list_from` returns a list of `Grid2DIrregular` objects, one per
tangential critical curve. Pass this list directly to `lines=`.
"""
tangential_critical_curve_list = lens_calc.tangential_critical_curve_list_from(grid=grid)

image = tracer.image_2d_from(grid=grid)

aplt.plot_array(
    array=image,
    title="Image with Tangential Critical Curves",
    lines=tangential_critical_curve_list,
)

"""
Radial critical curves can be overlaid in the same way. Combine both lists with `+` to
overlay tangential and radial critical curves together.
"""
radial_critical_curve_list = lens_calc.radial_critical_curve_list_from(grid=grid)

aplt.plot_array(
    array=image,
    title="Image with All Critical Curves",
    lines=tangential_critical_curve_list + radial_critical_curve_list,
)

"""
__Multiple Critical Curves__

If a `Tracer` has multiple lens galaxies it may have multiple tangential and radial critical
curves. These are all contained in the returned lists and plotted together.
"""
tangential_critical_curve_list = lens_calc_x2.tangential_critical_curve_list_from(grid=grid)
radial_critical_curve_list = lens_calc_x2.radial_critical_curve_list_from(grid=grid)

image_x2 = tracer_x2.image_2d_from(grid=grid)

aplt.plot_array(
    array=image_x2,
    title="Two-Galaxy System Critical Curves",
    lines=tangential_critical_curve_list + radial_critical_curve_list,
)

"""
__Caustics__

Caustics are the critical curves mapped to the source plane. They are plotted over the
source-plane image using `lines=`.
"""
tangential_caustic_list = lens_calc.tangential_caustic_list_from(grid=grid)
radial_caustic_list = lens_calc.radial_caustic_list_from(grid=grid)

source_image = tracer.image_2d_list_from(grid=grid)[1]

aplt.plot_array(
    array=source_image,
    title="Source Plane with Tangential Caustics",
    lines=tangential_caustic_list,
)

aplt.plot_array(
    array=source_image,
    title="Source Plane with All Caustics",
    lines=tangential_caustic_list + radial_caustic_list,
)

"""
__Image Positions__

The multiple image positions of a lensed source can be plotted using `positions=`.

`positions=` accepts an `al.Grid2DIrregular` object.
"""
solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)
multiple_images = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

aplt.plot_array(
    array=image,
    title="Image with Multiple Images",
    positions=multiple_images,
)

"""
Arbitrary (y,x) coordinates can also be plotted as positions, for example to mark
interesting regions on an image.
"""
positions = al.Grid2DIrregular(values=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

aplt.plot_array(
    array=data,
    title="Data with Positions",
    positions=positions,
)

"""
__Light Profile Centres__

The centres of light profiles can be extracted and plotted as positions over an image.

We extract image-plane centres from the first (lens) galaxy.
"""
light_profile_centres = tracer.galaxies[0].extract_attribute(
    cls=al.LightProfile, attr_name="centre"
)

aplt.plot_array(
    array=image,
    title="Image with Light Profile Centres",
    positions=light_profile_centres,
)

"""
Source-plane centres can be extracted from the last galaxy.
"""
source_profile_centres = tracer.galaxies[-1].extract_attribute(
    cls=al.LightProfile, attr_name="centre"
)

aplt.plot_array(
    array=source_image,
    title="Source Plane with Light Profile Centres",
    positions=source_profile_centres,
)

"""
__Mass Profile Centres__

Mass profile centres can be extracted and overlaid in the same way.
"""
mass_profile_centres = tracer.extract_attribute(
    cls=al.mp.MassProfile, attr_name="centre"
)

aplt.plot_array(
    array=image,
    title="Image with Mass Profile Centres",
    positions=mass_profile_centres,
)

"""
__Combined Overlays__

`lines=` and `positions=` can be used together on the same plot.
"""
tangential_critical_curve_list = lens_calc.tangential_critical_curve_list_from(grid=grid)

aplt.plot_array(
    array=image,
    title="Image with Critical Curves and Multiple Images",
    lines=tangential_critical_curve_list,
    positions=multiple_images,
)

"""
Finish.
"""
