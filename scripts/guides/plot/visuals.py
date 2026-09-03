"""
Plots: Visuals (Overlays)
=========================

This example illustrates how to add overlays to plots.

Overlays are specified via two keyword arguments on `aplt.plot_array()` and `aplt.plot_grid()`:

 - `lines=`: A list of `Grid2DIrregular` objects drawn as lines (e.g. critical curves, caustics).
 - `positions=`: A `Grid2DIrregular` object drawn as scatter points (e.g. image positions).

A third overlay, `regions=`, draws filled polygons (e.g. the multiple images a source region maps to). It is
available on the `autoarray` `plot_array` rather than the `autolens` one, so the `__Regions__` section below
imports it as `aaplt.plot_array`.

__Start Here Notebook__

Refer to `guides/plot/start_here.ipynb` for a general introduction to the plotting API.

__Contents__

- **Setup:** General setup for the analysis.
- **Critical Curves:** Critical curves are plotted as lines over the image using the `lines=` keyword argument.
- **Multiple Critical Curves:** If a `Tracer` has multiple lens galaxies it may have multiple tangential and radial critical curves.
- **Caustics:** Caustics are the critical curves mapped to the source plane.
- **Image Positions:** The multiple image positions of a lensed source can be plotted using `positions=`.
- **Light Profile Centres:** The centres of light profiles can be extracted and plotted as positions over an image.
- **Mass Profile Centres:** Mass profile centres can be extracted and overlaid in the same way.
- **Combined Overlays:** `lines=` and `positions=` can be used together on the same plot.
- **Regions:** `regions=` draws filled polygons, e.g. the image-plane regions a source region maps to.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import numpy as np

import autoarray as aa
import autoarray.plot as aaplt
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
tangential_critical_curve_list = lens_calc.tangential_critical_curve_list_from(
    grid=grid
)

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
tangential_critical_curve_list = lens_calc_x2.tangential_critical_curve_list_from(
    grid=grid
)
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
tangential_critical_curve_list = lens_calc.tangential_critical_curve_list_from(
    grid=grid
)

aplt.plot_array(
    array=image,
    title="Image with Critical Curves and Multiple Images",
    lines=tangential_critical_curve_list,
    positions=multiple_images,
)

"""
__Regions__

Regions are filled polygons, drawn one colour per region, and they are what a *mapping* is plotted with: the
multiple images a region of the source maps to, in the image plane, and the region itself in the source plane.

`regions=` takes a list of regions, each region being a list of `(N, 2)` arrays of `(y, x)` coordinates -- which
is exactly the `image_contours` (or `source_contours`) of a `Mapping`. `region_colors=` overrides the default
colour cycle `["r", "g", "b", "m", "c", "y"]`, `region_alpha=` sets the fill transparency (the outline is always
opaque) and `region_labels=` writes a label at the centre of every polygon of a region, so the same source
structure is identifiable across figures.

This overlay lives on `autoarray`'s `plot_array` (imported above as `aaplt.plot_array`), not the `autolens` one.

Below we build a mapping by tracing a source-plane `Circle` through the tracer -- note the positional `(y, x)`
ordering of a `Shape` -- and overlay its image-plane regions. `guides/mappings.py` covers mappings in full.
"""
mapping = al.mappings.source_mapping_from(
    tracer=tracer,
    grid=grid,
    shape=aa.Circle(
        source_galaxy.bulge.centre[0], source_galaxy.bulge.centre[1], radius=0.1
    ),
)

aaplt.plot_array(
    array=image,
    regions=[mapping.image_contours],
    region_labels=["1"],
    title="Image with the Regions One Source Region Maps To",
)

"""
Any closed polygon can be drawn this way, so `regions=` is also the overlay for marking an arbitrary area of an
image -- a masked region, an aperture, a footprint. Each entry of the outer list gets its own colour.
"""
box = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])

aaplt.plot_array(
    array=image,
    regions=[[box], [box + 1.5]],
    region_colors=["c", "m"],
    region_alpha=0.15,
    region_labels=["A", "B"],
    title="Image with Two Hand-Made Regions",
)

"""
__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
