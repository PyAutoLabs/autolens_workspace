"""
Mappings
========

A strong lens shows you the same source more than once. The science question that follows is always
the same: **which part of the source is this piece of the image?**

A `Mapping` answers it. It pairs one region of the source plane with the image-plane regions -- the
multiple images -- that the mass model sends it to. Both sides are described as polygons in
arc-seconds, so a mapping can be drawn over any figure, and the same colour identifies the same
source structure in every panel.

This guide shows the three ways to build one, in increasing order of generality:

- a **point**, via `PointSolver`, which gives the positions of the multiple images of a single
  source coordinate and nothing else;
- a **region of a parametric source**, via `ShapeSolver`, which ray-traces a source-plane shape (a
  `Circle` at a light profile's centre, say) and returns the image-plane pixels its images cover;
- a **region of a pixelized source**, via the inversion, which splits the source reconstruction
  into clumps and reads each clump's multiple images off the mapper's mapping matrix.

It then shows what the mappings are *for*: the `subplot_mappings` figure, the brightest image-plane
coordinate of every multiple image (which is what you point a spectroscopic fibre at), and the
magnification of each image.

__Contents__

- **Dataset:** A lens whose source is two clumps, each with two star-forming knots, simulated by this guide.
- **Mask:** The 3.0" circular mask applied before fitting.
- **Point Mappings:** `PointSolver` traces one source coordinate back to its multiple images.
- **Region Mappings, Parametric Source:** `ShapeSolver` traces a source-plane `Circle` to image-plane regions.
- **Solver Refinement Steps:** How the triangle solver homes in on the images, one frame per step.
- **Magnification, Parametric Source:** The measured magnification against the analytic value of the mass model.
- **Region Mappings, Pixelized Source:** `source_clumps_from` splits the reconstruction into clumps; the threshold decides what a clump is.
- **Subplot Mappings:** The 2x2 figure showing both planes at once, for either kind of source.
- **Brightest Image-Plane Positions for Spectroscopic Follow-Up (4MOST):** Where to point a fibre, in arc-seconds, pixels and RA / Dec.
- **Magnification per Image:** The magnification of each multiple image, for both kinds of source.
- **Wrap Up:** What to read next.

__Coordinate Convention__

Every grid in **PyAutoLens** is ordered `(y, x)`, and the `Shape` classes (`aa.Circle`, `aa.Square`,
`aa.Triangle`, `aa.Polygon`) index a coordinate pair the same way -- element 0 first. Their legacy
attribute names do not match the physical axes (the attribute called `x` holds element 0, which is
`y`), so always build a shape **positionally** from a `(y, x)` centre:

    aa.Circle(y_centre, x_centre, radius=...)

That is the only place in this guide where the naming bites, and it is why every shape below is
built positionally rather than with keywords.

__Two Engines__

`Mapping` and `ImageRegion` objects are the same whichever route produced them, so everything
downstream of them -- the polygon overlays, the brightest positions, the magnifications -- is
written once and works for both:

- **Engine B (parametric)**, `al.mappings.source_mapping_from`, tiles the image plane with triangles
  and keeps the ones whose traced image meets the source shape. It knows the true geometry of the
  lens model, so it can measure a real magnification.
- **Engine A (pixelized)**, `inversion.mappings_from`, has no analytic source, only a mesh and a
  reconstruction. It finds clumps in the reconstruction and reads their images from the mapping
  matrix, which is exactly what the data constrains.

`al.mappings.mappings_from_fit(fit)` dispatches between them, so you rarely have to name one.
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
__Dataset__

The lens is an `Isothermal` mass distribution with a modest ellipticity. The source is deliberately
lumpy: two clumps 0.25" apart, each of which contains two compact star-forming knots. Real
high-redshift sources look like this, and the lumpiness is what makes the clump threshold below
worth demonstrating -- a smooth Sersic has nothing to split.

The source sits inside the tangential caustic, so every clump is quadruply imaged.

The dataset is simulated by this script the first time it is run and written to
`dataset/imaging/mappings`, exactly like the simulator scripts elsewhere in the workspace. The
`should_simulate` guard also re-simulates it if the resolution regime has changed, so the data on
disk always matches the run in front of you.
"""
dataset_name = "mappings"
dataset_path = Path("dataset") / "imaging" / dataset_name

lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.75, angle=45.0),
    ),
)

"""
The source galaxy's two clumps, each an extended envelope with two compact knots inside it. The
profiles are built in a loop so the geometry is stated once: clump `j` is centred at
`x = +/- 0.125` and its knots sit `+/- 0.08` above and below it in `y`.
"""
source_profile_dict = {}

for clump_index, x_sign in enumerate((-1.0, 1.0)):
    clump_x = x_sign * 0.125

    source_profile_dict[f"clump_{clump_index}"] = al.lp.SersicCoreSph(
        centre=(0.0, clump_x),
        intensity=1.5,
        effective_radius=0.15,
        sersic_index=1.0,
    )

    for knot_index, y_sign in enumerate((-1.0, 1.0)):
        source_profile_dict[f"clump_{clump_index}_knot_{knot_index}"] = (
            al.lp.SersicCoreSph(
                centre=(y_sign * 0.08, clump_x),
                intensity=5.0,
                effective_radius=0.05,
                sersic_index=1.0,
            )
        )

source = al.Galaxy(redshift=1.0, **source_profile_dict)

tracer = al.Tracer(galaxies=[lens, source])

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system it is simulated here, so the guide runs
end-to-end without any other script being run first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    simulator_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)
    simulator_grid = simulator_grid.apply_over_sampling(over_sample_size=4)

    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11),
        sigma=0.1,
        pixel_scales=simulator_grid.pixel_scales,
        convolve_over_sample_size=1,
    )

    simulator = al.SimulatorImaging(
        exposure_time=1200.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
        noise_seed=1,
    )

    simulated_dataset = simulator.via_tracer_from(tracer=tracer, grid=simulator_grid)

    aplt.fits_imaging(
        dataset=simulated_dataset,
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        overwrite=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
__Mask__

A 3.0" circular mask, applied before anything is fitted. The image-plane regions of a mapping are
reported on the *masked* data, so the mask decides which pixels a region can contain.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_pixelization=4)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Point Mappings__

The simplest mapping is a single source coordinate. `PointSolver` tiles the image plane with
triangles, keeps those whose traced centres surround the source coordinate, and subdivides until it
has the image positions to the requested precision.

This is the right tool when the source really is a point -- a lensed quasar or supernova -- and it
is the cheapest way to see the image configuration of a mass model. What it cannot tell you is
which image *pixels* belong to which image, because a point has no size: for that you need a region,
which is the rest of this guide.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

point_solver = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)

source_centre = source.clump_0.centre

multiple_images = point_solver.solve(
    tracer=tracer,
    source_plane_coordinate=source_centre,
)

print(f"Source-plane coordinate: {source_centre}")
print("Multiple image positions (arcsec):")
print(np.asarray(multiple_images.array))

aplt.plot_array(
    array=dataset.data,
    positions=multiple_images,
    title="Point Mappings: Multiple Images of One Source Coordinate",
)

"""
__Region Mappings, Parametric Source__

Give the source a size and the answer becomes a region rather than a point. `ShapeSolver` keeps
every image-plane triangle whose traced image *overlaps* the source shape, so the kept triangles
tile the multiple images of the whole shape.

`al.mappings.source_mapping_from` is the one-line entry point: pass the tracer, the image-plane grid
the regions are reported on, and the source-plane shape. Here the shape is a `Circle` at the first
clump's centre with its half-light radius -- remember the `(y, x)` positional convention from the
header.

The returned `Mapping` carries `image_contours` (the boundary of each multiple image, in
arc-seconds) and `source_contours` (the shape's own boundary). Passing them to `regions=` draws
them as filled polygons.

Note the count printed below: the number of image *regions* is not always the number of point
images. Two images which merge across a critical curve are one connected region, because they touch.
"""
shape = aa.Circle(
    source_centre[0],
    source_centre[1],
    radius=source.clump_0.effective_radius,
)

mapping = al.mappings.source_mapping_from(tracer=tracer, grid=grid, shape=shape)

print(f"Image regions: {len(mapping.image_regions)}")
print(
    f"Pixels per region: {[len(region.slim_indexes) for region in mapping.image_regions]}"
)

aaplt.plot_array(
    array=dataset.data,
    regions=[mapping.image_contours],
    region_labels=["1"],
    title="Region Mappings: The Images of One Source Clump",
)

"""
The same mapping drawn in the source plane, where its one polygon is the circle itself. Drawn side
by side with the figure above, this is the complete statement of the mapping: this circle, those
arcs.

The source plane is tiny compared with the image plane, so it is evaluated on its own fine grid --
1" across at 0.01" per pixel -- which resolves the two clumps and their knots.
"""
source_plane_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.01)

source_plane_image = source.image_2d_from(grid=source_plane_grid)

aaplt.plot_array(
    array=source_plane_image,
    regions=[mapping.source_contours],
    region_labels=["1"],
    title="Region Mappings: The Source-Plane Region",
)

"""
Each `ImageRegion` is more than an outline. It knows its own pixels, so it can be interrogated for
the quantities a science case needs -- its area, its total flux, its brightest pixel and its
flux-weighted centroid -- all in arc-seconds and in the units of the array you hand it.
"""
image = tracer.image_2d_from(grid=grid)

for region_index, region in enumerate(mapping.image_regions):
    print(
        f"Region {region_index}: "
        f"area = {region.area():.3f} arcsec^2, "
        f"flux = {region.flux_from(array=image):.3f}, "
        f"brightest = {np.round(region.brightest_coordinate_from(array=image), 3)}"
    )

"""
__Solver Refinement Steps__

The solver is hierarchical: it starts with triangles the size of an image pixel, keeps those whose
traced images meet the source, then subdivides only the kept set and its immediate neighbours. Each
pass halves the triangle side length, so the kept set converges onto the images geometrically rather
than by brute force over a fine grid.

`steps()` yields that process one frame at a time. Below we scatter the centre of every kept
triangle over the data, one figure per step: the first frame is a coarse outline of the arcs, the
last is the arcs themselves resolved to a tenth of a pixel.

The kept *area* is not monotone across the frames -- each step filters the neighbourhood of the
previous kept set, which is larger than that set -- but the search envelope always shrinks.
"""
shape_solver = al.ShapeSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.01,
    magnification_threshold=0.1,
)

for step in shape_solver.steps(tracer=tracer, shape=shape):
    triangle_centres = np.asarray(step.filtered_triangles.means)

    print(f"Step {step.number}: {len(step.filtered_triangles)} triangles kept")

    aaplt.plot_array(
        array=dataset.data,
        grid=triangle_centres,
        title=f"Solver Step {step.number}: {len(step.filtered_triangles)} Triangles",
    )

"""
__Magnification, Parametric Source__

Because the solver knows the image-plane area the source's images cover, it can measure the
magnification directly: the total kept area divided by the area of the source shape.

The value to check it against is the analytic magnification of the mass model at the image
positions. For any mass distribution that is

    mu = 1 / |(1 - kappa)^2 - gamma^2|

with the convergence `kappa` and the shear magnitude `gamma` evaluated at each image position -- both
of which an `Isothermal` computes in closed form. Summing `mu` over the point images gives the total
magnification of a *point* source at the clump centre.

The two numbers answer slightly different questions, and the difference is physics rather than
error. The analytic sum is the magnification of a point; `find_magnification` measures a source of
finite size, which is the average of the point magnification over the circle. A source straddling a
caustic therefore measures *lower* than the point value at its centre -- the caustic is where the
point value diverges. Shrinking the circle recovers the point value, which is what the loop below
prints.
"""
convergence = np.asarray(lens.mass.convergence_2d_from(grid=multiple_images))
shear = np.asarray(lens.mass.shear_yx_2d_from(grid=multiple_images).magnitudes)

analytic_magnifications = 1.0 / np.abs((1.0 - convergence) ** 2.0 - shear**2.0)

print("Analytic point magnifications:")
print(np.round(np.sort(analytic_magnifications)[::-1], 2))
print(f"Analytic total: {np.sum(analytic_magnifications):.2f}")

measured_magnifications = shape_solver.find_magnification(
    tracer=tracer, shape=shape, per_image=True
)

print("Measured magnifications, one per image region:")
print(np.round(np.sort(measured_magnifications)[::-1], 2))

"""
The finite-source magnification against the point value, as the source shrinks: on this dataset it
climbs from 22.1 at a 0.15" radius to 29.6 at 0.05", against an analytic point value of 30.0. Do not
shrink the source indefinitely -- a source comparable to `pixel_scale_precision` is measured on
triangles as big as it is, so refine the solver (a smaller `pixel_scale_precision`) before trusting a
radius that small.
"""
for radius in (0.15, 0.10, 0.05):
    total = shape_solver.find_magnification(
        tracer=tracer,
        shape=aa.Circle(source_centre[0], source_centre[1], radius=radius),
    )

    print(f'radius = {radius:.2f}": total magnification = {total:.2f}')

"""
__Region Mappings, Pixelized Source__

A pixelized source has no analytic shape to trace. What it has is a reconstruction: a value per mesh
pixel, fitted to the data. The source-plane regions are therefore found *in the reconstruction*, by
taking the mesh pixels which are bright relative to its maximum and splitting them into connected
groups -- clumps.

We first fit the dataset with a rectangular mesh and constant regularization, exactly as
`imaging/features/pixelization/fit.py` does. The lens mass model is the true one, since this guide is
about the mappings rather than about inferring a model.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=(60, 60)),
    regularization=al.reg.Constant(coefficient=2.0),
)

source_pixelized = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer_pixelized = al.Tracer(galaxies=[lens, source_pixelized])

fit_pixelized = al.FitImaging(dataset=dataset, tracer=tracer_pixelized)

aplt.subplot_fit_imaging(fit=fit_pixelized)

inversion = fit_pixelized.inversion

"""
`threshold` is the knob which decides what counts as a distinct source structure. A mesh pixel joins
a clump if its reconstructed value exceeds `threshold` times the reconstruction's maximum, so raising
the threshold cuts the source at a higher contour and splits it into smaller, brighter pieces.

The three figures below are the same reconstruction cut at three heights, and they are three
different scientific statements about the same source:

- `threshold=0.2` cuts low enough that the two clumps are joined by the emission between them: one
  clump, i.e. one galaxy. Its multiple images merge into a single region running round the whole
  Einstein ring, and because the overlay fills each polygon's outer boundary the ring's interior is
  shaded too -- the region is the ring, not the disc.
- `threshold=0.5` cuts above that bridge: two clumps, the two galaxies.
- `threshold=0.8` cuts above the envelopes entirely and keeps only the compact knots: four clumps,
  the individual star-forming regions.

The printed counts are what this dataset actually gives; on your own data the numbers will differ,
which is the point of the knob.
"""
for threshold in (0.2, 0.5, 0.8):
    clumps = inversion.source_clumps_from(threshold=threshold, min_pixels=3)

    print(
        f"threshold = {threshold}: {len(clumps)} clumps, "
        f"mesh pixels per clump = {[len(clump) for clump in clumps]}"
    )

    mappings = inversion.mappings_from(threshold=threshold, min_pixels=3)

    region_labels = [str(index + 1) for index in range(len(mappings))]

    aaplt.plot_array(
        array=fit_pixelized.data,
        regions=[mapping.image_contours for mapping in mappings],
        region_labels=region_labels,
        title=f"Image-Plane Regions (threshold = {threshold})",
    )

"""
The source-plane half of the same statement, drawn on the reconstruction itself. Each clump is the
union of its mesh pixels, and it is drawn in the colour its multiple images carry in the figures
above.
"""
mapper = inversion.cls_list_from(cls=aa.Mapper)[0]

mappings = inversion.mappings_from(threshold=0.5, min_pixels=3)

aaplt.plot_inversion_reconstruction(
    pixel_values=inversion.reconstruction_dict[mapper],
    mapper=mapper,
    regions=[mapping.source_contours for mapping in mappings],
    title="Source Reconstruction, Clumps at threshold = 0.5",
)

"""
Clump finding can be bypassed entirely by naming the mesh pixels yourself, which is what
`Mapper.mappings_from` is for. This is the route to take when you want the images of one specific
source pixel -- the mapping a `Mapper` performs, seen one pixel at a time. Below we take the single
brightest mesh pixel; any index (or nested list of index groups) works the same way.
"""
brightest_pix_index = int(np.argmax(inversion.reconstruction_dict[mapper]))

mappings_by_hand = mapper.mappings_from(pix_indexes=[[brightest_pix_index]])

print(
    f"Source pixel {brightest_pix_index} maps to "
    f"{len(mappings_by_hand[0].image_regions)} image regions"
)

aaplt.plot_array(
    array=fit_pixelized.data,
    regions=[mapping.image_contours for mapping in mappings_by_hand],
    title="The Image Pixels Which Map To One Source Pixel",
)

"""
__Subplot Mappings__

`subplot_fit_imaging_mappings` puts both planes in one 2x2 figure: the data and the model image of
the source plane on the top row, with the image-plane regions overlaid and the critical curves; the
source plane zoomed and unzoomed on the bottom row, with the source regions and the caustics. Every
region is numbered, and its multiple images carry the same number and colour -- so a merging pair of
source galaxies can be read off the figure by colour alone.

It works for either kind of source. For the pixelized fit it draws the clumps of the reconstruction.
"""
aplt.subplot_fit_imaging_mappings(fit=fit_pixelized)

"""
For a parametric fit the same function traces a `Circle` at the source's first light profile instead,
with that profile's half-light radius. Pass `shape=` to choose the region yourself.
"""
fit_parametric = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging_mappings(fit=fit_parametric)

"""
__Brightest Image-Plane Positions for Spectroscopic Follow-Up (4MOST)__

A spectroscopic follow-up campaign -- 4MOST, or any fibre-fed instrument -- needs one coordinate per
multiple image: where to put the fibre. The brightest pixel of each image region is that coordinate,
and `multiple_image_positions_from` returns it for every image of every clump.

The brightness is read from the *model* image of the source plane rather than from the data, so the
positions describe the lens model instead of the noise realisation.
"""
positions = al.mappings.multiple_image_positions_from(fit=fit_pixelized)

print("Brightest coordinate of each multiple image (arcsec):")
print(np.asarray(positions.array))

aplt.plot_array(
    array=fit_pixelized.data,
    positions=positions,
    title="Brightest Pixel Of Every Multiple Image",
)

"""
The same positions as pixel coordinates of the data, which is what a WCS conversion needs. These are
in the FITS convention: 1-based, referred to pixel centres, and continuous, so the fractional part is
the sub-pixel offset.
"""
pixel_coordinates = al.mappings.multiple_image_pixel_coordinates_from(fit=fit_pixelized)

for y, x in pixel_coordinates:
    print(f"pixel (y, x) = ({y:.2f}, {x:.2f})")

"""
The library stops at pixel coordinates, because turning them into RA / Dec is a question about the
header of the FITS the data came from, not about the lens model. The conversion is one call to
astropy, and it belongs in your own script, right here.

`Array2D.from_fits` keeps the FITS header, so for real data with WCS keywords the whole of the block
below collapses to:

    from astropy.wcs import WCS

    wcs = WCS(dataset.data.header)

The dataset simulated by this guide carries no WCS keywords -- it was never on the sky -- so we build
one explicitly from the pixel scale and a stated reference coordinate. `crpix` is 1-based and points
at the centre of the image, `crval` is the RA / Dec that centre corresponds to, and `cdelt` is the
pixel scale in degrees, negative in RA because RA increases to the left.
"""
from astropy.wcs import WCS

reference_ra = 150.1
reference_dec = 2.2

wcs = WCS(naxis=2)
wcs.wcs.crpix = [
    dataset.data.shape_native[1] / 2.0 + 0.5,
    dataset.data.shape_native[0] / 2.0 + 0.5,
]
wcs.wcs.crval = [reference_ra, reference_dec]
wcs.wcs.cdelt = [
    -dataset.pixel_scales[1] / 3600.0,
    dataset.pixel_scales[0] / 3600.0,
]
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

print("Fibre positions for spectroscopic follow-up:")

for y, x in pixel_coordinates:
    ra, dec = wcs.all_pix2world(x, y, 1)

    print(
        f"pixel (y, x) = ({y:.2f}, {x:.2f}) -> RA = {float(ra):.6f}, Dec = {float(dec):.6f}"
    )

"""
__Fibre Diameter__

A fibre is not a pixel. 4MOST's fibres are 1.45" across, and comparable instruments are 1-2", which
is wider than the images of a compact source and comparable to a lensed arc. Two consequences
follow.

First, for an extended arc the brightest pixel and the flux-weighted centroid are different
coordinates, and the centroid is usually what you want a fibre centred on because it maximises the
flux the fibre collects. Pass `use_centroid=True` for it -- the centroid is also sub-pixel and varies
smoothly under small changes to the model, whereas the brightest pixel jumps from one pixel to the
next.

Second, images closer together than the fibre diameter cannot be separated by the instrument: check
the separations below before proposing one fibre per image.
"""
centroids = al.mappings.multiple_image_positions_from(
    fit=fit_pixelized, use_centroid=True
)

for brightest, centroid in zip(
    np.asarray(positions.array), np.asarray(centroids.array)
):
    offset = np.hypot(*(np.asarray(brightest) - np.asarray(centroid)))

    print(
        f"brightest = {np.round(brightest, 3)}, "
        f"centroid = {np.round(centroid, 3)}, "
        f'offset = {offset:.3f}"'
    )

"""
__Magnification per Image__

The magnification of each multiple image, ordered exactly as the positions above are.

For a **parametric** source this is a true magnification: the image-plane area the source's images
cover, divided by the source's own area, measured on the solver's finest triangles.
"""
magnifications_parametric = al.mappings.magnifications_from(fit=fit_parametric)

print("Parametric source, magnification per image:")
print(np.round(magnifications_parametric, 3))

"""
For a **pixelized** source there is no source-plane area to divide by, only a mesh, so the number
returned is a *flux share*: each image's summed model flux divided by the total model flux of its
clump across every image. The shares of one clump sum to 1.

That is the relative magnification of a clump's images, which is the quantity the data actually
constrains -- and it is what a flux-ratio measurement compares against. It is not on the same scale
as the parametric magnification above, so do not compare the two numbers directly. To get an absolute
magnification for a pixelized source, measure it from the mass model with the parametric engine (or
compare the reconstruction's total flux with the model image's).
"""
magnifications_pixelized = al.mappings.magnifications_from(fit=fit_pixelized)

print("Pixelized source, flux share per image:")
print(np.round(magnifications_pixelized, 3))

print(f"Sum over all images: {np.sum(magnifications_pixelized):.3f} (one per clump)")

"""
__Wrap Up__

The three routes to a mapping, and what each is for:

- `PointSolver` for the image positions of a point source -- lensed quasars, supernovae, and the
  positional likelihoods that group- and cluster-scale modeling uses (`guides/point_source_pairing.py`).
- `al.mappings.source_mapping_from` for the images of a region of a parametric source, and with it
  the true magnification of the lens model.
- `inversion.mappings_from` for the clumps of a pixelized reconstruction, where the threshold decides
  what counts as a distinct structure.

All three return the same `Mapping` objects, which is why `subplot_mappings`, the brightest
positions and the magnifications are written once and work for all of them.

Where to go next:

- `imaging/features/pixelization/fit.py` for pixelized source reconstruction in full.
- `guides/plot/visuals.py` for the `regions=` overlay used throughout this guide, alongside the other
  overlays.
- `imaging/source_science.py` for the science these mappings feed: magnification, intrinsic size and
  the unlensed properties of the source.
- Chapter 3 of **HowToLens** for the mapper and inversion machinery from first principles.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

This guide simulates its own 100x100 dataset and reads clump counts off the
reconstruction; SMALL_DATASETS would cap the grid and change what the
thresholds find, so the prose would no longer describe the figures.

ENV: full_datasets
"""
