"""
Delaunay Pixelization (Multi Galaxy)
====================================

This script fits a multi-galaxy strong lens with a source reconstructed on a **Delaunay** mesh, the alternative
to the rectangular meshes used elsewhere in this package.

A Delaunay mesh differs from a rectangular one in three ways:

 - **Irregular source pixels:** the source is reconstructed on triangles of varying size and shape, rather than
   on a grid of equal-sized rectangles.

 - **Image-plane mesh grid:** the triangle vertices are (y, x) coordinates placed in the image plane and
   ray-traced to the source plane, so where the source pixels land is set by the mass model rather than fixed
   in advance. This grid must be computed before the fit.

 - **Split regularization:** the Delaunay meshes supply the split-cross mappings that `ConstantSplit` and
   `AdaptSplit` regularization need. Pairing those schemes with a rectangular mesh raises a
   `PixelizationException` instead.

__Contents__

- **Dataset:** Load the multi-galaxy dataset that is fitted.
- **Extra Galaxies Noise Scaling:** Scale the contaminating galaxy's light out of the fit.
- **Mask, Centres & Over Sampling:** Standard set up, over-sampled at every deflector centre.
- **Image-Plane Mesh Grid:** Build the grid whose ray-traced positions become the Delaunay vertices.
- **Edge Zeroing:** Add the ring of edge points the inversion zeroes.
- **Fit:** Fit the dataset with a Delaunay source, without a non-linear search.
- **Inversion Visualization:** The bespoke Delaunay inversion plots.
- **Reconstructed Source:** The reconstructed fluxes and their source-plane positions.
- **Model:** Compose a multi-galaxy model with a Delaunay source.
- **Search & Analysis:** Configure the fit, pairing the mesh grid to the source galaxy.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/features/pixelization/fit.py` for the rectangular-mesh
version of the fit below, and `imaging/features/pixelization/delaunay.py` for the full galaxy-scale walkthrough
of the Delaunay API, including its barycentric interpolation and mesh internals.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt
from autoarray.inversion.plot.inversion_plots import subplot_of_mapper, subplot_mappings

"""
__Dataset__

The `simple` multi-galaxy dataset, the same co-dominant pair fitted by `multi_galaxy/modeling.py`.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Extra Galaxies Noise Scaling__

Scale the faint contaminant out of the fit, as `multi_galaxy/modeling.py` explains.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.

`over_sample_size_pixelization` sets the over-sampling of the grid used by the inversion, which is separate from
the light-profile over-sampling above.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    ),
    over_sample_size_pixelization=4,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Image-Plane Mesh Grid__

An `Overlay` image-mesh places a regular grid of (y, x) points across the masked image plane. These are
ray-traced to the source plane for every mass model sampled, and the traced positions become the Delaunay
vertices.

The number of `pixels` given to the mesh must equal the number of coordinates in this grid, and must be fixed
before the fit — JAX uses it to determine static array shapes, so it cannot vary between samples.

Ray-tracing the vertices is what couples the mesh to the mass model: with two co-dominant deflectors the
deflection field is the sum of both, so where the source pixels land depends on both mass models rather than on
one.
"""
image_mesh = al.image_mesh.Overlay(shape=(26, 26))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=dataset.mask)

"""
__Edge Zeroing__

Pixels at the edge of the source-plane mesh are forced to zero brightness by the linear solver, which prevents
edge pixels from reconstructing bright flux to absorb lens-light subtraction residuals.

A rectangular mesh knows which of its pixels are edge pixels and does this internally. A Delaunay mesh does not,
so a ring of edge points is appended to the image-plane grid by hand and their total passed to the mesh as
`zeroed_pixels`. They are ray-traced and triangulated like any other vertex, then zeroed in the inversion.
"""
edge_pixels_total = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

"""
__Fit__

Fit the dataset without a non-linear search, so the Delaunay objects can be inspected directly.

The deflectors are the same two used in `multi_galaxy/features/pixelization/fit.py`, with the source's
rectangular mesh replaced by a Delaunay one. The image-plane mesh grid is passed to the fit via `AdaptImages`,
keyed by the source galaxy.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

pixelization = al.Pixelization(
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
    ),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source_galaxy])

adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={source_galaxy: image_plane_mesh_grid}
)

fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log Likelihood: {fit.log_likelihood}")

"""
__Inversion Visualization__

The Delaunay source reconstruction has its own inversion plots, which show the triangulation itself and the
mappings between image and source pixels.
"""
inversion = fit.inversion

subplot_of_mapper(inversion=inversion, mapper_index=0)
subplot_mappings(inversion=inversion, pixelization_index=0)

"""
__Reconstructed Source__

The reconstructed fluxes and the (y, x) positions of the triangle vertices in the source plane. Unlike a
rectangular mesh, those positions change with the mass model, because they are the ray-traced image-plane grid.
"""
mapper = inversion.cls_list_from(cls=al.Mapper)[0]

reconstruction = inversion.reconstruction
source_plane_mesh_grid = mapper.source_plane_mesh_grid

print(f"Number of source pixels: {len(reconstruction)}")
print(f"Total source flux: {np.sum(reconstruction)} e- s^-1")
print(f"Source plane mesh grid: {source_plane_mesh_grid}")

"""
__Model__

The standard multi-galaxy composition — one `lens_i` per deflector with its mass centre fixed, the shear in its
own `shear_galaxy` — with a Delaunay source.

The mesh is a concrete instance rather than an `af.Model` because its `pixels` count is fixed by the image-plane
mesh grid built above, and must stay static across samples.

`ConstantSplit` regularization is used rather than `AdaptSplit`, because the split-adapt scheme needs adapt
images from an earlier fit which a standalone script does not have. `multi_galaxy/slam.py` is where that upgrade
happens.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

# External Shear:

shear_galaxy_model = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

# Source (Delaunay):

pix = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
    ),
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pix)

model = af.Collection(
    galaxies=af.Collection(
        **lens_dict, shear_galaxy=shear_galaxy_model, source=source
    )
)

print(model.info)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features" / "pixelization",
    name="delaunay",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
)

"""
__Analysis__

The image-plane mesh grid is paired with the source galaxy via `AdaptImages` again, but keyed by the source's
model path rather than the galaxy object, so it resolves at instance time during the search.

The multiple images are loaded from the dataset and used as a `PositionsLH` penalty, as in
`multi_galaxy/features/pixelization/modeling.py`.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "positions.json")
)

adapt_images = al.AdaptImages(
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
    use_jax=True,
)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/pixelization/fit.py` — the same fit with a rectangular mesh.
 - `multi_galaxy/features/pixelization/adaptive.py` — the adaptive rectangular mesh and regularization, set up
   by search chaining.
 - `multi_galaxy/slam.py` — the SLaM pipeline, whose source stages upgrade to the split-adapt schemes.
 - `imaging/features/pixelization/delaunay.py` — the full galaxy-scale walkthrough of the Delaunay API.
"""
