"""
Delaunay Pixelization (Group)
=============================

This script demonstrates using a Delaunay triangulation mesh for source reconstruction in a group-scale
strong lens, as an alternative to the rectangular mesh used in other examples.

The Delaunay mesh has several unique properties:

 - **Adaptive Mesh**: In the source plane, the Delaunay mesh uses irregularly shaped triangles rather than
   uniform rectangular pixels. This allows the mesh to better adapt to irregular and asymmetric source
   morphologies.

 - **Image Mesh**: The vertices of the Delaunay triangles are computed by overlaying a coarse uniform grid
   in the image plane and ray-tracing these coordinates to the source plane. This helps the mesh adapt
   to the magnification pattern of the lens.

 - **Interpolation**: The Delaunay mesh uses barycentric interpolation within each triangle, which is
   different from the bilinear interpolation used by rectangular meshes.

For group-scale lenses, the Delaunay mesh is particularly advantageous because the complex mass
distribution (from multiple galaxies) creates an irregular magnification pattern, and the Delaunay
triangles naturally adapt to follow the source morphology in this environment.

__Contents__

**Dataset & Mask:** Standard set up of the group dataset and 7.5" mask.
**Galaxy Centres:** Load centres for main lens and extra galaxies.
**Fit:** Create a FitImaging with a Delaunay pixelized source for a group lens.
**Model:** Compose a group lens model with a Delaunay pixelized source for modeling.
**Advantages for Group Lenses:** Why Delaunay meshes are well-suited for group-scale lensing.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt
from autoarray.inversion.plot.inversion_plots import subplot_of_mapper, subplot_mappings

"""
__Dataset__

Load the strong lens group dataset `simple`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxy Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=4,
)

"""
__Image-Plane Mesh Grid__

The Delaunay mesh requires an image-plane mesh grid whose ray-traced positions become the Delaunay
vertices in the source plane. We build it via an `Overlay` image-mesh covering the masked field, then
append a ring of edge pixels at the mask boundary so the linear inversion can zero them out. The same
image-plane mesh grid is reused below for both the concrete fit and the modeling section.
"""
edge_pixels_total = 30

image_mesh = al.image_mesh.Overlay(shape=(22, 22))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=dataset.mask)

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

"""
__Fit__

We create a fit using a Delaunay mesh whose source-pixel count matches the image-plane mesh grid
computed above, with constant regularization.

For the group lens, we include the main lens galaxy and extra galaxies with their light and mass
profiles, and a source galaxy with the Delaunay pixelization.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

pixelization = al.Pixelization(
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
    ),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={source_galaxy: image_plane_mesh_grid}
)

fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)

"""
The fit subplot shows the pixelized source does a good job at capturing the appearance of the lensed
source galaxy across the wide field of the group lens.
"""
aplt.subplot_fit_imaging(fit=fit)

print(f"Log Likelihood: {fit.log_likelihood}")

"""
__Inversion Visualization__

The Delaunay source reconstruction can be visualized using bespoke inversion plots.
"""
inversion = fit.inversion

subplot_of_mapper(inversion=inversion, mapper_index=0)
subplot_mappings(inversion=inversion, pixelization_index=0)

"""
__Reconstructed Source__

The reconstructed source pixel fluxes and their (y,x) positions in the source plane.
"""
mapper = inversion.linear_obj_list[0]

reconstruction = inversion.reconstruction
source_plane_mesh_grid = mapper.source_plane_mesh_grid

print(f"Number of source pixels: {len(reconstruction)}")
print(f"Total source flux: {np.sum(reconstruction)} e- s^-1")
print(f"Source plane mesh grid: {source_plane_mesh_grid}")

"""
__Advantages for Group Lenses__

The Delaunay mesh is particularly well-suited for group-scale lensing because:

 1. **Irregular magnification**: The combined mass of multiple galaxies creates a complex magnification
    pattern with multiple critical curves. The Delaunay mesh naturally adapts to this by placing more
    triangles in regions of high magnification (where many image pixels map to a small source-plane area).

 2. **Extended arcs**: Group lenses often produce extended arcs that span a large fraction of the image.
    The Delaunay mesh can efficiently cover these extended structures with triangles of varying size.

 3. **Multiple images**: The mass distribution of a group lens can produce many distinct images of the
    source. The Delaunay mesh places vertices based on the image-plane grid, so it naturally creates
    source pixels wherever images of the source appear.

 4. **Source complexity**: The lensed sources in group systems are often complex (e.g. merging galaxies,
    star-forming clumps) and benefit from the adaptive triangle sizes of the Delaunay mesh.
"""

"""
__Model__

We compose a group lens model with a Delaunay source for full modeling.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

    lens_dict[f"lens_{i}"] = lens

# Extra Galaxies:

extra_galaxies_list = []

for centre in extra_galaxies_centres:

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_fixed=centre
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source: Delaunay pixelization with ConstantSplit regularization.
#
# The Delaunay mesh is constructed as a concrete instance (not `af.Model`) because its `pixels` count
# is fixed by the image-plane mesh grid built above. JAX requires this to be static across samples.
#
# `ConstantSplit` is used for this first-pass model because adapt data (per-galaxy images from a
# previous search) is not yet available. A SLaM pipeline can later upgrade to `AdaptSplit` once the
# source has been imaged — see `scripts/group/slam.py` for the canonical chained pattern.

pix = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
    ),
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pix)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    extra_galaxies=extra_galaxies,
)

print(model.info)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features" / "pixelization",
    name="delaunay",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
    iterations_per_quick_update=10000,
)

"""
__Analysis__

The image-plane mesh grid is paired with the source galaxy via `AdaptImages`, keyed by the model
path so it resolves at instance time during the non-linear search.
"""
adapt_images = al.AdaptImages(
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    settings=al.Settings(use_mixed_precision=True),
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

This script demonstrated Delaunay pixelization for group-scale lenses. The Delaunay mesh provides an
irregular, adaptive triangulation that follows the source morphology, making it well-suited for the
complex magnification patterns produced by group-scale mass distributions.

For automated modeling, the SLaM pipeline uses a Hilbert mesh by default, but the Delaunay mesh can be
substituted by changing the mesh class in the source pixelization pipeline stages.
"""
