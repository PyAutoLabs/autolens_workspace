"""
Fit Features: Pixelization (Group)
==================================

This script demonstrates how to create a `FitImaging` object for a group-scale strong lens where the source
galaxy is reconstructed using a pixelized mesh, rather than parametric light profiles.

For group-scale lenses, the fit automatically handles contributions from all galaxies: the main lens galaxy,
extra galaxies, and the pixelized source. The combined deflection field from all mass profiles is used to
ray-trace image pixels to the source plane, where the Delaunay mesh reconstructs the source emission.

This example uses concrete (non-model) galaxy objects to illustrate the API, rather than performing a
full model-fit.

__Contents__

**Dataset & Mask:** Standard set up of the group dataset and 7.5" mask.
**Galaxy Centres:** Load centres for main lens and extra galaxies from JSON files.
**Fitting:** Create a FitImaging with a pixelized source for a group lens.
**Inversion:** Inspect the pixelized source reconstruction via the inversion object.
**Over Sampling:** Adaptive over-sampling at all galaxy centres.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
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

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
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

We use a 7.5 arcsecond circular mask for the group-scale lens.
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

Load centres for the main lens galaxies and extra galaxies.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__

We apply adaptive over-sampling at all galaxy centres for the lens light profiles, and a uniform
over-sampling for the pixelization grid.
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
__Fitting__

We create concrete galaxy objects for the group lens system. The main lens and extra galaxies have
light and mass profiles, while the source galaxy uses a pixelized reconstruction.

The `Pixelization` is created with a `Delaunay` mesh (500 pixels) and `Constant` regularization.
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
    mesh=al.mesh.Delaunay(pixels=500),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

"""
We create a tracer including all group galaxies and the pixelized source.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

"""
The `FitImaging` object handles the full group-scale fit automatically: it sums the lens light from all
galaxies, computes the combined deflection field from all mass profiles, ray-traces to the source plane,
performs the pixelized source reconstruction, convolves with the PSF, and computes the log likelihood.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log Likelihood: {fit.log_likelihood}")

"""
__Inversion__

The pixelized source reconstruction is stored in the `inversion` attribute of the fit. This contains the
reconstructed source pixel fluxes, the mapping between image and source pixels, and diagnostic quantities.
"""
inversion = fit.inversion

print(f"Inversion Object: {inversion}")

"""
The reconstructed source pixel fluxes are available as a 1D array.
"""
reconstruction = inversion.reconstruction

print(f"Reconstructed Source Pixel Fluxes: {reconstruction}")
print(f"Total Source Flux: {np.sum(reconstruction)} e- s^-1")

"""
Bespoke pixelization visualizations show the source reconstruction and image-source mapping.
"""
subplot_of_mapper(inversion=inversion, mapper_index=0)
subplot_mappings(inversion=inversion, pixelization_index=0)

"""
__Linear Algebra Matrices__

The inversion exposes the linear algebra matrices used for the source reconstruction.
"""
print(inversion.curvature_matrix)
print(inversion.regularization_matrix)
print(inversion.curvature_reg_matrix)

"""
__Plane Quantities__

For a group-scale fit with a pixelized source, the model images of each plane are accessible. The first
plane contains all lens galaxies (main + extra), the second plane contains the pixelized source.
"""
print(fit.model_images_of_planes_list[0].slim)
print(fit.model_images_of_planes_list[1].slim)

"""
Subtracted images isolate the emission of each plane. For example, the second subtracted image has all
lens galaxy light removed, leaving only the lensed source emission.
"""
print(fit.subtracted_images_of_planes_list[0].slim)
print(fit.subtracted_images_of_planes_list[1].slim)

"""
__Figures of Merit__

The fit provides single-valued figures of merit. For a pixelized source, the log likelihood includes
the Bayesian evidence from the regularization, which penalizes overly complex source reconstructions.
"""
print(f"Chi Squared: {fit.chi_squared}")
print(f"Noise Normalization: {fit.noise_normalization}")
print(f"Log Likelihood: {fit.log_likelihood}")

"""
__Wrap Up__

This script demonstrated how to create a FitImaging with a pixelized source for a group-scale lens.

The key points are:

 - The FitImaging automatically handles all group galaxies when computing the fit.
 - The combined deflection field from all mass profiles is used for ray-tracing to the source plane.
 - The inversion object provides access to the source reconstruction, mapping matrices, and evidence terms.
 - Plane quantities allow isolating the contributions of lens galaxies vs. the pixelized source.
"""
