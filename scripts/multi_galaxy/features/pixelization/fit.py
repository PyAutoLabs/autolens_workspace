"""
Fits: Pixelization (Multi Galaxy)
=================================

This script fits a multi-galaxy strong lens with a pixelized source reconstruction, without a non-linear search,
so the pixelization API and the objects it produces can be inspected directly.

__Contents__

- **Dataset:** Load the multi-galaxy dataset that is fitted.
- **Extra Galaxies Noise Scaling:** Scale the contaminating galaxy's light out of the fit.
- **Mask:** Standard set up of the mask that is fitted.
- **Over Sampling:** Adaptive over-sampling at every deflector centre.
- **Mesh Shape:** The resolution of the source-plane mesh.
- **Pixelization:** Compose the mesh and regularization.
- **Main Lens Galaxies:** The co-dominant deflectors.
- **Fit:** Fit the tracer to the dataset and plot the result.
- **Inversion:** Inspect the source reconstruction.
- **Wrap Up:** Where to go next.

__Start Here Notebook__

If any code in this script is unclear, refer to `imaging/features/pixelization/fit.py`, which gives the full
pixelization API walkthrough at galaxy scale, including the linear objects, mapping matrices and reconstruction
internals. This script shows the same API applied to a lens with two co-dominant deflectors.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the `simple` multi-galaxy dataset, the same one fitted by `multi_galaxy/fit.py`.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
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

The `simple` dataset includes a faint extra galaxy whose light is scaled out of the fit before masking, as
described in `multi_galaxy/modeling.py`.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Mask__

We create a 3.0 arcsecond circular mask and apply it to the `Imaging` object that is fitted.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

The adaptive over-sampling scheme is centred on **every** main lens galaxy, because each has its own steep
central light profile. `centre_list` takes as many centres as you give it.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mesh Shape__

The shape of the mesh the source is reconstructed on, in source-plane pixels.
"""
mesh_shape = (30, 30)

"""
__Pixelization__

A `Pixelization` combines two objects:

 - a `mesh`, which determines where the source pixels are placed. `RectangularUniform` uses a uniform grid.
 - a `regularization`, which smooths the reconstruction. `Constant` applies one smoothing strength everywhere.

The flux in every source pixel is solved for via linear algebra, so none of them are input here.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source = al.Galaxy(redshift=1.0, pixelization=pixelization)

"""
__Main Lens Galaxies__

The two co-dominant deflectors, each with its own light and mass profile, as in `multi_galaxy/fit.py`. Their
light uses linear profiles, so their intensities are solved rather than input.

The system's `ExternalShear` is held in its own `shear_galaxy` at the system centre, rather than attached to
either deflector.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
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
    bulge=al.lp_linear.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
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

"""
__Fit__

Create a `Tracer` from the galaxies and fit the dataset with it. Both deflectors are at the same redshift, so
their deflection fields are summed and ray tracing is single-plane.
"""
tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log likelihood = {fit.log_likelihood}")

"""
__Inversion__

The source reconstruction is held on the fit's `inversion`, as an array holding one flux value per source pixel.

The inversion's `reconstruction` is the solution vector over **every** linear object it solves, not just the source
pixels. Both deflectors here use `lp_linear` bulges, whose intensities are solved by the same linear algebra as the
source pixels, so `reconstruction` carries 2 entries more than the mesh has pixels. To count the source pixels, pull
the mapper out of the inversion's linear objects and read its own reconstruction.

The `subplot_fit_imaging` above already shows the reconstructed source alongside the data, model image and
residuals. `imaging/features/pixelization/fit.py` covers the inversion's internals — its linear objects, mapping
matrices and grids — in full. That script uses standard light profiles, so its source is the only linear object and
this distinction does not arise.
"""
mapper = fit.inversion.cls_list_from(cls=al.Mapper)[0]

print(
    f"Number of source pixels reconstructed = "
    f"{fit.inversion.reconstruction_dict[mapper].shape[0]}"
)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/pixelization/modeling.py` — fitting this model with a non-linear search.
 - `multi_galaxy/fit.py` — the multi-galaxy fit anatomy, including the summed deflection fields.
 - `imaging/features/pixelization/fit.py` — the full pixelization API walkthrough at galaxy scale.
"""
