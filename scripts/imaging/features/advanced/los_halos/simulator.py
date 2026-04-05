"""
Simulator: Line-of-Sight Halos
==============================

This script simulates a strong gravitational lens including line-of-sight (LOS) dark matter halos
that perturb the lensed images via multi-plane ray tracing.

LOS halos are sampled from a cosmological halo mass function within a light-cone geometry, converted
to truncated NFW profiles, and placed on multiple redshift planes between the observer and the source.
A compensatory negative convergence (kappa) sheet is added to each plane to maintain mass conservation,
following the methodology of He et al. (2022, MNRAS 511, 3046).

Without the negative kappa sheets, LOS halos systematically over-lens the images because the total
convergence is not conserved. The negative sheets account for the smooth average contribution of
all halos, so that only the *fluctuations* above the mean affect the lensing.

__Contents__

**Model:** Compose the lens model fitted to the data.
**Dataset Paths:** The simulated dataset is output to the ``dataset/imaging/los_halos`` folder.
**Grid:** Define the 2D grid on which the image is evaluated and simulated.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**PSF:** A Gaussian PSF models the telescope optics.
**Simulator:** The simulator defines the exposure time, PSF, background sky level, and noise properties.
**LOS Configuration:** Parameters controlling the line-of-sight halo population.
**Sample LOS Halos:** The ``LOSSampler`` handles the full pipeline.
**Ray Tracing:** Define the main lens galaxy and source galaxy, then combine with the LOS galaxies to create a.
**Output:** Output the simulated dataset to .fits files.
**Visualize:** Output subplots and summary images as .png files for quick inspection.
**Tracer json:** Save the tracer as a .json file for reproducibility.
**LOS Diagnostics:** Save the LOS halo sample list and negative sheet values for post-simulation analysis.

__Model__

This script simulates ``Imaging`` of a galaxy-scale strong lens where:

 - The lens galaxy's total mass distribution is a ``PowerLaw`` and ``ExternalShear``.
 - The source galaxy's light is a ``SersicCore``.
 - Line-of-sight halos are ``NFWTruncatedSph`` profiles on multiple redshift planes.
 - Each redshift plane includes a ``MassSheet`` with negative kappa.
"""

from autoconf import jax_wrapper

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

from autolens.lens.los import LOSSampler, los_planes_from

"""
__Dataset Paths__

The simulated dataset is output to the ``dataset/imaging/los_halos`` folder.
"""
dataset_type = "imaging"
dataset_name = "los_halos"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2D grid on which the image is evaluated and simulated.
"""
grid = al.Grid2D.uniform(
    shape_native=(161, 161),
    pixel_scales=0.05,
)

"""
__Over Sampling__

Adaptive oversampling ensures the central bright regions are evaluated at higher resolution.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
__PSF__

A Gaussian PSF models the telescope optics.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(13, 13), sigma=0.05, pixel_scales=grid.pixel_scales
)

"""
__Simulator__

The simulator defines the exposure time, PSF, background sky level, and noise properties.
"""
simulator = al.SimulatorImaging(
    exposure_time=8000.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=666,
)

"""
__LOS Configuration__

Parameters controlling the line-of-sight halo population.

 - ``z_lens``: Redshift of the main lens galaxy.
 - ``z_source``: Redshift of the background source galaxy.
 - ``planes_before_lens`` / ``planes_after_lens``: Number of LOS planes in front of and behind the main lens.
   With [4, 4] we get 9 planes total (4 in front, the lens plane, and 4 behind).
 - ``m_min`` / ``m_max``: Halo mass range in solar masses (M_sun).
 - ``cone_radius_arcsec``: Angular radius of the light cone in arcseconds.
 - ``c_scatter``: Log-normal scatter in the concentration-mass relation (dex).
 - ``truncation_factor``: Overdensity factor defining the truncation radius (e.g. 100 for r_100).
 - ``seed``: Random seed for reproducibility.
"""
z_lens = 0.5
z_source = 1.0

planes_before_lens = 4
planes_after_lens = 4

m_min = 1e7
m_max = 1e10

cone_radius_arcsec = 5.0
c_scatter = 0.15
truncation_factor = 100.0

seed = 42

"""
__Mass Function and Mass-Concentration Coefficients__

The ``LOSSampler`` can compute these from the ``hmf`` and ``colossus`` libraries, but here we
provide pre-computed values for each plane to avoid those dependencies.

Each row contains ``[A, B]`` where:

 - For the mass function: ``log10(dn/dm) = A * log10(m) + B``
 - For the mass-concentration relation: ``c(m) = A * log10(m) + B``

These coefficients were computed using the Sheth-Mo-Tormen mass function and the Ludlow+16
concentration-mass relation for Planck15 cosmology at the redshift of each plane centre.

If you have ``hmf`` and ``colossus`` installed, you can pass ``cosmology_astropy`` and
``cosmology_name_colossus`` to ``LOSSampler`` instead, and it will compute these automatically.
"""
_, plane_centres = los_planes_from(
    z_lens=z_lens,
    z_source=z_source,
    planes_before_lens=planes_before_lens,
    planes_after_lens=planes_after_lens,
)

n_planes = len(plane_centres)

print(f"Number of LOS planes: {n_planes}")
print(f"Plane centres: {plane_centres}")

try:
    from autolens.lens.los import mass_function_ab_from, mass_concentration_ab_from
    from astropy.cosmology import Planck15 as astropy_planck15

    mass_function_coefficients = np.zeros((n_planes, 2))
    mass_concentration_coefficients = np.zeros((n_planes, 2))

    for i, z in enumerate(plane_centres):
        mass_function_coefficients[i] = mass_function_ab_from(
            redshift=z, cosmology_astropy=astropy_planck15
        )
        mass_concentration_coefficients[i] = mass_concentration_ab_from(redshift=z)

    print("Computed mass function and mass-concentration coefficients from hmf/colossus.")

except ImportError:
    print("hmf/colossus not available, using approximate pre-computed coefficients.")

    mass_function_coefficients = np.tile([-1.9, 8.0], (n_planes, 1))
    mass_concentration_coefficients = np.tile([-3.0, 40.0], (n_planes, 1))

"""
__Sample LOS Halos__

The ``LOSSampler`` handles the full pipeline:

 1. Slices the light cone into redshift planes.
 2. For each plane, samples halo masses, positions, and concentrations.
 3. Converts each halo to an ``NFWTruncatedSph`` via physical-to-lensing unit conversion.
 4. Computes the negative kappa sheet for each plane.
 5. Returns a list of ``Galaxy`` objects ready for the ``Tracer``.
"""
from autogalaxy.cosmology import Planck15

cosmology = Planck15()

sampler = LOSSampler(
    z_lens=z_lens,
    z_source=z_source,
    planes_before_lens=planes_before_lens,
    planes_after_lens=planes_after_lens,
    m_min=m_min,
    m_max=m_max,
    cone_radius_arcsec=cone_radius_arcsec,
    c_scatter=c_scatter,
    truncation_factor=truncation_factor,
    cosmology=cosmology,
    mass_function_coefficients=mass_function_coefficients,
    mass_concentration_coefficients=mass_concentration_coefficients,
    seed=seed,
)

los_galaxies = sampler.galaxies_from()

n_halos = sum(
    1
    for g in los_galaxies
    if hasattr(g, "mass") and isinstance(g.mass, al.mp.NFWTruncatedSph)
)
n_sheets = sum(
    1
    for g in los_galaxies
    if hasattr(g, "mass_sheet") and isinstance(g.mass_sheet, al.mp.MassSheet)
)

print(f"Sampled {n_halos} LOS halos across {n_sheets} planes.")

for g in los_galaxies:
    if hasattr(g, "mass_sheet") and isinstance(g.mass_sheet, al.mp.MassSheet):
        print(f"  Plane z={g.redshift:.4f}: kappa_neg = {g.mass_sheet.kappa:.6e}")

"""
__Ray Tracing__

Define the main lens galaxy and source galaxy, then combine with the LOS galaxies
to create a multi-plane ``Tracer``.

The ``Tracer`` automatically groups galaxies by redshift into planes and performs
multi-plane ray tracing through all of them.
"""
lens_galaxy = al.Galaxy(
    redshift=z_lens,
    mass=al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.059, -0.027),
        slope=2.264,
        einstein_radius=1.6,
    ),
    shear=al.mp.ExternalShear(gamma_1=0.0, gamma_2=0.0),
)

source_galaxy = al.Galaxy(
    redshift=z_source,
    bulge=al.lp.SersicCore(
        centre=(0.02, -0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=75.0),
        intensity=1.5,
        effective_radius=0.15,
        sersic_index=3.5,
        radius_break=0.025,
    ),
)

all_galaxies = los_galaxies + [lens_galaxy, source_galaxy]

tracer = al.Tracer(galaxies=all_galaxies)

"""
We can plot the tracer's image to see the combined lensing effect of the main lens
and all LOS halos.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image with LOS Halos")

"""
By passing the tracer and grid to the simulator, we create the simulated CCD imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Output__

Output the simulated dataset to .fits files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__

Output subplots and summary images as .png files for quick inspection.
"""
aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

aplt.subplot_tracer(tracer=tracer, grid=grid, output_path=dataset_path, output_format="png")
aplt.subplot_galaxies_images(tracer=tracer, grid=grid, output_path=dataset_path, output_format="png")

"""
__Tracer json__

Save the tracer as a .json file for reproducibility.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__LOS Diagnostics__

Save the LOS halo sample list and negative sheet values for post-simulation analysis.

A useful validation check is to compare the deflection field of the full tracer (with LOS halos)
against a smooth tracer (without LOS halos). If the negative kappa sheets are correct, the
*difference* in deflection angles should not systematically point toward the centre of the image.
"""
halo_info = []
sheet_info = []

for g in los_galaxies:
    if hasattr(g, "mass") and isinstance(g.mass, al.mp.NFWTruncatedSph):
        halo_info.append([
            g.redshift,
            g.mass.centre[0],
            g.mass.centre[1],
            g.mass.kappa_s,
            g.mass.scale_radius,
            g.mass.truncation_radius,
        ])
    elif hasattr(g, "mass_sheet") and isinstance(g.mass_sheet, al.mp.MassSheet):
        sheet_info.append([g.redshift, g.mass_sheet.kappa])

if len(halo_info) > 0:
    np.save(dataset_path / "los_halo_list.npy", np.array(halo_info))
    print(f"Saved {len(halo_info)} halo parameters to los_halo_list.npy")

if len(sheet_info) > 0:
    np.save(dataset_path / "los_sheet_values.npy", np.array(sheet_info))
    print(f"Saved {len(sheet_info)} sheet values to los_sheet_values.npy")

"""
The dataset can be viewed in the folder ``autolens_workspace/dataset/imaging/los_halos``.
"""
