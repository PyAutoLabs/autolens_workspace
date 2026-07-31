"""
Fits: Shapelets (Multi Galaxy)
==============================

This script fits a multi-galaxy strong lens with a shapelet source, without a non-linear search, so the `Basis`
API and the solved shapelet intensities can be inspected directly.

__Contents__

- **Dataset:** Load the multi-galaxy dataset that is fitted.
- **Extra Galaxies Noise Scaling:** Scale the contaminating galaxy's light out of the fit.
- **Mask:** Standard set up of the mask that is fitted.
- **Centres:** The centres of the co-dominant deflectors.
- **Over Sampling:** Adaptive over-sampling at every deflector centre.
- **Basis:** Build the shapelet basis by hand.
- **Main Lens Galaxies:** The two deflectors, at their simulated values.
- **Fit:** Fit the tracer to the dataset.
- **Intensities:** Read the solved intensity of each shapelet.
- **Basis Image:** Plot the basis functions themselves.
- **Wrap Up:** Where to go next.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light is a `Sersic` and its mass an `Isothermal`, both at their simulated values.
 - The system has a single overall `ExternalShear` at the system centre.
 - The source's light is a `Basis` of linear `ShapeletPolar` profiles, whose intensities are solved.

Because the deflectors' light uses ordinary (non-linear) profiles, the only linear objects in the fit are the
shapelets — so everything the inversion solves belongs to the source.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/fit.py` for the multi-galaxy fit anatomy and
`imaging/features/advanced/shapelets/fit.py` for the galaxy-scale walkthrough of the shapelet API.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt
from autogalaxy.profiles.plot.basis_plots import subplot_image as subplot_basis_image

"""
__Dataset__

The `simple` multi-galaxy dataset, the same one fitted by `multi_galaxy/fit.py`.

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
__Mask__

The standard 3.0" circular mask.
"""
mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Over Sampling__

The adaptive over-sampling scheme, centred on every deflector.
"""
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 1],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Basis__

Build the shapelet basis by hand, rather than through the `af.Model` machinery `modeling.py` uses.

Each shapelet is indexed by two integers, `n` and `m`, which set how many oscillations it has radially and
azimuthally. The loop below enumerates every valid `(n, m)` pair up to `total_n`, which is what "a basis of order
`total_n`" means.

Every shapelet shares one `centre`, one `ell_comps` and one `beta`. Here they are given concrete values — the
source's simulated centre and a reasonable guess at its size — where `modeling.py` samples them.

`beta` is the basis's characteristic scale. Too small and the high orders are spent describing the core; too
large and the basis cannot resolve it.
"""
total_n = 5
total_m = sum(range(2, total_n + 1)) + 1

source_centre = (0.0, 0.03)
beta = 0.2

shapelets_bulge_list = [
    al.lp_linear.ShapeletPolar(
        n=0,
        m=0,
        centre=source_centre,
        ell_comps=(0.0, 0.0),
        beta=beta,
    )
]

n_count = 1
m_count = -1

for i in range(total_n + total_m):
    shapelets_bulge_list.append(
        al.lp_linear.ShapeletPolar(
            n=n_count,
            m=m_count,
            centre=source_centre,
            ell_comps=(0.0, 0.0),
            beta=beta,
        )
    )

    m_count += 2

    if m_count > n_count:
        n_count += 1
        m_count = -n_count

source_bulge = al.lp_basis.Basis(profile_list=shapelets_bulge_list)

source = al.Galaxy(redshift=1.0, bulge=source_bulge)

print(f"Number of shapelets in the basis = {len(shapelets_bulge_list)}")

"""
__Main Lens Galaxies__

The two co-dominant deflectors at their simulated values, with ordinary `Sersic` light profiles rather than
linear ones. That keeps the shapelets the only linear objects in the fit, so the intensities read out below are
unambiguously the source's.
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

"""
__Fit__

`use_positive_only_solver=False` is required: shapelets of order `n > 0` are negative over part of their extent,
and forcing the solve positive would prevent the basis from summing to anything but a bump.
"""
tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source])

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    settings=al.Settings(use_positive_only_solver=False),
)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log likelihood = {fit.log_likelihood}")

"""
__Intensities__

`linear_light_profile_intensity_dict` is keyed by the profile object itself, so a shapelet's solved intensity is
read back with the profile that produced it — no need to know its position in the reconstruction vector.
"""
print(
    f"Intensity of the source's first shapelet = "
    f"{fit.linear_light_profile_intensity_dict[source_bulge.profile_list[0]]}"
)

"""
The fit also exposes a `Tracer` in which every linear profile has been replaced by an ordinary one carrying its
solved intensity, which is what the plotting functions need.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

"""
__Basis Image__

Plot the basis functions themselves. The first is a simple bump; the higher orders oscillate, and it is their sum
that describes structure the bump alone cannot.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

subplot_basis_image(basis=tracer.galaxies[-1].bulge, grid=grid)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/shapelets/modeling.py` — fitting this basis with a non-linear search.
 - `multi_galaxy/fit.py` — the multi-galaxy fit anatomy, including the summed deflection fields.
 - `multi_galaxy/features/pixelization/fit.py` — a free-form source, for structure a centred basis cannot
   describe.
 - `imaging/features/advanced/shapelets/fit.py` — the galaxy-scale walkthrough of the shapelet API.
"""
