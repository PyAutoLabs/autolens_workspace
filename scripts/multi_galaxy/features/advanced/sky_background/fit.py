"""
Fits: Sky Background (Multi Galaxy)
===================================

This script fits a multi-galaxy strong lens whose data still contains the sky background, without a non-linear
search, so the `DatasetModel` API can be seen directly.

Everything is set to its simulated value, including the sky, so this is the fit the model in
`modeling.py` is trying to find.

__Contents__

- **Dataset:** Load the multi-galaxy dataset that is fitted.
- **Mask:** Standard set up of the mask that is fitted.
- **Centres:** The centres of the co-dominant deflectors.
- **Over Sampling:** Adaptive over-sampling at every deflector centre.
- **Main Lens Galaxies:** The two deflectors, at their simulated values.
- **Fit:** Fit with a `DatasetModel` carrying the true sky level.
- **Sky Omitted:** The same fit with the sky left out.
- **Wrap Up:** Where to go next.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light is a `Sersic` and its mass an `Isothermal`, both at their simulated values.
 - The system has a single overall `ExternalShear` at the system centre.
 - The source's light is a `SersicCore`.
 - The sky background is a `DatasetModel` carrying the true sky level.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/fit.py` for the multi-galaxy fit anatomy and
`imaging/features/advanced/sky_background/fit.py` for the galaxy-scale walkthrough.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `sky_background` multi-galaxy dataset — the `simple` pair with the sky left in.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "sky_background"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/sky_background/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

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
__Main Lens Galaxies__

The two co-dominant deflectors and the source, at the values `simulator.py` used.
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

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source_galaxy])

"""
__Fit__

The sky is passed to the fit as a `DatasetModel`, not as a galaxy — it is a property of the data rather than of
anything being lensed. `background_sky_level` is set to the value `simulator.py` used.
"""
dataset_model = al.DatasetModel(background_sky_level=5.0)

fit = al.FitImaging(dataset=dataset, tracer=tracer, dataset_model=dataset_model)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log likelihood with the sky modelled = {fit.log_likelihood}")

"""
__Sky Omitted__

The same tracer fitted without a `DatasetModel`, so the sky is assumed to be zero.

Everything about the lens is identical between the two fits. The only difference is whether the sky is accounted
for, so the residual map below shows what the light profiles would otherwise be asked to absorb — and it covers
both deflectors, not one.
"""
fit_no_sky = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit_no_sky)

print(f"Log likelihood with the sky omitted  = {fit_no_sky.log_likelihood}")

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/sky_background/modeling.py` — fitting the sky rather than fixing it.
 - `multi_galaxy/fit.py` — the multi-galaxy fit anatomy, on sky-subtracted data.
 - `imaging/features/advanced/sky_background/fit.py` — the galaxy-scale walkthrough.
"""
