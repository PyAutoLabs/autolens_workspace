"""
Fit: Combined Strong + Weak Lensing
===================================

This script fits the combined strong+weak dataset simulated by `simulator.py` in this folder with a single
shared `Tracer`: the imaging data via `FitImaging` and the shear catalogue via `FitWeak`.

Because the two datasets are statistically independent measurements of the same mass distribution, their
joint log likelihood is simply the sum of the two individual log likelihoods — this additivity is all the
joint modeling in `modeling.py` needs, and this script makes it explicit before a non-linear search is
involved.

__Contents__

- **Dataset:** Load both datasets (auto-simulating them if missing) and mask the imaging data.
- **Tracer:** The single mass model shared by both fits.
- **Imaging Fit:** Fit the strong-lensing image.
- **Weak Fit:** Fit the shear catalogue.
- **Joint Likelihood:** The sum that a joint analysis samples.
- **Shear Profile:** Data vs model in the space cluster weak lensing is usually shown in.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load both halves of the combined dataset from `dataset/weak/strong_lensing/`.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
dataset_path = Path("dataset") / "weak" / "strong_lensing"

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/weak/features/strong_lensing/simulator.py"],
        check=True,
    )

dataset_imaging = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset_weak = al.from_json(file_path=dataset_path / "dataset.json")

"""
The imaging data is masked with the standard 3.0" circular mask — inside it live the arcs; the shear
catalogue carries the information outside it, to 10".
"""
mask = al.Mask2D.circular(
    shape_native=dataset_imaging.shape_native,
    pixel_scales=dataset_imaging.pixel_scales,
    radius=3.0,
)

dataset_imaging = dataset_imaging.apply_mask(mask=mask)

"""
__Tracer__

One tracer fits both datasets. We use the simulator's true parameters, so both fits below are "perfect"
up to noise — swap any parameter to see both likelihoods respond, which is exactly what a non-linear
search exploits in `modeling.py`.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.05, 0.05),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Imaging Fit__

The strong-lensing side is the standard `FitImaging` of every imaging example: model image, PSF convolution,
residuals and a Gaussian likelihood over the masked pixels.
"""
fit_imaging = al.FitImaging(dataset=dataset_imaging, tracer=tracer)

print(f"imaging log_likelihood : {fit_imaging.log_likelihood:.2f}")

aplt.subplot_fit_imaging(fit=fit_imaging, output_path=dataset_path, output_format="png")

"""
__Weak Fit__

The weak-lensing side is the `FitWeak` of `scripts/weak/fit.py`: the tracer's shear field evaluated at the
catalogue positions, compared to the measured shears over 2N independent components.
"""
fit_weak = al.FitWeak(dataset=dataset_weak, tracer=tracer)

print(f"weak log_likelihood    : {fit_weak.log_likelihood:.2f}")
print(
    f"weak chi_squared       : {fit_weak.chi_squared:.1f} "
    f"(expected ~{2 * dataset_weak.n_galaxies} for the true model)"
)

aplt.subplot_fit_weak(fit=fit_weak, output_path=dataset_path, output_format="png")

"""
__Joint Likelihood__

The datasets are independent (different galaxies, different noise), so the joint log likelihood of the
shared tracer is their sum. This one line is the entire statistical content of "combining strong and weak
lensing" — `modeling.py` wires it into PyAutoFit's factor-graph API so a non-linear search samples it.
"""
log_likelihood_joint = fit_imaging.log_likelihood + fit_weak.log_likelihood

print(f"joint log_likelihood   : {log_likelihood_joint:.2f}")

"""
__Shear Profile__

The tangential shear profile shows where the weak information lives: the model curve (from the same tracer
fitting the arcs at 1.6") is tested by the binned data points out to 10" — radii the imaging mask never
sees. The cross component scattering around zero is the standard B-mode systematics null test.
"""
aplt.plot_shear_profile(
    fit_weak,
    centre=(0.0, 0.0),
    bins=8,
    output_path=dataset_path,
    output_format="png",
)
