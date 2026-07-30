"""
Plots: Weak
===========

This example shows how to plot a `WeakDataset` and a `FitWeak` fit.

Weak-lensing data is a catalogue of background galaxy positions, each with a measured shear
vector, rather than a 2D image. Its figures therefore show the shear field, its magnitude and
orientation, and profiles binned about the lens centre.

For an introduction to the plotting API refer to `guides/plot/start_here.py`. For the scientific
interpretation of these figures (what a good weak-lensing fit looks like, the tangential/cross
shear decomposition) refer to `scripts/weak/fit.py`. The final section documents the `Visualizer`,
which outputs these figures automatically during a model-fit.

__Contents__

- **Dataset:** Load the weak lensing dataset of galaxy positions and shear measurements.
- **Dataset Figures:** Plot the shear field, its magnitude, position angle and noise-map individually.
- **Dataset Subplot:** Plot all dataset quantities in one multi-panel subplot.
- **Fit:** Set up a tracer and fit the dataset with a `FitWeak` object.
- **Fit Subplot:** Plot the data, model and chi-squared mosaic of the fit.
- **Shear Profile:** Plot the binned tangential and cross shear profiles of the fit.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the simulated `WeakDataset` produced by `scripts/weak/simulator.py`: background source-galaxy
positions around a cluster-scale lens, each with a measured shear vector.
"""
dataset_path = Path("dataset") / "weak" / "simple"

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/weak/simulator.py"],
        check=True,
    )

dataset = al.from_json(file_path=dataset_path / "dataset.json")

print(dataset.info)

"""
__Dataset Figures__

The dataset's quantities are plotted individually: the shear field as headless quiver segments,
the shear magnitude, the shear position angle and the per-galaxy noise-map.
"""
aplt.plot_shear_yx_2d(shear_yx=dataset.shear_yx)

aplt.plot_ellipticities(shear_yx=dataset.shear_yx)

aplt.plot_phis(shear_yx=dataset.shear_yx)

aplt.plot_noise_map(dataset=dataset)

"""
__Dataset Subplot__

A multi-panel subplot of the dataset is produced with `aplt.subplot_weak_dataset()`, combining the
four figures above in one mosaic.
"""
aplt.subplot_weak_dataset(dataset=dataset)

"""
__Fit__

To plot a fit, we compose a tracer whose mass model matches the simulator's true values and fit
the dataset with a `FitWeak` object (the source galaxy has no light profile — weak lensing is
sensitive to the lens mass, not the source's appearance).
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=25.0,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(redshift=1.0)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitWeak(dataset=dataset, tracer=tracer)

"""
__Fit Subplot__

The `aplt.subplot_fit_weak()` mosaic shows the observed shear field, the model shear field, the
two overlaid, and the per-galaxy chi-squared map.
"""
aplt.subplot_fit_weak(fit=fit)

"""
__Shear Profile__

The `aplt.plot_shear_profile()` function bins the catalogue about a chosen centre and plots the
tangential shear profile of the data with the model overlaid as a line, together with the cross
component null test.
"""
aplt.plot_shear_profile(
    fit,
    centre=(0.0, 0.0),
    bins=8,
)

"""
__Visualizer__

During a model-fit (e.g. `search.fit(model=model, analysis=analysis)` in `modeling.py`), the fit
figures above are output to hard-disk automatically by the `Visualizer` attached to the `Analysis`
class:
"""
print(al.AnalysisWeak.Visualizer)

"""
At regular intervals during the non-linear search, and again once it finishes, the `Visualizer`
outputs the maximum likelihood fit's figures to the fit's output folder, under `image/`
(e.g. `output/<path_prefix>/<name>/image/`).

Which figures are output is controlled by the config file `config/visualize/plots.yaml`. The
machinery is described in full in `scripts/imaging/plot.py` — the same config-driven visualization
applies to every dataset type.
"""
