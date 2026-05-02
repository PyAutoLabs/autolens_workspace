"""
Results: Quick Fit Helper
=========================

Internal helper invoked via subprocess from the tutorials in this folder
(``start_here.py`` and everything under ``aggregator/``). Produces a fast,
capped Nautilus fit at ``output/results_folder/`` so the aggregator examples
have a populated results directory to read from.

Idempotent: exits immediately if ``output/results_folder/`` already exists,
so concurrent or repeated invocations are cheap.

Not a tutorial. The model and dataset mirror those used in ``start_here.py``
(``simple__no_lens_light`` imaging, isothermal lens + MGE source), but the
search is hard-capped at ``n_like_max=300`` likelihood evaluations rather
than running to convergence. This produces a shallow but valid posterior
fast enough to fit inside the per-script CI timeout.
"""

import sys
from pathlib import Path

results_path = Path("output") / "results_folder"
if results_path.exists():
    sys.exit(0)

import autofit as af
import autolens as al

dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess

    subprocess.run(
        [sys.executable, "scripts/imaging/features/no_lens_light/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=bulge, disk=None),
    ),
)

search = af.Nautilus(
    path_prefix=Path("results_folder"),
    name="results",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
    n_like_max=300,
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

search.fit(model=model, analysis=analysis)
