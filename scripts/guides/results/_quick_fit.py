"""
Results: Quick Fit Helper
=========================

Internal helper invoked via subprocess from the tutorials in this folder.
Produces two fast, capped Nautilus fits at ``output/results_folder/`` so the
aggregator and workflow examples have a populated results directory to read
from.

Idempotent: exits immediately if ``output/results_folder/`` already contains
the two completed imaging fits, so concurrent or repeated invocations are
cheap.

Not a tutorial. The model and dataset mirror those used in ``start_here.py``
(``simple__no_lens_light`` imaging, isothermal lens + MGE source), but the
search is hard-capped at ``n_like_max=300`` likelihood evaluations rather
than running to convergence. This produces a shallow but valid posterior
fast enough to fit inside the per-script CI timeout.
"""

import shutil
import sys
from pathlib import Path


def has_workflow_results(results_path):
    return (
        len(list(results_path.glob("**/image/dataset.fits"))) >= 2
        and len(list(results_path.glob("**/files/latent/latent_summary.json"))) >= 2
        and len(list(results_path.glob("**/image/fit.png"))) >= 2
        and len(list(results_path.glob("**/image/fit.fits"))) >= 2
    )


results_path = Path("output") / "results_folder"
if has_workflow_results(results_path):
    sys.exit(0)

if results_path.exists():
    shutil.rmtree(results_path)

import os

# The aggregator tutorials that invoke this helper read image/dataset.fits via
# fit.value("dataset"). Visualization-skipping environment variables suppress
# the visualizer that writes that file, so neutralize them here.
os.environ.pop("PYAUTO_SKIP_VISUALIZATION", None)
os.environ.pop("PYAUTO_SKIP_FIT_OUTPUT", None)
os.environ.pop("PYAUTO_FAST_PLOTS", None)

import autofit as af
import autolens as al
from autolens import conf

# This deliberately shallow helper must retain its exploratory samples because
# the results tutorials demonstrate indexed sample access.
conf.instance["output"]["samples_weight_threshold"] = None

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
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            mass=al.mp.Isothermal,
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=bulge, disk=None),
    ),
)


class LatentShear(al.Latent):
    """
    Custom latent catalogue reporting the lens external-shear magnitude and
    angle for the workflow CSV example.
    """

    @staticmethod
    def keys(analysis):
        return [
            "galaxies.lens.shear.magnitude",
            "galaxies.lens.shear.angle",
        ]

    @staticmethod
    def variables(analysis, parameters, model):
        instance = model.instance_from_vector(vector=parameters)

        import jax.numpy as jnp

        magnitude, angle = al.convert.shear_magnitude_and_angle_from(
            gamma_1=instance.galaxies.lens.shear.gamma_1,
            gamma_2=instance.galaxies.lens.shear.gamma_2,
            xp=jnp,
        )

        return (magnitude, angle)


class AnalysisLatent(al.AnalysisImaging):
    Latent = LatentShear


analysis = AnalysisLatent(dataset=dataset, use_jax=True)

# Two fits are written to `results_folder/`:
#
#  - The first uses the bare `dataset_name` as its `unique_tag`. This is the
#    canonical fit that the single-result tutorials (`start_here.py`,
#    `aggregator/galaxies_fits.py`, `aggregator/samples.py`) resume: they build
#    the *same* model and search below, so `search.fit(...)` returns this
#    completed fit instead of redoing the search.
#  - The second (`_1`) gives the aggregator tutorials a second result to iterate
#    over, demonstrating how the `Aggregator` loads many fits from one folder.
#
# Both share the model, search and analysis, so both retain the full samples
# (`samples_weight_threshold = None` above) needed for indexed sample access.
for unique_tag in [dataset_name, f"{dataset_name}_1"]:
    search = af.Nautilus(
        path_prefix=Path("results_folder"),
        name="results",
        unique_tag=unique_tag,
        n_live=100,
        n_batch=50,
        iterations_per_quick_update=10000,
        n_like_max=300,
    )

    search.fit(model=model, analysis=analysis)

"""
__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape. Results guides must produce real
(reduced) samples so downstream aggregator reads (data_fitting, queries,
...) are non-empty.

ENV: full_datasets real_search
"""
