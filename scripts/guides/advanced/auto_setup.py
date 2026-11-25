"""
Auto Setup
==========

Every likelihood evaluation in a lens model-fit can be computed in several ways. It can run on JAX or numba, on a
CPU or a GPU, with the dense mapping-matrix linear algebra or a precomputed sparse operator, and with one or many
CPU cores working in parallel. Which of these is fastest depends on the dataset, the model and the hardware.

Passing `setup="auto"` to an analysis and `number_of_cores="auto"` to a search makes **PyAutoLens** choose for you.
The choices come from profiling runs of every likelihood on a many-core CPU node and on an NVIDIA A100 GPU, so the
same Python script runs the fastest way available on your laptop, on a CPU node of an HPC cluster and on a GPU node.

This guide explains what the automatic setup decides, the setups it picks for each kind of dataset, how to read the
lines it prints when a fit starts, how to override it and how to use it outside a model-fit.

__Contents__

- **What Auto Decides:** The choices the automatic setup makes, and the ones it never makes.
- **Designated Setups:** The setup chosen for each kind of dataset, on a CPU node and on an A100 GPU.
- **Dataset:** Load and mask the imaging dataset used in this guide.
- **Model:** A lens model with a pixelized source, so the setup has a real choice to make.
- **Fit With Auto:** Run a fit with `setup="auto"` and `number_of_cores="auto"`.
- **Reading The Output:** What the `Setup`, `Parallel` and `Estimate` lines tell you.
- **Overriding The Setup:** Choosing the setup yourself, in code or with the `PYAUTO_SETUP` environment variable.
- **Estimating Run Time:** Estimating how long a fit takes before you run it, with `al.estimate_runtime`.
- **Outside Modeling:** Getting the same speed-up for single fits, with `lazy_sparse_operator`.
- **Dataset Operations:** How over-sampling, masking and noise scaling keep or rebuild the setup.
- **Output Files:** Where the chosen setup is saved, and how results loaded later reuse it.
- **Factor Graphs:** Fitting several datasets together with one shared setup.
- **Wrap Up:** Summary and where to go next.

__What Auto Decides__

The automatic setup chooses how each likelihood is computed:

- **Backend:** JAX or numba.
- **Device:** CPU or GPU.
- **Operator:** the dense mapping-matrix linear algebra, or a precomputed sparse operator for pixelized sources.
- **Precision:** 64-bit (every designated setup is currently fp64).
- **Transformer:** for interferometer data, the w-tilde or NUFFT route.
- **Threads:** how many threads numba and JAX use.

With `number_of_cores="auto"` the search also chooses how many likelihoods run in parallel. That is the number of
detected CPU cores for a numba setup and 1 for a JAX setup (JAX parallelizes inside one process instead). When the
analysis knows how much memory each likelihood needs, the number is capped so the workers fit in memory.

The automatic setup never changes your model, your non-linear search or the resources you request from an HPC
cluster. It decides how the likelihood is computed, never what is computed, so the result of a fit is the same
whichever setup runs it.

The setup is opt-in in this release. If you do not pass `setup`, the analysis behaves as before and prints one line
saying what `setup="auto"` would have chosen.

__Designated Setups__

Each choice is named by a setup id, `<backend>_<device>_<operator>_<precision>[_<transformer>]`, for example
`numba_cpu_sparse_fp64` or `jax_gpu_dense_fp64`. These are the setups `setup="auto"` picks:

| Dataset and source | CPU node | A100 GPU |
|---|---|---|
| Imaging, pixelized source | `numba_cpu_sparse_fp64` | `jax_gpu_dense_fp64` |
| Imaging, MGE or parametric source | `jax_cpu_dense_fp64` | `jax_gpu_dense_fp64` |
| Interferometer, pixelized source | `numba_cpu_sparse_fp64_wtilde` | `jax_gpu_sparse_fp64_wtilde` |
| Interferometer, MGE or parametric source | `jax_cpu_dense_fp64_nufft` | `jax_gpu_dense_fp64_nufft` |
| Group, multi galaxy, multi dataset | as imaging | as imaging |
| Datacube, pixelized source | `numba_cpu_sparse_fp64_wtilde` | CPU setup used |
| Point source and cluster | `jax_cpu_na_fp64` | `jax_gpu_na_fp64` |
| Weak lensing | `jax_cpu_dense_fp64` | `jax_gpu_dense_fp64` |

Whether a source is pixelized comes from the model: a source with a mesh uses the pixelized row (for its mesh type),
and a model with no mesh uses the MGE row. The operator `na` means the likelihood has no linear-algebra operator.

The rows were measured on a CPU node with at least 16 cores and on an A100 GPU. On other hardware, such as a laptop or
a different GPU, the nearest row is used and the printed setup is marked "extrapolated".
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al

"""
__Dataset__

We load and mask the imaging dataset used throughout this guide.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/imaging/features/no_lens_light/simulator.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model__

We fit an isothermal mass model with external shear and a source reconstructed on a rectangular mesh. Because the
source has a mesh, the automatic setup uses the pixelized imaging row of the table above.
"""
mesh_shape = (20, 20)

lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)

field = af.Model(al.MassField, redshift=0.5, shear=af.Model(al.mp.ExternalShear))

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

"""
__Fit With Auto__

We pass `setup="auto"` to the analysis and `number_of_cores="auto"` to the search. Nothing else in the script
changes: the same code runs on a laptop, a CPU node and a GPU node.
"""
search = af.Nautilus(
    path_prefix=Path("guides") / "advanced",
    name="auto_setup",
    unique_tag=dataset_name,
    n_live=50,
    number_of_cores="auto",
)

analysis = al.AnalysisImaging(dataset=dataset, setup="auto")

result = search.fit(model=model, analysis=analysis)

"""
__Reading The Output__

When the fit starts, a short block like this is printed (this one is from an 8-core laptop):

```
Automatic compute setup (CPU/GPU choice from setup="auto"; override with PYAUTO_SETUP or setup=...):
Setup:    numba · CPU · sparse operator · fp64   (rule imaging/rectangular · extrapolated: 8 cores, rules measured on ≥16-core nodes)
Parallel: 8 workers (auto: detected cores)
Estimate: ~10 min on this hardware   (283 ms/eval, euclid cell · Nautilus n_live=50 ≈ 7.3e3 evals, heuristic · η=0.55)
```

The first line says the setup was chosen automatically. If you set it yourself it reads
`Compute setup (set explicitly by setup=...):`, and if it came from the environment variable it reads
`Compute setup (set by PYAUTO_SETUP=<setup_id>):`. The lines are printed with the search's other log lines when the
fit starts, so you may see a few unrelated lines in between.

- **Setup:** the chosen backend, device, operator and precision, and the rule that chose them. "extrapolated"
  means your hardware differs from the hardware the rule was measured on (here, fewer than 16 cores), so the
  choice is the closest match rather than a measured best.
- **Parallel:** how many likelihoods run in parallel. A numba setup uses one worker per detected core; a JAX setup
  prints `1 worker (JAX on CPU)`. If you pass a number yourself it prints, for example, `2 workers (requested 2)`.
- **Estimate:** a rough run time for this fit on this hardware. It combines the measured time per likelihood
  evaluation for a similar dataset with a heuristic for how many evaluations Nautilus needs, and the parallel
  efficiency η of the workers. Interferometer and weak lensing estimates are marked "uncalibrated", because their
  run times have not yet been checked against real fits. Searches other than Nautilus print
  `Estimate: none for <Search>` and the reason.

Treat the estimate as an order of magnitude: it tells you whether a fit takes minutes or days.

__Overriding The Setup__

You can choose the setup yourself by passing a setup id, or an `al.Setup` object, instead of `"auto"`. Here we force
the numba sparse setup and two parallel workers.
"""
search = af.Nautilus(
    path_prefix=Path("guides") / "advanced",
    name="auto_setup_override",
    unique_tag=dataset_name,
    n_live=50,
    number_of_cores=2,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    setup="numba_cpu_sparse_fp64",
)

result = search.fit(model=model, analysis=analysis)

"""
The setup id and the `al.Setup` object are interchangeable:
"""
setup = al.Setup.from_setup_id("numba_cpu_sparse_fp64")

print(setup)
print(setup.setup_id)

"""
You can also override the setup without editing a script, with the `PYAUTO_SETUP` environment variable. It takes a
setup id, or one of `auto`, `cpu` or `gpu`:

```
PYAUTO_SETUP=numba_cpu_sparse_fp64 python my_script.py
PYAUTO_SETUP=gpu python my_script.py
```

`PYAUTO_SETUP=<setup_id>` replaces the setup every `setup="auto"` analysis chooses. `cpu` and `gpu` keep the
automatic choice but tell it which hardware class to choose for.

If you set `JAX_PLATFORMS=cpu`, JAX cannot see a GPU, so the automatic setup chooses the CPU row.

__Estimating Run Time__

`al.estimate_runtime` gives the same estimate before you run anything. It takes the dataset, the model, the search,
a setup and the number of cores, and returns an estimate whose `line()` is the `Estimate:` line above.
"""
search = af.Nautilus(n_live=150)

estimate = al.estimate_runtime(
    dataset=dataset,
    model=model,
    search=search,
    setup="numba_cpu_sparse_fp64",
    number_of_cores=16,
)

if estimate is not None:
    print(estimate.line())

"""
This is useful for deciding how many cores, or which kind of node, to request on an HPC cluster before submitting a
job: compare the estimate for `numba_cpu_sparse_fp64` on 16 cores with `jax_gpu_dense_fp64` on one GPU.

__Outside Modeling__

The automatic setup applies to model-fits. A single fit of a tracer you have written down, such as
`al.FitImaging(dataset=dataset, tracer=tracer)`, uses the dense linear algebra unless you attach an operator
yourself.

`al.Settings(lazy_sparse_operator="auto")` gives such fits the same speed-up. The first pixelized fit builds the
numba sparse operator for the dataset, caches it, and every later fit of that dataset reuses it. You can turn this
on for every script with `lazy_sparse_operator: "auto"` under `inversion:` in `config/general.yaml`, and
`PYAUTO_SETUP=auto` also turns it on. It is off by default.
"""
tracer = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=1.6,
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            ),
        ),
        al.Galaxy(
            redshift=1.0,
            pixelization=al.Pixelization(
                mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
                regularization=al.reg.Constant(coefficient=1.0),
            ),
        ),
    ]
)

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    settings=al.Settings(lazy_sparse_operator="auto"),
)

print(fit.log_likelihood)

"""
The first such fit prints a line like this:

```
Automatic compute setup: built the numba sparse operator for this dataset (2828 image pixels, 0.24 s; cached; lazy_sparse_operator: auto)
```

__Dataset Operations__

A dataset carries its setup and operator through the usual operations:

- `apply_over_sampling` keeps both, so you can change over-sampling before or after the setup is applied.
- `apply_mask` and `apply_noise_scaling` change the data the operator was built from. An operator you attached by
  hand (for example with `apply_sparse_operator_cpu()`) is dropped with a warning, while a setup applied with
  `dataset.apply_setup(...)` is re-applied, rebuilding its operator for the new mask or noise-map.

__Output Files__

A fit with a setup writes it to `files/setup.json` in its output folder, and the number of workers to
`files/parallel.json`. When you load results later with the aggregator, the analysis reapplies the setup from
`setup.json`, so reloaded fits are computed the same way as the original. It prints:

```
Compute setup: reapplied numba_cpu_sparse_fp64 from the search's setup.json
```

__Factor Graphs__

When several datasets are fitted together as a factor graph (see `multi_dataset/` and the graphical model
examples), every factor must resolve to the same setup. Pass `setup="auto"` to every analysis: datasets of the same
kind resolve to the same setup. If they resolve different setups (for example an imaging and an interferometer
dataset), the fit stops at the start with an error listing each factor and its setup. Pass one explicit setup id to
every analysis instead.

__Wrap Up__

You have seen how to:

- run a fit with `setup="auto"` and `number_of_cores="auto"`, and read the lines it prints;
- override the setup with a setup id, an `al.Setup` or `PYAUTO_SETUP`;
- estimate run time with `al.estimate_runtime`;
- speed up single fits with `lazy_sparse_operator="auto"`.

For running the same script on CPU and GPU nodes of an HPC cluster, see `guides/hpc/example_cpu_and_gpu.py`. For
an example that forces the numba sparse setup in a full pipeline, see
`imaging/features/pixelization/cpu_fast_modeling.py`.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
