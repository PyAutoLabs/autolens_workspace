"""
Using JAX
=========

**PyAutoLens** runs on either **NumPy** (the default) or **JAX** (Google's array library with GPU support and
just-in-time compilation). JAX makes lens modeling 10-100x faster on large grids — sometimes more on GPU — so the
library is built to use it automatically wherever it helps.

The `start_here.py` introduction covers the one thing every user needs to know: install the JAX extra
(`pip install autolens[jax]` on Python 3.11+) and lens modeling uses JAX automatically. This guide covers the
technical detail behind that, and the situations where you interact with JAX directly.

__Contents__

- **Auto-Enabled Modeling:** What "JAX is used automatically" means under the hood for model-fits.
- **Disabling JAX:** Forcing the NumPy path per-analysis or globally, and why that helps when debugging.
- **Writing @jax.jit Yourself:** Custom simulations and custom likelihood functions.
- **Custom Likelihood Functions:** Writing your own log likelihood — via the `Analysis` object, by hand, and via `Fitness`.
- **JIT-ing Library Methods:** The advanced path wrapping library methods like `tracer.image_2d_from` directly.
- **Return-Type Contract:** What `jax.Array` data inside results means for plotting, saving and arithmetic.

__Auto-Enabled Modeling__

If JAX is installed, the `AnalysisImaging`, `AnalysisInterferometer`, and `AnalysisPoint` classes default to
`use_jax=True`. The non-linear search driver (Nautilus, dynesty, ...) batches parameter vectors and evaluates the
likelihood through `jax.vmap(jax.jit(...))` internally. You'll see a one-time log line like
`JAX: Applying vmap and jit to likelihood function -- may take a few seconds.` the first time a search starts;
that's the JIT compile kicking in, after which evaluations re-use the compiled trace.

If JAX is not installed, the analysis warns once and falls back to NumPy automatically.

__Disabling JAX__

You can force the NumPy path explicitly with `al.AnalysisImaging(dataset=dataset, use_jax=False)`, or globally by
setting the environment variable `PYAUTO_DISABLE_JAX=1`.

This is useful when debugging: NumPy stack traces are easier to read than JAX traces, and you can drop a debugger
or `print` statement into code that JAX would otherwise trace and compile.

__Writing @jax.jit Yourself__

Two situations call for it:

1. **Custom simulations.** Pass `use_jax=True` to the simulator constructor to run the image calculation through
   JAX, for parameter sweeps, mock-data studies or batch figure generation:

   ```python
   simulator = al.SimulatorImaging(
       exposure_time=300.0, psf=psf, background_sky_level=0.1, use_jax=True
   )

   dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
   ```

   **Wrapping that call in `@jax.jit` does not currently work.** Two things stop it, and it is worth knowing
   which is which:

   - **You must register the pytrees yourself first.** Nothing in the library does it for you, and nothing can:
     JAX flattens a jitted function's arguments at trace time, *before* entering the callee, so a simulator that
     registered internally would already be too late. The one-time call is
     `autolens.jax.register_tracer_classes(tracer)`.
   - **Even with that, the jitted simulator call fails inside autoarray** on array sites that do not yet thread
     `xp` — see PyAutoLabs/PyAutoArray for the tracked issue. Until it is fixed, use the eager call above.

   Note the eager call returns a dataset whose `.data.array` is a `numpy.ndarray`, not a `jax.Array`.

   `scripts/point_source/simulator.py` and `scripts/cluster/simulator.py` show the registration step in a
   `PointSolver` context, where `@jax.jit` *does* work and is the reason those scripts are fast.

2. **Custom likelihood functions** that you assemble by hand rather than reaching for `AnalysisImaging`. Same
   shape: `@jax.jit` around your own `def log_likelihood(instance): ...`. The next section works this through.

__Custom Likelihood Functions__

The `likelihood_function.py` script in each dataset-type folder (`scripts/imaging/likelihood_function.py`,
`scripts/interferometer/likelihood_function.py`, ...) walks through a lens model's log likelihood one NumPy step
at a time, so you can see exactly what a model-fit computes. A real fit does not run that step-by-step code — it
runs the same calculation compiled by JAX. This section shows how to reach it.

**One-time setup.** A `ModelInstance` is not a JAX type, so it cannot cross a `@jax.jit` boundary until the
model's classes are registered as pytrees:

```python
import jax
import jax.numpy as jnp
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(model)
```

Skip this and the first call raises `TypeError: Error interpreting argument ... as an abstract array. The
problematic value is of type ModelInstance`.

**Via the `Analysis` object.** The short path, and the one to reach for first. The analysis threads `xp` through
the whole calculation for you, so `@jax.jit` is the only JAX you write:

```python
analysis = al.AnalysisImaging(dataset=dataset)  # use_jax=True by default

@jax.jit
def log_likelihood(instance):
    return analysis.log_likelihood_function(instance=instance)
```

`instance` is a model instance — e.g. `model.instance_from_prior_medians()`, or
`model.instance_from_vector(vector=...)` — the same object a non-linear search hands the analysis on every
iteration.

**Assembling the fit yourself.** If you want the `Tracer` and `FitImaging` in your own hands — a custom forward
model, a modified fit, a likelihood the shipped `Analysis` classes do not cover — build them inside the jitted
function. Here you must pass `xp=jnp` explicitly, because nothing is threading it for you:

```python
@jax.jit
def log_likelihood(instance):
    tracer = al.Tracer(galaxies=instance.galaxies)
    return al.FitImaging(dataset=dataset, tracer=tracer, xp=jnp).log_likelihood
```

Omit `xp=jnp` and the fit falls back to NumPy internals, raising `TracerArrayConversionError` the moment JAX
traces it. You do *not* need `register_tracer_classes` here — the tracer is built inside the jitted function
rather than passed across its boundary. That call is for the `__JIT-ing Library Methods__` case below, where a
`Tracer` is an argument.

For interferometer data the same shape applies with `al.FitInterferometer`. Both `TransformerDFT` and the
nufftax-backed `TransformerNUFFT` are JAX-traceable, so either works; only the legacy pynufft-backed
`TransformerNUFFTPyNUFFT` is not. Note the defaults differ by class: `Interferometer` (what a fit uses) defaults
to `TransformerNUFFT`, while `SimulatorInterferometer` defaults to `TransformerDFT`.

**Via `Fitness` — the production path.** A non-linear search does not call your function; it calls a `Fitness`
object, which maps a raw parameter vector to a model instance, calls the analysis, and returns the figure of
merit. `Fitness` performs the pytree registration itself, so it needs none of the setup above:

```python
from autofit.non_linear.fitness import Fitness   # not exported at autofit's top level

fitness = Fitness(
    model=model,
    analysis=al.AnalysisImaging(dataset=dataset),
    fom_is_log_likelihood=True,
)

log_likelihood = fitness._vmap(jnp.array([parameters]))[0]
```

`parameters` is a flat list of physical parameter values in the model's order (e.g.
`model.physical_values_from_prior_medians`), and `_vmap` evaluates a *batch* of them — hence the extra dimension
and the `[0]`.

This is the pattern to use when checking a JAX likelihood against the NumPy value a `likelihood_function.py`
walkthrough computes. Prefer it over a single `jax.jit(fn)(concrete)` call: `_vmap` is `jax.vmap(jax.jit(call))`,
exactly what a search runs, and vmapping over a batch forces every operation through JAX tracing. A single
concrete call can quietly succeed on code with NumPy leaking through an un-threaded `xp`, which then breaks as
soon as a real search batches parameter vectors.

__JIT-ing Library Methods__

The advanced path is JIT-ing library methods directly (`tracer.image_2d_from`,
`LensCalc.magnification_2d_via_hessian_from`, etc.) without going through a `Simulator` or `Analysis`.

The `lens_calc.py` workspace guide (`scripts/guides/lens_calc.py`) covers this "JIT-it-yourself" pattern, most
importantly the pairing rule that a library method called inside `@jax.jit` must be passed `xp=jnp`. Before your
first `@jax.jit` around a method that takes a `Tracer` as an argument, make the one-time call
`autolens.jax.register_tracer_classes(tracer)` so JAX can trace the tracer as a pytree.

__Return-Type Contract__

When `use_jax=True`, the data structures you get back (`Imaging`, `FitImaging`, `Tracer.image_2d_from(...)`
results, ...) carry `jax.Array` data inside instead of `numpy.ndarray`. For nearly everything you'd do in a
workspace — plotting, saving to `.fits`, comparing fit residuals — this is transparent: the plotters and FITS
writers call `numpy.asarray()` internally and you see the same images and numbers you would on the NumPy path.

What changes:

- Arithmetic on JAX arrays stays on the JAX path. Direct calls into NumPy (`np.sqrt(fit.residual_map.array)`)
  will host-transfer the array off the GPU; not wrong, but slower than `jnp.sqrt(...)` if you're inside a hot
  loop. For one-off analysis code, don't worry about it.
- The `.array` property of `aa.Array2D` etc. is the raw backing array — a `numpy.ndarray` on the NumPy path, a
  `jax.Array` on the JAX path.

The `data_structures.py` guide (`scripts/guides/data_structures.py`) covers the wrapper-vs-raw-array distinction
in detail.
"""
