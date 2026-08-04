"""
Using JAX
=========

**PyAutoLens** runs on either **NumPy** (the default) or **JAX** (Google's array library with GPU support and
just-in-time compilation). JAX makes lens modeling 10-100x faster on large grids — sometimes more on GPU — so the
library is built to use it automatically wherever it helps.

The `start_here.py` introduction covers the one thing every user needs to know: install the JAX extra
(`pip install autolens[jax]` on Python 3.11+) and lens modeling uses JAX automatically. This guide covers the
technical detail behind that, and the situations where you interact with JAX directly.

Unlike most guides, the JAX recipes below are **executable code cells this script runs** — they are verified
against the installed library every time the workspace's automated tests run, so what you read is what works.

__Contents__

- **Auto-Enabled Modeling:** What "JAX is used automatically" means under the hood for model-fits.
- **Disabling JAX:** Forcing the NumPy path per-analysis or globally, and why that helps when debugging.
- **Writing @jax.jit Yourself:** Custom simulations and custom likelihood functions.
- **Custom Likelihood Functions:** Writing your own log likelihood — via the `Analysis` object, by hand, and via `Fitness`.
- **JIT-ing Library Methods:** The advanced path wrapping library methods like `tracer.image_2d_from` directly.
- **Pytree Registration:** What the library registers for you, and the one-time call for scripts that go it alone.
- **The Pairing Rule:** `@jax.jit` outside means `xp=jnp` inside — the one rule every recipe shares.
- **Grids Stay Outside the Boundary:** Why a `Grid2D` is closed over, never passed as a jitted argument.
- **Bound Methods and the JIT Cache:** Why `jax.jit(tracer.image_2d_from)` does not work, and the cache footgun.
- **Return-Type Contract:** What `jax.Array` data inside results means for plotting, saving and arithmetic.

__Auto-Enabled Modeling__

If JAX is installed, the `AnalysisImaging`, `AnalysisInterferometer`, and `AnalysisPoint` classes default to
`use_jax=True`. The non-linear search driver (Nautilus, dynesty, ...) batches parameter vectors and evaluates the
likelihood through `jax.vmap(jax.jit(...))` internally. You'll see a one-time log line like
`JAX: Applying vmap and jit to likelihood function -- may take a few seconds.` the first time a search starts;
that's the JIT compile kicking in, after which evaluations re-use the compiled trace.

If JAX is not installed, the analysis warns once and falls back to NumPy automatically.

This automatic path covers multi-plane systems too: the recursive multi-plane lens equation in
`tracer.traced_grid_2d_list_from(grid)` is pure numerical code with no data-dependent control flow, so it
compiles cleanly (this script verifies that below). For multi-plane point-source solving (forward-solving
multiple-image positions through several planes), use the higher-level `al.PointSolver(use_jax=True)` — see
`scripts/point_source/simulator.py` `__JAX Variant (Advanced)__`.

How much faster is JAX? It depends on grid size, model complexity and hardware (GPU gains dwarf CPU gains), so
this guide quotes no headline number — measured timings for representative configurations live in
`autolens_workspace_developer/jax_profiling/`.

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

   Wrapping that call in `@jax.jit` works for **imaging**, with one setup line you must write yourself:

   ```python
   import jax
   from autolens.jax import register_tracer_classes

   register_tracer_classes(tracer)   # one-time, before the first jitted call

   @jax.jit
   def simulate(tracer):
       return simulator.via_tracer_from(tracer=tracer, grid=grid)

   dataset_jax = simulate(tracer)    # .data.array is a jax.Array
   ```

   Why that registration call is needed — and why the library cannot make it for you — is the
   `__Pytree Registration__` section below.

   `scripts/imaging/simulator.py` runs this exact block, and is in `smoke_tests.txt` — so CI executes the recipe
   rather than this guide merely asserting it works.

   **Interferometer is the exception.** The jitted simulator path does not yet work there: `TransformerDFT` fails
   at the jit boundary because `Interferometer` is not a registered pytree, and `TransformerNUFFT` fails inside
   its own transform. Tracked in PyAutoLabs/PyAutoArray. Use the eager call for interferometer simulations.

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

Audience: users building custom forward models or scientific tools using PyAutoLens primitives directly — not
running standard fits (which go through `Analysis(use_jax=True)` and need zero JAX code from you) or standard
simulations (which use `Simulator(use_jax=True)` similarly).

Everything from here down is executable. First the imports — `jax_wrapper` must come before the other imports,
because it configures the JAX environment:
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import jax
import jax.numpy as jnp
import numpy as np

import autolens as al

"""
The recipes below need only a grid and a tracer — no dataset files. Any lens system works; this one is a simple
isothermal lens with a Sersic source.
"""

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.2, ell_comps=(0.05, 0.05)
    ),
)
source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5, sersic_index=2.0
    ),
)
tracer = al.Tracer(galaxies=[lens, source])

"""
__Pytree Registration__

When a jitted function takes a `Tracer`, `Galaxy` or `Galaxies` as an *argument*, JAX must flatten and unflatten
that object across the JIT boundary — i.e. the class must be registered as a JAX **pytree**. The library handles
this for you automatically in two situations:

1. You constructed an `Analysis` with `use_jax=True` (the default for modeling fits). Its first `fit_from` call
   walks the dataset and registers every reachable `Galaxy` / profile class.
2. You constructed a `Simulator` with `use_jax=True` and made a call — the same walk happens.

After either, every `Galaxy`, `LightProfile`, `MassProfile`, `Point`, etc. of the same class is JIT-safe for the
rest of the process. You never register individual classes yourself.

For a script like this one — no `Analysis` or `Simulator` in sight — the library provides a dedicated walker.
Call it once on a representative tracer, before the first jitted call:
"""

from autolens.jax import register_tracer_classes

register_tracer_classes(tracer)

"""
The library cannot make this call for you, and that is not an oversight: JAX flattens a jitted function's
arguments at trace time, *before* entering the callee, so a method that registered internally would already be
too late. `PointSolver.solve_triangles` documents the same constraint at its own call site.

__The Pairing Rule__

The single rule every recipe shares: when you decorate a function with `@jax.jit` that calls a PyAutoLens library
method internally, **pass `xp=jnp` to that method inside the function body**. The `xp=jnp` tells the library
"you're inside a JAX trace — route array operations through `jax.numpy`".

Here is the pattern for JIT-ing `tracer.image_2d_from`. Note two things: the `tracer` is a traced argument
(which is why we registered its classes above), and the `grid` is **closed over** from the enclosing scope — it
is deliberately *not* an argument (the `__Grids Stay Outside the Boundary__` section explains why). The `.array`
on the return unwraps the result to a raw `jax.Array`, because autoarray wrapper types cannot cross the JIT
boundary outward (see `__Return-Type Contract__`):
"""


@jax.jit
def image_fn(tracer):
    return tracer.image_2d_from(grid=grid, xp=jnp).array


image = image_fn(tracer)

print(
    f"jitted tracer image: type={type(image).__name__}, total flux={float(image.sum()):.4f}"
)

"""
Because `tracer` is a traced argument, a different tracer (same classes, different parameter values) re-uses the
compiled trace — this is what makes parameter sweeps and `jax.vmap` over tracers work naturally.

The same pattern JITs `LensCalc` methods. A `LensCalc` is constructed from the deflection-field callable of any
mass model, and its derived-quantity methods follow the pairing rule like everything else. One difference worth
seeing: under `xp=jnp`, `LensCalc`'s numeric methods return a **raw** `jax.Array` (no wrapper), so there is no
`.array` to unwrap — the return passes straight out of the jit:
"""


@jax.jit
def magnification_fn(tracer):
    lens_calc = al.LensCalc(deflections_yx_2d_from=tracer.deflections_yx_2d_from)
    return lens_calc.magnification_2d_via_hessian_from(grid=grid, xp=jnp)


magnification = magnification_fn(tracer)

print(
    f"jitted magnification: type={type(magnification).__name__}, max={float(magnification.max()):.2f}"
)

"""
The multi-plane lens equation compiles just as cleanly — `traced_grid_2d_list_from` is pure numerical code, so a
three-plane trace works with the identical recipe (each returned plane grid unwraps via `.array`):
"""

lens_2 = al.Galaxy(
    redshift=1.0, mass=al.mp.Isothermal(centre=(0.3, 0.3), einstein_radius=0.4)
)
source_2 = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.Sersic(centre=(0.0, 0.0), intensity=1.0, effective_radius=0.5),
)

tracer_multi = al.Tracer(galaxies=[lens, lens_2, source_2])


@jax.jit
def traced_grids_fn(tracer):
    return [
        traced_grid.array
        for traced_grid in tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    ]


traced_grids = traced_grids_fn(tracer_multi)

print(
    f"multi-plane trace: {len(traced_grids)} planes, leaf type={type(traced_grids[0]).__name__}"
)

"""
__Forgetting xp=jnp__

The default for `xp` in library method signatures is `np`, so forgetting the pairing rule is easy. What happens
depends on where you forget it:

- **Inside `@jax.jit` on an ordinary grid:** the library's NumPy internals call `numpy` functions on JAX tracer
  objects, raising `TracerArrayConversionError` the moment JAX traces the function. Loud and immediate.
- **Chaining library calls:** grids *produced by* an `xp=jnp` call are marked as JAX-backed. Pass one to a second
  library call without `xp=jnp` and the library raises immediately:

  ```
  ValueError: Called Sersic.image_2d_from with xp=np but the input grid is JAX-backed (grid.use_jax=True).
  Inside @jax.jit, pass xp=jnp explicitly to the library call.
  ```

If you see either error: add `xp=jnp` to the call site. Done.

__Grids Stay Outside the Boundary__

In every recipe above the grid is closed over, never passed as a jitted argument. This is a hard rule, not a
style choice: `Grid2D` is **not a registered pytree** (`register_tracer_classes` registers `Tracer`, `Galaxy` and
profile classes — not grids). Pass a grid as an argument and JAX flattens it into an opaque leaf; the library
then fails inside the trace with errors like `AttributeError: DynamicJaxprTracer has no attribute array`:

```python
@jax.jit
def image_fn(tracer, grid):                       # WRONG — grid as a traced argument
    return tracer.image_2d_from(grid=grid, xp=jnp).array

image_fn(tracer, grid)                            # AttributeError inside the trace
```

Passing the raw coordinate array instead (`image_fn(tracer, grid.array)`) does not work either — library methods
need the full `Grid2D` (its mask and over-sampling), not a bare array of coordinates.

Closing over the grid costs you nothing in practice: in a parameter sweep or a fit, the grid is fixed while the
model varies — exactly the split between closure constants (grid) and traced arguments (tracer) the recipes use.
If you need several grid resolutions, define a jitted function per grid (e.g. via a small factory function); each
grid would get its own compiled trace anyway, since the grid's shape is baked into the trace at compile time.

__Bound Methods and the JIT Cache__

You may be tempted by the shorter bound-method form, JIT-ing the method itself so the tracer rides along as
`self`:

```python
jitted = jax.jit(tracer.image_2d_from)
jitted(grid=grid, xp=jnp)                         # WRONG — TypeError
```

This does not work, for the same reason as above: now the *grid* is the traced argument, and a `Grid2D` cannot
cross the boundary. With grid-taking library methods, the closed-over-grid function form is the only correct
shape — which is why this guide teaches it exclusively.

The bound-method form also carries a cache footgun worth knowing about wherever you do use it (e.g. on methods
taking only raw arrays): attribute access creates a fresh bound-method object every time, so
`jax.jit(obj.method)` inside a loop silently misses the JIT cache and recompiles every iteration. Assign the
jitted callable once, outside the loop.

Three rules summarize everything above:

1. **Register first** when a `Tracer`/`Galaxy` is a jitted argument — one `register_tracer_classes(tracer)` call
   (or let an `Analysis`/`Simulator` with `use_jax=True` do it as a side effect).
2. **`@jax.jit` outside means `xp=jnp` inside** — on every library call in the function body.
3. **Grid closed over, model traced** — grids never cross the JIT boundary; tracers and galaxies do.

__Return-Type Contract__

When `use_jax=True`, the data structures you get back (`Imaging`, `FitImaging`, `Tracer.image_2d_from(...)`
results, ...) carry `jax.Array` data inside instead of `numpy.ndarray`. For nearly everything you'd do in a
workspace — plotting, saving to `.fits`, comparing fit residuals — this is transparent: the plotters and FITS
writers call `numpy.asarray()` internally and you see the same images and numbers you would on the NumPy path.

The wrapper types themselves (`al.Array2D`, `al.Grid2D`, ...) are **backend-polymorphic**: the same class holds a
`numpy.ndarray` on the NumPy path or a `jax.Array` on the JAX path, and the `.array` property is the raw backing
array in both cases. Calling a library method eagerly (outside any jit) with `xp=jnp` shows this — the result is
a normal `Array2D` whose backing is a `jax.Array`:
"""

image_eager = tracer.image_2d_from(grid=grid, xp=jnp)

print(
    f"eager xp=jnp call: wrapper={type(image_eager).__name__}, "
    f"backing (.array)={type(image_eager.array).__name__}"
)

"""
Three situations switch the backing from `numpy.ndarray` to `jax.Array`:

1. A fit or analysis with `use_jax=True` produced the structure.
2. A simulator with `use_jax=True` produced it.
3. You called a library method with `xp=jnp` yourself (as above).

**Inside `@jax.jit`, the wrapper cannot cross the boundary outward.** Returning an `al.Array2D` (or
`Grid2DIrregular`, or any autoarray wrapper) from a jitted function is a hard error, always:

```python
@jax.jit
def image_fn_wrong(tracer):
    return tracer.image_2d_from(grid=grid, xp=jnp)    # returns the wrapper — TypeError

# TypeError: ... traced for jit returned a value of type <class '...Array2D'>, which is not a valid JAX type
```

The contract is: **unwrap with `.array` before returning from the jit, rewrap on the host if you want the
structure back.** The rewrapped object keeps the JAX backing:
"""

image_rewrapped = al.Array2D(values=image, mask=grid.mask)

print(
    f"host rewrap: wrapper={type(image_rewrapped).__name__}, "
    f"backing (.array)={type(image_rewrapped.array).__name__}"
)

"""
With the structure rewrapped, everything downstream (plotters, FITS output, `.native` reshaping) works exactly as
on the NumPy path.

What else changes with `jax.Array` data:

- Arithmetic on JAX arrays stays on the JAX path. Direct calls into NumPy (`np.sqrt(fit.residual_map.array)`)
  will host-transfer the array off the GPU; not wrong, but slower than `jnp.sqrt(...)` if you're inside a hot
  loop. For one-off analysis code, don't worry about it.
- JAX arrays are immutable — in-place assignment (`arr[0] = ...`) raises; use `arr.at[0].set(...)` if you must.

The backing type, by construction route:

| Construction | `.array` backing |
|---|---|
| `al.Grid2D.uniform(shape_native, pixel_scales)` | `numpy.ndarray` |
| `fit = analysis.fit_from(instance)` from `AnalysisImaging(use_jax=True)` | `jax.Array` |
| `dataset = simulator.via_tracer_from(...)` from `SimulatorImaging(use_jax=True)` | `jax.Array` |
| `tracer.image_2d_from(grid=grid, xp=jnp)` | `jax.Array` |

`.array` is the safe accessor for the raw backing in all cases; plotting and `.fits` writers handle the
conversion transparently.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

The `jax` token releases PYAUTO_DISABLE_JAX so the JAX recipes above actually
execute through JAX under the smoke profile (which disables JAX by default).

ENV: jax
"""
