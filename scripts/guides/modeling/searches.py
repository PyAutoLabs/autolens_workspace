"""
Modeling: Searches
==================

This script gives a run through of all non-linear searches that are available for modeling.

Two searches account for essentially all lens modeling you will do, and they are the first two documented below:

- **`Nautilus`** is the nested sampling algorithm used by every `modeling.py` example. It returns the full
  posterior -- the errors on every parameter and the covariances between them -- and is therefore the search you
  use when you need results you can quote. Extensive testing of lens modeling has shown it is the most accurate
  and efficient search available. For users familiar with statistical inference this may be surprising, as nested
  samplers are traditionally slower than MCMC methods such as Emcee and maximum likelihood methods such as LBFGS.
  A description of why Nautilus performs better than these other searches is beyond the scope of this script, but
  if you add me on SLACk I'd be happy to have a discussion about it!

- **`MultiStartProdigy`** is the JAX multi-start gradient optimizer used by every `start_here.py` example. It is
  far faster than Nautilus, but returns a single best-fit lens model with no errors at all, so it is the search
  you use to check quickly that your model and data are sensible.

Every other search documented below is an alternative you would reach for only if you really know what you are
doing, or want to cross-check a result against a different method.

Three different categories of searches are available, nested samplers (E.g. Nautilus, Dynesty), MCMC (E.g. Emcee) and
maximum likelihood (e.g. LBFGS). MCMC and MLE methods can often optionally use a "starting point" to initialize the
model-fit with the parameters where it should begin. Nested samplers do not use a starting point, but a similar
approach can be applied by putting tight priors on certain parameters.

To perform a model-fit, a fully modeling script will include steps which compose a model, create an `Analysis`
object and pass these to the search to perform the fit. We skip these steps for brevity.

__Contents__

- **Nautilus:** Nautilus (https://nautilus-sampler.readthedocs.io/en/latest/) is the recommended nested sampling algorithm, which returns the full posterior. It is gradient-free, so it does not use JAX gradients, but it does exploit JAX GPU acceleration via batched likelihood evaluation.
- **MultiStartProdigy:** The recommended JAX multi-start gradient maximum a posteriori (MAP) optimizer (learning-rate free), with `MultiStartAdam` and `MultiStartADABelief` as alternatives. Works for parametric sources (e.g. MGE, Sersic) and, with `resurrect=True` and an appropriate regularization scheme, pixelized sources.
- **Dynesty:** Dynesty (https://github.com/joshspeagle/dynesty) is a nested sampling algorithm.
- **Emcee:** Emcee (https://github.com/dfm/emcee) is an ensemble MCMC sampler that is commonly used in Astronomy.
- **Zeus:** Zeus (https://zeus-mcmc.readthedocs.io/en/latest/) is an ensemble MCMC slice sampler.
- **LBFGS:** LBFGS is a quasi-Newton optimization algorithm from scipy.
- **Start Point:** For maximum likelihood estimator (MLE) and Markov Chain Monte Carlo (MCMC) non-linear searches.
- **Search Cookbook:** There are a number of other searches supported by **PyAutoFit** and therefore which can be used.

__Start Here Notebook__

If any code in this script is unclear, refer to the `imaging/start_here.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al

"""
__Nautilus__

Nautilus (https://nautilus-sampler.readthedocs.io/en/latest/) is a nested sampling algorithm, and is the
recommended search for lens modeling. It is the search used by every `modeling.py` example in the workspace.

__Full Posterior__

Nautilus does not just return a best-fit lens model, it maps out the **full posterior**: the probability density
of every parameter, the errors on each one, and the covariances between them. If a fit infers an Einstein radius
of 1.6", Nautilus tells you whether that is 1.6 +/- 0.01 or 1.6 +/- 0.5, and whether it trades off against the
other mass parameters.

This is what most science needs, and it is why Nautilus remains the default recommendation even though the
gradient optimizer described next is faster. An optimizer hands you a single point in parameter space; Nautilus
hands you a measurement you can quote.

__JAX__

Nautilus is a **gradient-free** search. It explores parameter space by drawing new live points from a learned
boundary around the current live point set, using only evaluations of the likelihood. It therefore never
differentiates the likelihood, and unlike `MultiStartProdigy` below it neither needs nor uses JAX's gradients.

It does, however, exploit JAX on a GPU. Nautilus proposes points in batches rather than one at a time, and when
the analysis is JAX-traceable (created via `use_jax=True`) **PyAutoFit** evaluates each batch through the
`jax.vmap(jax.jit(...))` wrapped likelihood in a single call. All `n_batch` lens models are therefore fitted
simultaneously on the GPU, and the JAX speed-up applies in full to a Nautilus fit -- it simply comes from batched
likelihood evaluation instead of gradient descent.

`n_batch` is also the main control on GPU memory: a larger batch fits more lens models per call, but holds more
of them in VRAM at once.

__Live Points__

`n_live` is the main setting trading off accuracy against run-time. More live points give a more accurate result
but a longer run-time, whereas fewer are faster but risk inferring a local maxima. More complex models (that is,
models with more parameters) require more live points.
"""
search = af.Nautilus(
    path_prefix=Path("imaging", "searches"),
    name="Nautilus",
    unique_tag="example",
    # search specific settings
    n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, and this bounds VRAM use.
    iterations_per_quick_update=10000,
)

"""
__MultiStartProdigy (JAX multi-start gradient optimizers)__

`MultiStartProdigy` is the recommended JAX / `optax` multi-start gradient optimizer, a maximum a posteriori (MAP)
estimator. It directly addresses the weakness of a single-start optimizer like `LBFGS` (described below): instead
of descending from a single starting point (which, for the complex parameter spaces of lens models, frequently
gets stuck in a local maximum), it launches `n_starts` independent optimizations from broad starting points in
parallel via `jax.vmap` and returns the best one.

This "multi-start" approach was introduced for strong-lens modeling by GIGA-Lens (Gu, Huang et al. 2022,
arXiv:2202.07663; GIGA-Lens 2.0, arXiv:2606.30633): a wide population of parallel starts reliably finds the global
maximum-likelihood basin where single-start optimizers fail, which is what makes a gradient optimizer robust on the
multi-modal lens likelihood.

Prodigy is a *learning-rate free* update rule (Mishchenko & Defazio 2024, arXiv:2306.06101): it estimates its own
step size as it runs, so unlike `MultiStartAdam` it takes no `learning_rate` and there is nothing to tune. On
lens-modeling likelihoods it reaches the same maximum as a carefully hand-tuned `MultiStartAdam`, which is why it is
the recommended default of the family.

Three JAX multi-start optimizers are available (they share the same multi-start machinery, differing only in the
`optax` update rule each start uses):

- `MultiStartProdigy` — learning-rate free (recommended); no `learning_rate` to set.
- `MultiStartAdam` — the GIGA-Lens original; robust, but you must choose a `learning_rate`.
- `MultiStartADABelief` — an Adam variant; a drop-in alternative at the same `learning_rate`.

(`MultiStartLion` is a further sign-based alternative that prefers a ~10x smaller `learning_rate`.)

__Pixelized sources__

Because these are gradient-based, they require a JAX-traceable analysis (created via `use_jax=True`). They are
validated for **parametric sources** (e.g. an MGE or Sersic source) and — as of the 2026-07 pixelized-mesh
campaign — for **pixelized sources** (a `Pixelization` / reconstructed source) too, where from fully broad mass
priors they recover the true lens model on every mesh family (rectangular, k-nearest-neighbour, Delaunay), reaching
higher likelihoods than a comparably-configured `Nautilus` run in a fraction of the time.

Pixelized likelihoods do have regions where the fit (and hence the gradient) becomes non-finite — dominated by the
regularization parameters, not the mass model — so three settings matter:

- **`resurrect=True`**: starts whose trajectory goes non-finite are redrawn each step instead of dying in place.
  This is what converts the pixelized likelihood from unsearchable to searchable, and it is off by default (it is
  unnecessary on parametric sources).
- **The regularization scheme**: schemes whose high-coefficient region is numerically fragile (e.g. the adaptive
  split schemes, whose effective coefficient is raised to the 4th power) slow the search dramatically — it must
  rediscover the good regularization mode by luck. Two reliable choices: **fix or inherit the regularization** from
  an earlier fit (exactly what SLaM pipeline chaining does), which converges in a few hundred steps; or use a
  **kernel regularization** (e.g. `MaternKernel`), whose likelihood degrades smoothly at extreme coefficients and
  searches cleanly with the regularization left free.
- **`n_steps` and `batch_size`**: with a free regularization, budget 1500-3000 steps — the best regularization mode
  is often found late, and a long likelihood plateau is a regularization mode, not convergence. `batch_size`
  (e.g. 4) bounds the memory of the batched gradient, which pixelized models need.

Gradient-based searches on rectangular meshes must use the kernel-CDF `RectangularRTUAdaptDensity` /
`RectangularRTUAdaptImage` meshes (or set `over_sample_size_pixelization >= 4`): the default
`RectangularBilinearAdaptDensity` mesh's rank-CDF likelihood is exactly piecewise-constant in the mass model at
the default over-sampling, so its gradients are identically zero. For the RTU meshes one further setting
matters: a sharp `bandwidth` (e.g. 0.1) gives the mesh narrow gradient support that can stall descent, whereas
the default `bandwidth=1.0` searches on a smoother likelihood at only a small cost in final fit quality — if a
rectangular-mesh search stalls, search at the default bandwidth first and only sharpen afterwards.

Because it manages its own broad starting points, this search does not use the start-point API described below. Like
all optimizers it returns a single best-fit lens model, not a posterior with errors, so `Nautilus` above remains the
default recommendation when parameter uncertainties are required.
"""
search = af.MultiStartProdigy(
    path_prefix=Path("imaging", "searches"),
    name="MultiStartProdigy",
    n_starts=50,
    batch_size=None,  # Starts evaluated at once: `None` vmaps all 50 together, which is fastest but allocates the whole batched gradient; set an integer (e.g. 4) if you hit an out-of-memory error.
    n_steps=500,
)

"""
__Dynesty__

Dynesty (https://github.com/joshspeagle/dynesty) is a nested sampling algorithm.

Dynesty used to be the default model-fitting algorithm, before Nautilus was found to be better. However, Dynesty with
random walk nested sampling is still an effective method for modeling and worth using if you want to check your
results with an alternative to Nautilus.

Dynesty itself supports a wide variety of different nested sampling methods, including static 
sampling (`DynestyStatic` where the number of live point is fixed), dynamic sampling (`DynestyDynamic` where the number 
of live points varies with the fit) and different approaches to point sampling (e.g. slice sampling, uniform sampling). 

If you are familiar with nested sampling you can use all dynesty's different options by customizing the code below.
"""
search = af.DynestyStatic(
    path_prefix=Path("searches"),
    name="DynestyStatic",
    unique_tag="example",
    iterations_per_quick_update=2500,
    # search specific settings
    nlive=50,
    sample="rwalk",
    walks=10,
    bound="multi",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
)

search = af.DynestyDynamic(
    path_prefix=Path("searches"),
    name="DynestyDynamic",
    unique_tag="example",
    iterations_per_quick_update=2500,
    # search specific settings
    nlive=50,
    sample="rwalk",
    walks=10,
    bound="multi",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
)

"""
__Emcee__

Emcee (https://github.com/dfm/emcee) is an ensemble MCMC sampler that is commonly used in Astronomy and Astrophysics.

The wrapper with **PyAutoFit** supports different initialization methods, including a ball around the center of the
priors on the model parameters, which is the recommend initialization method for Emcee.

It also includes functionality which checks the auto correlations of the chains, and terminates the search early
if they meet certain convergence criteria. This is useful for ensuring that the chains have converged.

Whilst Emcee is a popular choice of MCMC method in astrophsyics, note that the MCMC method `Zeus`, described next, has
proven better as lens modeling for our tests.
"""
search = af.Emcee(
    path_prefix=Path("imaging", "searches"),
    name="Emcee",
    unique_tag="example",
    iterations_per_quick_update=5000,
    # search specific settings
    nwalkers=30,
    nsteps=500,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
)

"""
__Zeus__

Zeus (https://zeus-mcmc.readthedocs.io/en/latest/) is an ensemble MCMC slice sampler.

The wrapper with **PyAutoFit** supports different initialization methods, including a ball around the center of the
priors on the model parameters, which is the recommend initialization method for Emcee.

It also includes functionality which checks the auto correlations of the chains, and terminates the search early
if they meet certain convergence criteria. This is useful for ensuring that the chains have converged.

Zeus is the most effective MCMC method for lens modeling that we have tested, and is the recommended MCMC method,
however its performance is not as good as Nautilus.
"""
search = af.Zeus(
    path_prefix=Path("imaging", "searches"),
    name="Zeus",
    unique_tag="example",
    iterations_per_quick_update=5000,
    # search specific settings
    nwalkers=30,
    nsteps=20,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
    tune=False,
    tolerance=0.05,
    patience=5,
    maxsteps=10000,
    mu=1.0,
    maxiter=10000,
    vectorize=False,
    check_walkers=True,
    shuffle_ensemble=True,
    light_mode=False,
)

"""
__LBFGS__

LBFGS is a quasi-Newton optimization algorithm from scipy.

An optimizer only seeks to find the maximum likelihood lens model, unlike MCMC or nested sampling algorithms
like Zeus and Nautilus, which aim to map out parameter space and infer errors on the parameters. Therefore, in
principle, an optimizer like LBFGS should fit a lens model very fast.

In our experience, the parameter spaces fitted by lens models are often too complex for optimizers to be used without
careful initialization.
"""
search = af.LBFGS(
    path_prefix=Path("imaging", "searches"),
    name="LBFGS",
    unique_tag="example",
)

"""
__Start Point__

For maximum likelihood estimator (MLE) and Markov Chain Monte Carlo (MCMC) non-linear searches, parameter space
sampling is built around having a "location" in parameter space.

This could simply be the parameters of the current maximum likelihood model in an MLE fit, or the locations of many
walkers in parameter space (e.g. MCMC).

For many model-fitting problems, we may have an expectation of where correct solutions lie in parameter space and
therefore want our non-linear search to start near that location of parameter space. Alternatively, we may want to
sample a specific region of parameter space, to determine what solutions look like there.

The start-point API allows us to do this, by manually specifying the start-point of an MLE fit or the start-point of
the walkers in an MCMC fit. Because nested sampling draws from priors, it cannot use the start-point API.

Similar behaviour can be achieved by customizing the priors of a model-fit. We could place `GaussianPrior`'s
centred on the regions of parameter space we want to sample, or we could place tight `UniformPrior`'s on regions
of parameter space we believe the correct answer lies.

The downside of using priors is that our priors have a direct influence on the parameters we infer and the size
of the inferred parameter errors. By using priors to control the location of our model-fit, we therefore risk
inferring a non-representative model.

For users more familiar with statistical inference, adjusting ones priors in the way described above leads to
changes in the posterior, which therefore impacts the model inferred.
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

mass.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.5, upper_limit=0.5)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.5, upper_limit=0.5)
mass.einstein_radius = af.UniformPrior(lower_limit=0.2, upper_limit=3.0)

shear = af.Model(al.mp.ExternalShear)

shear.gamma_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

bulge = af.Model(al.lp_linear.SersicCore)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
We now define the start point of certain parameters in the model:

 - The galaxy is centred near (0.0, 0.0), so we set a start point for its mass distribution there.

 - The size of the lensed source galaxy is around 1.6" thus we set the `einstein_radius` to start here.

 - We know the source galaxy is a disk galaxy, thus we set its `sersic_index` to start around 1.0.

For all parameters where the start-point is not specified (in this case the `ell_comps`, their 
parameter values are drawn randomly from the prior when determining the initial locations of the parameters.
"""
initializer = af.InitializerParamBounds(
    {
        model.galaxies.lens.mass.centre_0: (-0.01, 0.01),
        model.galaxies.lens.mass.centre_1: (-0.01, 0.01),
        model.galaxies.lens.mass.einstein_radius: (1.58, 1.62),
        model.galaxies.source.bulge.sersic_index: (0.95, 1.05),
    }
)

"""
The `initializer` is passed to the search (e.g. the MCMC method Emcee below), which uses it to set the start-point of 
the walkers in parameter space. 
"""
search = af.Emcee(
    path_prefix=Path("imaging", "customize"),
    name="start_point",
    nwalkers=50,
    nsteps=500,
    initializer=initializer,
)

"""
__Search Cookbook__

There are a number of other searches supported by **PyAutoFit** and therefore which can be used, which are not 
explictly documented here. These include LBFGS.

The **PyAutoFit** search cookbook documents all searches that are available, including those not documented here,
and provides the code you can easily copy and paste to use these methods.

https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
