"""
Modeling: Weak Lensing
======================

This script fits a lens mass model to a weak gravitational lensing shear catalogue using a non-linear search.
It is the weak-lensing analogue of `scripts/imaging/modeling.py`: where that script infers a lens model from a
2D image of a lensed source, this script infers one from the (gamma_2, gamma_1) shear measurements of background
source galaxies — a `WeakDataset` produced by `scripts/weak/simulator.py`.

The previous script in the weak-lensing series, `scripts/weak/fit.py`, fitted a hand-picked mass model via the
`FitWeak` class and inspected its residuals and log-likelihood. Here we complete the workflow: the same `FitWeak`
machinery is wrapped in an `AnalysisWeak` object whose `log_likelihood_function` is called by the nested sampling
algorithm Nautilus to infer the posterior probability distribution of the lens mass parameters.

Weak-lensing model-fits are computationally much cheaper than imaging fits — there is no PSF convolution, no
masking, no pixelized source inversion — so this fit runs in minutes on an ordinary CPU.

__Scientific Context__

The weak-lensing regime modeled here is the shear signal around a galaxy-scale or group/cluster-scale lens:
background galaxies far from the lens centre are weakly sheared tangentially around it, and their measured
ellipticities constrain the mass distribution at radii the strong-lensing features (arcs, multiple images) do
not reach. Combining this large-radius information with strong lensing is a well-established technique for
galaxy clusters and groups, and a dedicated combined strong-plus-weak example follows later in the series.

__Contents__

- **Dataset:** Load the simulated `WeakDataset` (simulating it first if required).
- **Model:** Compose the lens mass model using the Model and Collection API.
- **Search:** Configure the Nautilus non-linear search.
- **Analysis:** Create the `AnalysisWeak` object defining how the model is fitted to the data.
- **Run Times:** The expected run time of a weak-lensing model-fit.
- **Model-Fit:** Perform the model-fit.
- **Output Folder Layout:** The structure of the `output` folder where results are written.
- **Result:** Inspect the inferred model and posterior.

__Model__

This script fits a `WeakDataset` of a 'cluster-scale' lens with a model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].

 - The background source galaxies are treated purely as shear probes — they have no light or mass model
   [0 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We load the simulated `WeakDataset` produced by `scripts/weak/simulator.py`: 1500 background source-galaxy
positions in a 50"-200" annulus around a cluster-scale lens, each with a measured `(gamma_2, gamma_1)` shear
vector and per-galaxy shape noise of 0.25.

If the dataset does not already exist on your system, it is created by running the simulator script, so this
example can be run without manually simulating data first.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "weak" / dataset_name

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
Before fitting, we visualise the dataset with `aplt.subplot_weak_dataset`, a 2x2 mosaic showing the shear field
as headless quiver segments, the per-galaxy noise map, the shear magnitude `|gamma|` and the position angle
`phi`. The tangential alignment of the segments around the lens centre at `(0.0", 0.0")` is the signal the
model-fit will exploit.
"""
aplt.subplot_weak_dataset(
    dataset=dataset,
    output_path=dataset_path,
    output_format="png",
)

"""
__Model__

We compose the lens model using `Model` and `Collection` objects, imported from **PyAutoLens**'s parent
project **PyAutoFit** — the identical API used by every other modeling script in the workspace:

 - The lens galaxy's total mass distribution is an `Isothermal`, with free centre, elliptical components and
   Einstein radius [5 parameters].

 - The source galaxy carries no model components: in a weak-lensing fit the background galaxies are pure
   probes of the foreground shear field, so the source's appearance is irrelevant. It is included only so the
   `Tracer` has a source-plane redshift for the lensing geometry.

Note what is absent compared to `scripts/imaging/modeling.py`: no light profiles, no `ExternalShear` (the
shear field *is* the data here — an external shear component would be degenerate with the signal at leading
order for this single-lens dataset) and therefore a much smaller parameter space (N=5 versus N=21+).

__Cluster-Scale Priors__

The default `Isothermal` priors are tuned for galaxy-scale lenses: the Einstein radius is a `UniformPrior`
over 0"-8" and the centre is a tight `GaussianPrior` of width 0.1". For this cluster the truth (25") lies
outside that Einstein-radius prior entirely, so we widen both to the cluster scale — a wide Einstein-radius
prior out to 60" and a 20"-wide centre prior about the field origin. Setting priors to the scale of the
system being fitted is standard practice; the defaults are simply a galaxy-scale starting point.
"""
# Lens:

mass = af.Model(al.mp.Isothermal)
mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=60.0)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=20.0)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=20.0)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

source = af.Model(al.Galaxy, redshift=1.0)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format, including the priors on each parameter.

(The `info_whitespace_length` parameter in `config/general.yaml`'s [output] section controls the whitespace
formatting if the display does not render well on your screen.)
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm
Nautilus (https://nautilus-sampler.readthedocs.io/en/latest/), the default search used throughout the
workspace.

Other data types fit their `start_here.py` with `af.MultiStartProdigy`, a much faster multi-start gradient
optimizer, and reserve `Nautilus` for the `modeling.py` example where the full posterior is needed. Weak-lensing
fits use `Nautilus` in both: that optimizer requires the JAX likelihood path, which these examples do not use,
and this parameter space is small enough that Nautilus is quick regardless.

With only N=5 parameters and a smooth, near-Gaussian likelihood surface, this is an easy parameter space —
100 live points is ample and keeps the run time to minutes.

An identical combination of model, search and dataset generates the same `unique_identifier`, meaning that
rerunning the script will resume the existing fit rather than starting again; the `unique_tag` below folds the
dataset name into that identifier.
"""
search = af.Nautilus(
    path_prefix=Path("weak"),  # The path where results and output are stored.
    name="modeling",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
    iterations_per_quick_update=5000,  # Every N iterations the max likelihood model is visualized and output to hard-disk.
)

"""
__Analysis__

We next create an `AnalysisWeak` object, whose `log_likelihood_function` is what the non-linear search calls
at every iteration:

 1. It builds a `Tracer` from the sampled mass parameters.
 2. It evaluates the tracer's shear field at the catalogue's galaxy positions — the same
    `LensCalc` Hessian primitive used by both the simulator and `FitWeak`.
 3. It returns the Gaussian log-likelihood of the observed shears given the model, summed over the
    `2 * n_galaxies` independent shear components.

A step-by-step walkthrough of this likelihood function is the next entry in the weak-lensing series
(`scripts/weak/likelihood_function.py`), following the format of the imaging and interferometer
likelihood-function guides.

Unlike `AnalysisImaging`, no `use_jax` option is passed: the weak-lensing fit is a NumPy calculation
(it is cheap enough that JAX acceleration is unnecessary for catalogues of this size).
"""
analysis = al.AnalysisWeak(dataset=dataset)

"""
__Run Times__

A single log-likelihood evaluation — one shear-field evaluation at 1500 galaxy positions plus a chi-squared
sum — takes of order a few milliseconds. Nautilus needs roughly 10,000–30,000 evaluations for this 5-parameter
model, so the full fit completes in a few minutes on a single CPU.

This is the key practical difference from imaging fits: the data volume of a shear catalogue is tiny (a few
hundred numbers rather than tens of thousands of pixels), which is also why weak-lensing constraints are so
cheap to add to a strong-lensing analysis.

__Model-Fit__

We begin the model-fit by passing the model and analysis objects to the search.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Output Folder Layout__

Results are written on the fly to the `autolens_workspace/output/weak/simple/modeling/<unique_hash>/` folder
in human-readable formats, so the fit can be inspected while it runs:

    output/weak/<dataset_name>/modeling/<unique_hash>/
        files/                         <- JSON + CSV: loadable Python objects
            dataset.json               <- the WeakDataset (reload via al.from_json)
            tracer.json                <- max log likelihood Tracer
            model.json                 <- fitted af.Collection model
            samples.csv                <- full Nautilus samples
            samples_summary.json       <- max log likelihood parameter values + errors
        image/                         <- PNG: visualization
            subplot_weak_dataset.png   <- the dataset mosaic (shear field, noise, |gamma|, phi)
            subplot_fit_weak.png       <- the fit mosaic (data, model, overlay, chi-squared map)
            galaxies.png               <- the mass model's convergence over the field extent
        model.info                     <- human-readable model summary
        model.results                  <- human-readable fit summary

__Result__

The search returns a result object; its `info` attribute shows the outcome in a readable format, including
the median and error estimates of every mass parameter.
"""
print(result.info)

"""
The `Result` object contains the maximum log likelihood instance, `Tracer` and `FitWeak`. Plotting the fit's
2x2 mosaic shows what a converged weak-lensing model looks like: short, randomly-oriented residual segments in
the data-vs-model overlay and a spatially uniform chi-squared map, consistent with the shape-noise floor.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_fit_weak(
    fit=result.max_log_likelihood_fit,
    output_path=dataset_path,
    output_format="png",
)

"""
Below, we make a corner plot of the "Probability Density Function" of every parameter in the model-fit.

For a shear-only fit, note how well the Einstein radius and mass-profile orientation are constrained relative
to the centre: the shear signal at each background galaxy is dominated by the enclosed mass and its
quadrupole, whereas the centre is only weakly pinned by the field's geometry. This complementarity — weak
lensing constrains the profile at large radius, strong lensing pins the centre and inner mass — is exactly
why the two are combined in cluster and group studies, and is the subject of the combined
strong-plus-weak example later in the series.
"""
aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

This script completed the core weak-lensing workflow: simulate (`simulator.py`), fit (`fit.py`) and now model
(`modeling.py`). The next entries in the series are:

 - `scripts/weak/likelihood_function.py`: a step-by-step guide to the weak-lensing likelihood.
 - Weak-lensing analyses of real shear catalogues and combined strong-plus-weak modeling.
"""
