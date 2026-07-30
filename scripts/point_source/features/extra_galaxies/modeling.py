"""
Modeling Features: Extra Galaxies
=================================

There may be extra galaxies projected near a lensed point source, for example a faint companion beside the main
lens galaxy of a lensed quasar, whose mass contributes to the ray-tracing and therefore to where the multiple
images appear.

Users coming from the imaging or interferometer versions of this example should be aware that the problem takes a
different shape here. For extended sources, extra galaxies present *two* problems: their light blends with the
lensed source emission, and their mass perturbs the ray-tracing. The imaging example therefore illustrates two
levers — noise-scale their emission out of the fit, or model their light and mass explicitly.

For point sources, only the second problem exists. A `PointDataset` is a list of image positions and fluxes, not
an image, so there is no extra-galaxy light in the data to contaminate anything and nothing to mask or
noise-scale. This example is therefore mass-only, and the question it answers is simply: does the model include
the extra galaxies' mass, or not?

That question matters more here than it does for imaging, because multiple image positions are extraordinarily
sensitive to perturbing mass. In the dataset fitted below, two companions with Einstein radii of 0.1" and 0.15"
move the four images by between 50 and 795 mas — one to two orders of magnitude above the 5 mas astrometric
precision of the data (the `simulator.py` script quantifies this). A model which omits them cannot fit the
positions, and will distort the main lens galaxy's mass distribution trying to absorb the difference.

__Contents__

- **Data Preparation:** Where the extra galaxy centres come from, which for point sources is not the point data.
- **Dataset:** Load and plot the point-source dataset and its accompanying imaging.
- **Point Solver:** Set up the `PointSolver` which determines the multiple images of the mass model.
- **Extra Galaxies Centres:** Load the centres which set up the extra galaxies in the model.
- **Model:** Compose the main lens and source components of the lens model.
- **Extra Galaxies Model:** Use the modeling API to add the extra galaxies to the model.
- **Information Budget:** Why a point-source model can only afford so many extra galaxy parameters.
- **Name Pairing:** The `PointDataset` name is paired with the `PointFlux` model component of the same name.
- **Search + Analysis:** Configure the non-linear search and the `AnalysisPoint` object.
- **Run Time:** Profiling the expected run time of the model-fit.
- **Model-Fit:** Perform the fit.
- **Result:** Overview of the results of the model-fit.
- **Approaches to Extra Galaxies:** The choices available for point-source data, which differ from imaging.
- **Wrap Up:** Summary of the script and where to go next up the regime ladder.

__Model__

This script fits a `PointDataset` of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 - The source galaxy is a `PointFlux` [3 parameters].
 - Each extra galaxy's total mass distribution is an `IsothermalSph` with a fixed centre
   [2 extra galaxies x 1 parameter = 2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

__Data Preparation__

The extra galaxies are set up in the model from a list of their centres, loaded from an
`extra_galaxies_centres.json` file which is already included in the dataset folder for this example.

For point sources this file cannot come from the point-source data. A `PointDataset` holds image positions and
fluxes and nothing else — it contains no information about where a faint companion galaxy sits on the sky. The
centres are instead measured from the imaging that accompanies the point-source observation, which is the same
imaging the multiple image positions were measured from in the first place.

The tutorial `autolens_workspace/*/imaging/data_preparation/examples/optional/extra_galaxies_centres.py`
describes how to mark these centres on an image and output them to a `.json` file. The simulator for this example
writes the accompanying imaging to `data.fits` in the dataset folder, and it is loaded and plotted below so you
can see the two companions the centres refer to.

Note that there is no `mask_extra_galaxies.fits` here, and no data-preparation step which produces one. Masking
is a tool for removing unwanted *emission* from an image, and there is no image being fitted.

__Start Here Notebook__

If any code in this script is unclear, refer to the `point_source/start_here.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens point-source dataset `extra_galaxies`, which is the dataset we will fit.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / "point_source" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/point_source/features/extra_galaxies/simulator.py"],
        check=True,
    )

"""
We load the dataset as a `PointDataset`, which contains the positions and fluxes of the four multiple images.
"""
dataset = al.from_json(
    file_path=dataset_path / "point_dataset.json",
)

print("Point Dataset Info:")
print(dataset.info)

aplt.subplot_point_dataset(dataset=dataset)

"""
We next load the imaging which accompanies this point-source dataset.

For the other point-source examples this imaging is optional, loaded purely so the multiple images can be seen in
context. Here it is doing real work: the two extra galaxies are visible in it, and it is where their centres were
measured. Plotting it is the quickest way to confirm the centres loaded below actually land on the companions.
"""
data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.05)

aplt.plot_array(array=data, title="Accompanying Imaging (Extra Galaxies Visible)")

"""
__Point Solver__

Set up the `PointSolver` which determines the multiple images of the mass model for a point source at a
given (y,x) position in the source plane, by ray tracing progressively smaller triangles from the image-plane to
the source-plane. A full description of the solver and its settings is given in `point_source/modeling.py`.

There are no special solver settings required for extra galaxies. The solver operates on whatever tracer the
analysis builds from the model, and the extra galaxies are simply additional galaxies in that tracer.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)

"""
__Extra Galaxies Centres__

To set up a model including each extra galaxy, we input the centres of the extra galaxies, which are used to fix
the centres of their mass profiles.

In principle a model including extra galaxies could be composed without these centres, by simply adding two
additional mass profiles with free centres. The modeling API supports this, but we do not use it here.

The reason is the same one given in the imaging version of this example — models where extra galaxies have free
centres are often too complex to fit, and an extra galaxy's mass profile may recentre itself and act as part of
the main lens galaxy's mass distribution — but it bites considerably harder for point sources. With only 12 data
points (see **Information Budget** below), four additional free centre parameters would leave the model close to
unconstrained, and the degeneracy between a free-centre perturber and the main lens's mass is not something a
handful of image positions can break.
"""
extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "extra_galaxies_centres.json")
)

print(extra_galaxies_centres)

"""
__Model__

Perform the normal steps to set up the main model of the lens galaxy and source.

The source is a `PointFlux` rather than a `Point`, because this dataset includes fluxes as well as positions. The
`features/fluxes.py` example describes flux fitting in detail.

The `ExternalShear` is not included in the mass model. As `point_source/modeling.py` explains, a quadruply imaged
point source does not carry enough information to constrain an `Isothermal` and an `ExternalShear` — and in this
example the extra galaxies are already spending part of that budget.

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

point_0 = af.Model(al.ps.PointFlux)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

"""
__Extra Galaxies Model__

We now use the modeling API to create the model for the extra galaxies.

Each extra galaxy is a `Galaxy` with an `IsothermalSph` mass profile whose `centre` is fixed to the centre loaded
above, leaving a single free parameter per galaxy: its `einstein_radius`. There is no light profile, because
there is no light in the data to fit.

Extra galaxy mass profiles can run away to unphysically high `einstein_radius` values, degrading the fit. The
`einstein_radius` is therefore given a `UniformPrior` with an upper limit of 0.5", comfortably above the true
values of 0.1" and 0.15" while excluding solutions where a companion takes over as a main deflector.

The `extra_galaxies` collection is passed to the overall model as a separate entry alongside `galaxies`. The
`AnalysisPoint` object appends these galaxies to the tracer it builds from each model instance, so they
contribute to the ray-tracing the `PointSolver` performs — no further wiring is required.
"""
# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    # Extra Galaxy

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), extra_galaxies=extra_galaxies
)

"""
The `info` attribute confirms the model includes the extra galaxies we defined above.
"""
print(model.info)

"""
__Information Budget__

It is worth being explicit about how little data a point-source fit has to work with, because it governs every
design choice above.

This dataset provides 12 numbers: 4 multiple images x 2 coordinates, plus 4 fluxes. The model has 10 free
parameters. Compare this to the imaging version of this example, where the same physical system is constrained by
tens of thousands of image pixels and a model with twice as many parameters is comfortable.

Three consequences follow, and they are why this example looks the way it does:

 - **Centres are fixed, not free.** Four extra free parameters would take the model to N=14 against 12 data
   points.

 - **Fluxes are fitted, not just positions.** Positions alone give 8 data points, which will not support a
   10-parameter model. This is why the dataset simulated for this example includes fluxes, whereas the
   `point_source/simulator.py` default dataset is positions-only.

 - **The `ExternalShear` is omitted.** In the imaging examples shear is a standard ingredient; here it competes
   directly with the extra galaxies for the same scarce constraints.

If your own point-source system has more information — a second lensed source at a different redshift, or
measured time delays — the budget grows and you can afford a richer extra-galaxies model. The `multiple_sources`
and `time_delays` examples in this folder show how to add those datasets.

__Name Pairing__

Every point-source dataset has a `name`, which in this example is `point_0`. This name pairs the dataset to the
`PointFlux` in the model above, which is also named `point_0`.

Name pairing applies only to the point sources being fitted. The extra galaxies are not point-source datasets and
have no names — they enter the model through the `extra_galaxies` collection, not through name pairing.

__Search + Analysis__

The code below performs the normal steps to set up a model-fit.

Given the two extra free parameters due to the extra galaxies, we increase the number of live points from the 100
used in `point_source/modeling.py` to 150.
"""
search = af.Nautilus(
    path_prefix=Path("point_source") / "features",
    name="extra_galaxies",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see `point_source/modeling.py`.
    iterations_per_quick_update=10000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeat,  # Image-plane chi-squared with repeat image pairs.
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
__Run Time__

Adding extra galaxies to a point-source model increases the likelihood evaluation time only slightly, because the
extra work is computing two spherical isothermal deflection fields — trivial compared to the iterative triangle
solve the `PointSolver` performs.

The real cost is the two extra free parameters, which increase the dimensionality of parameter space and so the
number of iterations Nautilus needs to converge. Expect a modest increase over the ~20 minute CPU / ~5 minute GPU
run times quoted for the simpler point-source fits.

Note that this is the opposite balance to the imaging version of this example, where each extra galaxy adds a
light profile whose image must be evaluated and blurred on every likelihood call.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a
non-linear search to find which models fit the data with the highest likelihood.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the result in a readable format, and confirms the `einstein_radius` of each extra
galaxy has been inferred alongside the main lens model.

Both extra galaxy Einstein radii are recovered to within ~10% of the input values (0.1" and 0.15") and at high
significance — twelve data points are enough, because each extra galaxy costs only one free parameter and the
image positions are so sensitive to it. The main lens galaxy's `einstein_radius`, centre and `ell_comps` are
recovered accurately at the same time, which is the real payoff: had the extra galaxies been omitted, those
parameters would have absorbed the perturbation and been biased.

(Your exact numbers will differ slightly from any quoted here, because `simulator.py` draws fresh noise on every
run.)
"""
print(result.info)

"""
We plot the tracer of the maximum log likelihood model, which includes the extra galaxies.
"""
aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grid)

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

__Approaches to Extra Galaxies__

The imaging version of this example lays out two extremes — noise-scale the extra galaxies' emission out of the
fit, or model their light and mass — with a spectrum of options in between.

For point sources that spectrum collapses, because there is no emission to remove. The choices are:

- **Model their mass**, as done above. Appropriate whenever the companions are close enough to the multiple
  images to shift them by more than the astrometric precision, which for the system fitted here they are by two
  orders of magnitude.

- **Omit them**, accepting that the main lens galaxy's mass model will absorb their effect. This is the right
  call when the companions are far from the images or so faint that their masses are negligible, and it is what
  every other example in the `point_source` package does. It is a decision to make deliberately, not a default:
  an omitted perturber does not produce a visibly bad fit the way an unmasked galaxy does in imaging, it
  produces a subtly biased mass model.

There is no equivalent of the imaging middle ground (mask the light, model the mass) because the two halves of
that choice do not exist separately here.

What you can still vary is how much freedom the extra galaxies are given: their redshifts could be made free
parameters, different mass profiles could be used for each, or their masses could be tied together. Extending the
API should be straightforward given the example above, and if anything is unclear then checkout the model
cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Scaling Relations__

The modeling API supports composing extra galaxies such that their masses follow a scaling relation with their
luminosities, so that a population of them adds a fixed, small number of free parameters rather than one per
galaxy. Given how tight the information budget is for point sources, this is often the only affordable way to
include more than a handful of them.

This is documented in the `autolens_workspace/*/point_source/features/scaling_relation` example, which ties each
companion's Einstein radius to the main lens's own so the whole population costs zero free parameters. Against the
12-point budget discussed above that is decisive: in that example, tying five companions gives 8 free parameters where
freeing them gives 13 — more parameters than data points.

__Wrap Up__

The extra galaxies API makes it straightforward to include the mass of nearby galaxies in a point-source lens
model, using a single free parameter per galaxy.

When should you use the extra galaxies API as shown here, and when should you move up to a different package?

 - A galaxy-scale point-source lens is one which can be modeled accurately with a single main deflector. Extra
   galaxies are perturbations on that model — they improve it, sometimes substantially, but the system is
   recognisably one lens galaxy plus companions.

 - Once several galaxies contribute comparably to the deflection, or the lens sits in a common halo, the
   `group` package is the right tool. Its `start_here` and `modeling` examples use this same extra galaxies API,
   but alongside multiple main lens galaxies and a scaling-relation tier.

 - At cluster scale, where many point sources at many redshifts are fitted through a factor graph with a host
   halo and a member population, use the `cluster` package. There the tiered galaxy API is not a feature but the
   foundation of the model.
"""
