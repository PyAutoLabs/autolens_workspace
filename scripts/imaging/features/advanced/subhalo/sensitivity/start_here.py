"""
Sensitivity Mapping: Start Here
===============================

**What could this lens have revealed, even when no subhalo was found?**

Sensitivity mapping measures which dark matter halos a particular dataset and modeling procedure could detect.
It injects halos into simulated observations, fits each simulation with and without a halo, and compares the
Bayesian evidence. This calibrates the interpretation of both detections and non-detections.

Use sensitivity mapping to:

- **Interpret a non-detection:** establish which halo masses and positions your analysis could have recovered.
- **Build population constraints:** combine sensitivity-calibrated lens samples with a halo-population model to
  investigate the abundance and mass distribution of small halos, and test dark matter models.
- **Compare observing strategies:** repeat the simulations with different noise levels or angular resolution to
  assess how detectability changes. This requires additional simulation scenarios, not run by this example.

**Scientific context:** [Vegetti & Koopmans (2009), Statistics of mass substructure from strong gravitational
lensing](https://arxiv.org/abs/0903.4752) explains how detection limits and sample size enter inference of the
substructure mass fraction and mass function.

**What this script demonstrates:** four injected halo positions at one fixed mass of `1e10` solar masses, with an
MGE source. Each simulated dataset is fitted twice. This is a small demonstration, not a converged sensitivity map
or a dark matter population analysis. For calibration using reconstructed source morphology, continue to
[slam_source_pixelized.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/slam_source_pixelized.ipynb).

![Conceptual illustration of sensitivity around lensed arcs](../../../../../../images/features/sensitivity_mapping_illustration.png)

*Conceptual illustration, not a computed sensitivity map. The overlay represents recoverability at one illustrative
injected mass, not the presence of detected halos. It does not depict the four-cell result of this script.*

__How Sensitivity Is Measured__

Bayesian model comparison weighs the support for each model by averaging its likelihood over its parameter priors.
A model with an extra halo may not be favored because the data cannot distinguish it, even when a halo is present.
Sensitivity mapping explores this by simulating known perturbations and measuring how strongly they are recovered.

We define a function that simulates a dataset from the main lens/source model plus an injected halo. For each trial,
we fit the same simulated dataset with a base model and with a model that includes the halo. This example uses
nested sampling and the difference in log Bayesian evidence as its recovery statistic. Other statistics can be
chosen, but the calibration must use the detection criterion intended for the scientific analysis.

A sensitivity map is conditional on the simulation, source and lens models, priors and noise assumptions. A positive
evidence difference alone is not a detection criterion: choose and justify a threshold. Repeated noise realizations
and tests of model assumptions may be needed to measure recovery probabilities and robustness.

__Contents__

- **Subhalo Detection Discussion:** Distinguish a detection search from its sensitivity calibration.
- **Subhalo Sensitivity Mapping:** Inject halos and compare paired fits to map recoverability.
- **SLaM Pipelines:** Connect this self-contained example to the fuller analysis pipelines.
- **Pixelized Source:** Simulate and fit complex source morphology consistently.
- **Base Model:** Fit the smooth lens and source used to initialize the simulations.
- **Perturb Model:** Define the injected halo and the extra fitted component.
- **Mapping Grid:** Inject one fixed halo mass at four image-plane positions.
- **Perturb Model Prior Func:** Set the recovery-fit priors separately from injected values.
- **Simulation Instance:** Use the fitted main lens/source instance in the simulations.
- **Simulate Function Class:** Create mock observations for each injected halo.
- **Base Fit:** Fit each simulated observation without a halo.
- **Perturb Fit:** Fit the same observation with an additional halo.
- **Results:** Inspect the evidence differences from the four paired fits.

__Subhalo Detection Discussion__

In subhalo detection, our strong lens modeling tests whether the data favor a dark matter subhalo at a given (y,x)
image-plane location of the strong lens. We determine this by comparing lens models with and without a subhalo. However,
we are only able to detect dark matter subhalos with (y,x) locations near the lensed source light, and when the
subhalo is massive enough to perturb its light in an observable way.

Subhalo detection analysis therefore does not tell us where we could detect subhalos and of what mass. To know this,
we must perform sensitivity mapping.

__Subhalo Sensitivity Mapping__

A detailed sensitivity study can require many simulated strong lens images; this example uses four. Each simulated image includes a
dark matter subhalo at a given (y,x) coordinate and at a given mass. We fit each simulated dataset twice,
with a lens model which does not include a subhalo and with a lens model that does.

A trial is classified as recovered when the evidence difference exceeds the chosen detection threshold.
The threshold and recovery-fit assumptions must be consistent with the analysis being calibrated.

For many simulated datasets, we will find the evidence does not increase when we include a subhalo in the model-fit,
showing weak sensitivity to that injected mass and position under the adopted assumptions. Sensitivity is often
strongest near resolved arcs, but depends on the lens geometry and data quality.

A full position-and-mass study samples a three-dimensional (or higher) grid of subhalo (y,x) and mass. Here we
calculate a coarse two-dimensional slice at one injected mass. Repeating this over mass and refining the spatial
grid builds a more detailed map; 2D slices visualize how recoverability varies across the image plane.

The information provided by a sensitivity map is ultimately required to turn dark matter subhalo detections into
constraints on the dark matter subhalo mass function, which is the primary goal of subhalo detection. Thus, it is
necessary to make statements about the nature of dark matter: cold, warm, fuzzy, or something else entirely?

__SLaM Pipelines__

The Source, (lens) Light and Mass (SLaM) pipelines are advanced lens modeling pipelines which automate the fitting
of complex lens models. The SLaM pipelines provide workflows for detailed DM subhalo analyses.

This example script does not use a SLaM pipeline, to keep the sensitivity mapping self contained. However, it is
anticipated that any user performing sensitivity mapping on real data will use the SLaM pipelines, which in the
`subhalo` package have dedicated extensions for performing sensitivity mapping to both imaging and interferometer data.

Therefore you should be familiar with the SLaM pipelines before performing DM subhalo sensitivity mapping on real
data. If you are unfamiliar with the SLaM pipelines, checkout the
example `autolens_workspace/notebooks/guides/modeling/slam_start_here`.

__Pixelized Source__

Detecting a DM subhalo requires the lens model to be sufficiently accurate that the residuals of the source's light
are at a level where the subhalo's perturbing lensing effects can be detected.

For complex resolved sources, a pixelized reconstruction can capture structure that simple light profiles miss.
The sensitivity calibration should reproduce the source treatment used by the detection analysis. This example
uses an MGE light-profile source to keep the demonstration manageable; the pixelized-source example below
shows the corresponding workflow for irregular reconstructed sources.

The example `subhalo/sensitivity/slam_source_pixelized.ipynb` extends the SLaM pipelines with pixelized sources
and therefore shows how to perform sensitivity mapping using pixelized sources.

Note that the simulation procedure for a pixelized source is different to the one shown here. In this example, the
light profile source parameters are used to simulate each sensitivity mapping dataset. When pixelized sources are
used, the source reconstruction on the mesh is used, such that the simulations capture the irregular morphologies
of real source galaxies.

__Related Examples__

- [slam_source_parametric.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/slam_source_parametric.ipynb) — Use the parametric-source pipeline.
- [slam_source_pixelized.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/slam_source_pixelized.ipynb) — Use the pixelized-source pipeline.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
import os
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "dark_matter_subhalo"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/imaging/features/advanced/subhalo/simulator.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.05,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model + Search + Analysis + Model-Fit (Base Search)__

We are performing sensitivity mapping to determine where a subhalo is detectable. This will require us to simulate 
many realizations of our dataset with a lens model, called the `simulation_instance`. To get this model, we therefore 
fit the data before performing sensitivity mapping so that we can set the `simulation_instance` as the maximum 
likelihood model.

We perform this fit using the lens model we will use to perform sensitivity mapping, which we call the `base_model`.
"""
bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

base_model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=bulge),
    ),
)

search_base = af.Nautilus(
    path_prefix=Path("imaging", "advanced", "subhalo", "sensitivity"),
    name="sensitivity_mapping_base",
    unique_tag=dataset_name,
    n_live=100,
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

result = search_base.fit(model=base_model, analysis=analysis)

"""
__Base Model__

We now define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is fitted to 
every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model which 
includes one!). 

For this model, we can use the `base_model` above, however we will use the result of fitting this model to the dataset
before sensitivity mapping. This ensures the priors associated with each parameter are initialized so as to speed up
each non-linear search performed during sensitivity mapping.
"""
base_model = result.model

"""
__Perturb Model__

We now define the `perturb_model`, which is the model component whose parameters we iterate over to perform 
sensitivity mapping. In this case, this model is a `NFWMCRLudlowSph` model and we will iterate over its
`centre` at fixed injected `mass_at_200` in this example. A larger study can vary both. We set it up as a `Model` so it has an associated redshift and can be directly
passed to the tracer in the simulate function below.

Many instances of the `perturb_model` are created and used to simulate the many strong lens datasets that we fit. 
However, it is only included in half of the model-fits; corresponding to the lens models which include a dark matter 
subhalo and whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model` to 
determine if the subhalo was detectable.

By fitting both models to every simulated lens, we therefore infer the Bayesian evidence of every model to every 
dataset. Sensitivity mapping therefore maps out for what values of `centre` and `mass_at_200` in the dark matter 
subhalo the model-fit including a subhalo provides an evidence increase over the simpler model-fit. Compare
these differences with the chosen detection threshold to classify recovery.
"""
perturb_model = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.NFWMCRLudlowSph)

"""
__Mapping Grid__

This demonstration fixes the injected `mass_at_200` to `1e10` solar masses and varies both centre coordinates.
Each coordinate spans -3.0 to +3.0 arcsec. With `number_of_steps=2` below, the two bin centres per coordinate
are -1.5 and +1.5 arcsec, giving four injected (y,x) positions:

 - (-1.5, -1.5)
 - (-1.5, +1.5)
 - (+1.5, -1.5)
 - (+1.5, +1.5)

Each position generates one simulated dataset and two model fits: eight recovery fits in total, in addition to
the initial main lens/source fit. This coarse grid illustrates the workflow; it does not determine a minimum
detectable halo mass. That requires additional injected masses, adequate spatial sampling and a detection criterion.

The injected mass and position are distinct from the parameters allowed to vary during recovery. The next section
uses `perturb_model_prior_func` to set those recovery priors.
"""
grid_dimension_arcsec = 3.0

perturb_model.mass.mass_at_200 = 1e10
perturb_model.mass.centre.centre_0 = af.UniformPrior(
    lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
)
perturb_model.mass.centre.centre_1 = af.UniformPrior(
    lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
)
perturb_model.mass.redshift_object = 0.5
perturb_model.mass.redshift_source = 1.0

"""
__Perturb Model Prior Func__

The default priors on the `perturb_model` are `UniformPrior`'s bounded around each sensitivity grid cell.

For example, a simulated dark matter subhalo can be at location (1.5, -1.5) and its priors are:

- y is `UniformPrior(lower_limit=0.0, upper_limit=3.0)`.
- x is `UniformPrior(lower_limit=-3.0, upper_limit=0.0)`.

The `mass_at_200` is fixed to `1e10` for injection. Without the callback below it would also remain fixed
in the recovery model; the callback instead frees it during fitting.

By passing a `perturb_model_prior_func` to the sensitivity mapper, we can manually overwrite the priors on 
the `perturb_model` which are used instead for the fit.

Below, we update the priors as follows:

- The y and x priors are trimmed to much narrower bounded priors, confined to regions 0.05" each side of the 
  true DM subhalo.

- The `mass_at_200` is made a free parameter with `LogUniformPrior(lower_limit=1e6, upper_limit=1e12)`. This is a large range,
  but allows the recovery fit to explore lower and higher halo masses.

These narrow position priors use the known injected location. This is a targeted recovery experiment, not a blind
search of the entire image. Priors affect the Bayesian evidence, so its thresholds cannot automatically be
transferred to a detection search with broader position priors.
"""


def perturb_model_prior_func(perturb_instance, perturb_model):
    b = 0.05

    perturb_model.mass.centre.centre_0 = af.UniformPrior(
        lower_limit=perturb_instance.mass.centre[0] - b,
        upper_limit=perturb_instance.mass.centre[0] + b,
    )

    perturb_model.mass.centre.centre_1 = af.UniformPrior(
        lower_limit=perturb_instance.mass.centre[1] - b,
        upper_limit=perturb_instance.mass.centre[1] + b,
    )

    perturb_model.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1e6, upper_limit=1e12
    )

    return perturb_model


"""
__Simulation Instance__

We are performing sensitivity mapping to determine where a subhalo is detectable. This will require us to 
simulate many realizations of our dataset with a lens model, called the `simulation_instance`. This model uses the
result of the fit above.

The code below ensures that the lens light, mass and source parameters of the strong lens are used when simulating
each dataset with a dark matter subhalo.
"""
simulation_instance = result.instance

"""
__Simulate Function Class__

We now write the `simulate_cls`, which takes the `simulation_instance` of our model (defined above) and uses it to 
simulate a strong lens dataset, which include a dark matter subhalo, which is subsequently fitted.

Additional attributes required to simulate the data (mask, PSF) can be passed to the `__init__` method, and the 
simulation is  performed in the `__call__` method.

When this dataset is simulated, the quantity `instance.perturb` is used in `__call__`. This is an instance 
of the `NFWMCRLudlowSph`, and it is different every time the `simulate_cls` is called based on the value of sensitivity 
being computed. 

In this example, this `instance.perturb` corresponds to two different subhalos with values of `mass_at_200` of 
1e6 MSun and 1e13 MSun.
"""


class SimulateImaging:
    def __init__(self, mask, psf):
        """
        Class used to simulate the strong lens imaging used for sensitivity mapping.

        Parameters
        ----------
        mask
            The mask applied to the real image data, which is applied to every simulated imaging.
        psf
           The PSF of the real image data, which is applied to every simulated imaging and used for each fit.
        """
        self.mask = mask
        self.psf = psf

    def __call__(self, instance: af.ModelInstance, simulate_path: str):
        """
        The `simulate_function` called by the `Sensitivity` class which simulates each strong lens image fitted
        by the sensitivity mapper.

        The simulation procedure is as follows:

        1) Use the input galaxies of the sensitivity `instance` to set up a tracer, which generates the image-plane
           image of the strong lens system.

        2) Simulate this image using the input dataset noise (Poisson) and PSF.

        3) Apply the mask used in the analysis of the real image to the simulated image.

        4) Output information about the simulation to hard-disk.

        The `subhalo` in the sensitivity `instance` changes for every iteration of the sensitivity mapping, ensuring
        that we map out the sensitivity of the analysis to the subhalo properties (centre, mass, etc.).

        Parameters
        ----------
        instance
            The sensitivity instance, which includes the galaxies whose parameters are varied to perform sensitivity.
            The subhalo in this instance changes for every iteration of the sensitivity mapping.
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.

        Returns
        -------
        A simulated image of a strong lens, which id input into the fits of the sensitivity mapper.
        """

        """
        __Resume Fit__

        If sensitivity mapping already began on this grid cell, the dataset will have been simulated already and we
        do not want to resimulate it and change its noise properties. 

        We therefore load it from the `simulate_path` instead.
        """
        try:
            dataset = al.Imaging.from_fits(
                data_path=f"{simulate_path}/data.fits",
                psf_path=f"{simulate_path}/psf.fits",
                noise_map_path=f"{simulate_path}/noise_map.fits",
                pixel_scales=self.mask.pixel_scales,
                check_noise_map=False,
            )

            dataset = dataset.apply_mask(mask=self.mask)

            tracer = al.Tracer(
                galaxies=[
                    instance.galaxies.lens,
                    instance.perturb,
                    instance.galaxies.source,
                ]
            )

            traced_grid = tracer.traced_grid_2d_list_from(
                grid=dataset.grid,
            )[-1]

            source_centre = instance.galaxies.source.bulge.centre

            over_sample_size = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=traced_grid,
                    sub_size_list=[16, 8, 2],
                    radial_list=[0.1, 0.3],
                    centre_list=[source_centre],
                )
            )

            over_sample_size_lens = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=dataset.grid,
                    sub_size_list=[4, 2, 2],
                    radial_list=[0.3, 0.6],
                    centre_list=[(0.0, 0.0)],
                )
            )

            over_sample_size = np.where(
                over_sample_size > over_sample_size_lens,
                over_sample_size,
                over_sample_size_lens,
            )
            over_sample_size = al.Array2D(values=over_sample_size, mask=self.mask)

            return dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        except FileNotFoundError:
            pass

        """
        __Ray Tracing__

        Set up the `Tracer` which is used to simulate the strong lens imaging, which may include the subhalo in
        addition to the lens and source galaxy.
        """
        tracer = al.Tracer(
            galaxies=[
                instance.galaxies.lens,
                instance.perturb,
                instance.galaxies.source,
            ]
        )

        """
        __Simulate__

        Set up the grid, PSF and simulator settings used to simulate imaging of the strong lens. These should be 
        tuned to match the S/N and noise properties of the observed data you are performing sensitivity mapping on.
        """
        grid = al.Grid2D.uniform(
            shape_native=self.mask.shape_native,
            pixel_scales=self.mask.pixel_scales,
        )

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=grid,
            sub_size_list=[32, 8, 2],
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0)],
        )

        grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

        simulator = al.SimulatorImaging(
            exposure_time=1000.0,
            psf=self.psf,
            background_sky_level=0.1,
            add_poisson_noise_to_data=True,
            noise_seed=1,
        )

        dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

        """
        __Masking__

        The data generated by the simulate function is what is ultimately fitted.

        Therefore, we also apply the mask for the analysis before we return the simulated data.
        """
        dataset = dataset.apply_mask(mask=self.mask)

        traced_grid = tracer.traced_grid_2d_list_from(
            grid=dataset.grid,
        )[-1]

        source_centre = instance.galaxies.source.bulge.centre

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=traced_grid,
            sub_size_list=[16, 8, 2],
            radial_list=[0.1, 0.3],
            centre_list=[source_centre],
        )

        over_sample_size_lens = (
            al.util.over_sample.over_sample_size_via_radial_bins_from(
                grid=dataset.grid,
                sub_size_list=[4, 2, 2],
                radial_list=[0.3, 0.6],
                centre_list=[(0.0, 0.0)],
            )
        )

        over_sample_size = np.where(
            over_sample_size > over_sample_size_lens,
            over_sample_size,
            over_sample_size_lens,
        )
        over_sample_size = al.Array2D(values=over_sample_size, mask=self.mask)

        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        """
        Outputs info about the `Tracer` to the fit, so we know exactly how we simulated the image.
        """
        self.output_info(simulate_path=simulate_path, dataset=dataset, tracer=tracer)

        return dataset

    def output_info(self, simulate_path: str, dataset: al.Imaging, tracer: al.Imaging):
        """
        Output information about the data simulated for this iteration of sensitivity mapping.

        This information output is as follows:

        - A subplot of the simulated imaging dataset.
        - A subplot of the tracer used to simulate this imaging dataset.
        - A .json file containing the tracer galaxies.

        Parameters
        ----------
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.
        dataset
            The simulated imaging dataset which is visualized.
        tracer
            The tracer used to simulate the imaging dataset, which is visualized and output to a .json file.
        """

        aplt.subplot_imaging_dataset(dataset=dataset)

        aplt.subplot_lensed_images(
            tracer=tracer,
            grid=dataset.grid,
            output_path=simulate_path,
            output_format="png",
        )

        al.output_to_json(
            obj=tracer,
            file_path=Path(simulate_path) / "tracer.json",
        )


"""
__Base Fit__

We have defined a `Simulate` class that will be used to simulate every dataset simulated by the sensitivity mapper.
Each simulated dataset will have a unique set of parameters for the `subhalo` (e.g. due to different values of
`perturb_model`.

We will fit each simulated dataset using the `base_model`, which quantifies whether not including the dark matter
subhalo in the model changess the goodness-of-fit and therefore indicates if we are sensitive to the subhalo.

We now write a `BaseFit` class, defining how the `base_model` is fitted to each simulated dataset and the 
goodness-of-fit used to quantify whether the model fits the data well. As above, the `__init__` method can be
extended with new inputs to control how the model is fitted and the `__call__` method performs the fit.

In this example, we use a full non-linear search to fit the `base_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `base_model` fit
to the simulated data.

Fucntionality which adapts the mesh and regularization of a pixelized source reconstruction to the unlensed source's 
morphology require an `adapt_images`. This is an input of the __init__ constructor which is passed to the `Analysis` 
for every simulated dataset.
"""


class BaseFit:
    def __init__(self, adapt_images, number_of_cores: int = 1):
        """
        Class used to fit every dataset used for sensitivity mapping with the base model (the model without the
        perturbed feature sensitivity mapping maps out).

        In this example, the base model therefore does not include the dark matter subhalo, but the simulated
        dataset includes one.

        The base fit is repeated for every parameter on the sensitivity grid and compared to the perturbed fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of the mass of the subhalo).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `AnalysisImaging` class used to fit the model
        to the dataset.

        Parameters
        ----------
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        number_of_cores
            The number of cores used to perform the non-linear search. If 1, each model-fit on the grid is performed
            in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
        """
        self.adapt_images = adapt_images
        self.number_of_cores = number_of_cores

    def __call__(self, dataset, model, paths, instance):
        """
        The base fitting function which fits every dataset used for sensitivity mapping with the base model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which does not include the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        instance
            The simulation instance, which includes the perturbed feature that is used to simulate the dataset.
            This is often not used, but may be useful for certain sensitivity mapping tasks, for example using
            true values of the simulated instance to set up aspects of the model-fit (e.g. the priors).
        """

        search = af.Nautilus(
            paths=paths,
            n_live=50,
            n_batch=50,
        )

        analysis = al.AnalysisImaging(dataset=dataset)
        analysis._adapt_images = self.adapt_images

        return search.fit(model=model, analysis=analysis)


"""
__Perturb Fit__

We now define a `PerturbFit` class, which defines how the `perturb_model` is fitted to each simulated dataset. This
behaves analogously to the `BaseFit` class above, but now fits the `perturb_model` to the simulated data (as
opposed to the `base_model`).

Again, in this example we use a full non-linear search to fit the `perturb_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `perturb_model` fit
to the simulated data.
"""


class PerturbFit:
    def __init__(self, adapt_images, number_of_cores: int = 1):
        """
        Class used to fit every dataset used for sensitivity mapping with the perturbed model (the model with the
        perturbed feature sensitivity mapping maps out).

        In this example, the perturbed model therefore includes the dark matter subhalo, which is also in the
        simulated dataset.

        The perturbed fit is repeated for every parameter on the sensitivity grid and compared to the base fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of mass of the subhalo).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `Analysis` class used to fit the model to the
        dataset.

        Parameters
        ----------
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        number_of_cores
            The number of cores used to perform the non-linear search. If 1, each model-fit on the grid is performed
            in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
        """
        self.adapt_images = adapt_images
        self.number_of_cores = number_of_cores

    def __call__(self, dataset, model, paths, instance):
        """
        The perturbed fitting function which fits every dataset used for sensitivity mapping with the perturbed model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which includes the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        instance
            The simulation instance, which includes the perturbed feature that is used to simulate the dataset.
            This is often not used, but may be useful for certain sensitivity mapping tasks, for example using
            true values of the simulated instance to set up aspects of the model-fit (e.g. the priors).
        """

        search = af.Nautilus(
            paths=paths,
            n_live=50,
            n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
        )

        analysis = al.AnalysisImaging(dataset=dataset)
        analysis._adapt_images = self.adapt_images

        return search.fit(model=model, analysis=analysis)


"""
We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
object below are:

- `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this example 
it is a lens model that does not include a subhalo, which was inferred by fitting the dataset we perform sensitivity 
mapping on.

- `base_model`: This is the lens model that is fitted to every simulated dataset, which does not include a subhalo. In 
this example is composed of an `Isothermal` lens and MGE source.

- `perturb_model`: This is the extra model component that alongside the `base_model` is fitted to every simulated 
dataset. In this example it is a `NFWMCRLudlowSph` dark matter subhalo.

- `simulate_cls`: This is the function that uses the `simulation_instance` and many instances of the `perturb_model` 
to simulate many datasets that are fitted with the `base_model` and `base_model` + `perturb_model`.

- `base_fit_cls`: This is the function that fits the `base_model` to every simulated dataset and returns the
goodness-of-fit of the model to the data.

- `perturb_fit_cls`: This is the function that fits the `base_model` + `perturb_model` to every simulated dataset and
returns the goodness-of-fit of the model to the data.

- `number_of_steps`: The number of steps over which the parameters in the `perturb_model` are iterated. In this 
example, each subhalo ``centre` has a `UniformPrior` with lower limit -3.0 and upper limit 3.0, therefore 
the `number_of_steps=2` will simulate and fit 4 datasets where the `centre` values 
are [(-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5)].

- `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel processing
if set above 1.
"""
paths = af.DirectoryPaths(
    path_prefix=Path("features"),
    name="sensitivity_mapping",
)

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result)

"""
__Adapt Image S/N Cap__

The source adapt image is capped at a signal-to-noise of 3.0 before it is used by the adaptive
image-mesh and the adaptive regularization. Without the cap the brightest peak dominates the
weights (they scale as a power of the adapt image), so fainter multiply-imaged features get too
few source pixels and too little regularization weight. Capping makes every feature above S/N 3.0
count equally. The cap is applied to an explicit copy so the raw S/N image is untouched.
"""
adapt_image_snr_cap = 3.0

source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

sensitivity = af.Sensitivity(
    paths=paths,
    simulation_instance=simulation_instance,
    base_model=base_model,
    perturb_model=perturb_model,
    simulate_cls=SimulateImaging(mask=mask, psf=dataset.psf),
    base_fit_cls=BaseFit(adapt_images=adapt_images),
    perturb_fit_cls=PerturbFit(adapt_images=adapt_images),
    perturb_model_prior_func=perturb_model_prior_func,
    number_of_steps=2,
    #    number_of_steps=(4, 2),
    number_of_cores=2,
)

sensitivity_result = sensitivity.run()

"""
__Results__

You should now look at the results of the sensitivity mapping in the folder `output/features/sensitivity_mapping`. 

You will note the following 4 sets of x2 model-fits have been performed:

 - The `base_model` is fitted to a simulated dataset where a subhalo is included at the (y,x) 
   coordinates [(-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5)].

 - The `base_model` + `perturb_model` is fitted to a simulated dataset where a subhalo is included at the (y,x) 
   coordinates [(-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5)].

The fit produces a `sensitivity_result`. 

We can print the `log_evidence_differences` of every cell of the sensitivity map.
"""
print(sensitivity_result.log_evidence_differences.native)
