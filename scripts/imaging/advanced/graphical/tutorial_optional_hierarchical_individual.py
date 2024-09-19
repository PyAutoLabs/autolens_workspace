"""
Tutorial Optional: Hierarchical Individual
==========================================

In tutorial 4, we fit a hierarchical model using a graphical model, whereby all datasets are fitted simultaneously
and the hierarchical parameters are fitted for simultaneously with the model parameters of each lens in each
dataset.

This script illustrates how the hierarchical parameters can be estimated using a simpler approach, which fits
each dataset one-by-one and estimates the hierarchical parameters afterwards by fitting the inferred `slope`'s
with a Gaussian distribution.

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autolens_workspace/scripts/simulators/imaging/samples/advanced/mass_power_law.py`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af
import autolens as al

"""
__Dataset__

For each lens dataset in our sample we set up the correct path and load it by iterating over a for loop. 

We are loading a different dataset to the previous tutorials, where the lenses only have a single bulge component
which each have different Sersic indexes which are drawn from a parent Gaussian distribution with a mean centre value 
of 4.0 and sigma of 1.0.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autolens_workspace/scripts/simulators/imaging/samples/advanced/mass_power_law.py`. 
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "mass_power_law"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_sample_name)

total_datasets = 3

dataset_list = []

for dataset_index in range(total_datasets):
    dataset_sample_path = path.join(dataset_path, f"dataset_{dataset_index}")

    dataset_list.append(
        al.Imaging.from_fits(
            data_path=path.join(dataset_sample_path, "data.fits"),
            psf_path=path.join(dataset_sample_path, "psf.fits"),
            noise_map_path=path.join(dataset_sample_path, "noise_map.fits"),
            pixel_scales=0.1,
        )
    )

"""
__Mask__

We now mask each lens in our dataset, using the imaging list we created above.

We will assume a 3.0" mask for every lens in the dataset is appropriate.
"""
masked_imaging_list = []

for dataset in dataset_list:
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    masked_imaging_list.append(dataset.apply_mask(mask=mask))


"""
__Paths__

The path the results of all model-fits are output:
"""
path_prefix = path.join("imaging", "hierarchical")

"""
__Analysis__

For each dataset we now create a corresponding `AnalysisImaging` class, as we are used to doing for `Imaging` data.
"""
analysis_list = []

for masked_dataset in masked_imaging_list:
    analysis = al.AnalysisImaging(dataset=masked_dataset)

    analysis_list.append(analysis)


"""
__Model__

The model we fit to each dataset, which is a `PowerLawSph` lens mass model and `ExponentialSph` source.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.PowerLawSph)
lens.mass.centre = (0.0, 0.0)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.ExponentialCoreSph)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Model Fits (one-by-one)__

For every dataset we now create an `Analysis` class using it and use `Nautilus` to fit it with a lens model.

The `Result` is stored in the list `results`.
"""
result_list = []

for dataset_index, analysis in enumerate(analysis_list):
    dataset_name = f"dataset_{dataset_index}"

    """
    Create the `Nautilus` non-linear search and use it to fit the data.
    """
    Nautilus = af.Nautilus(
        name="",
        path_prefix=path.join("tutorial_optional_hierarchical_individual"),
        unique_tag=dataset_name,
        n_live=200,
        f_live=1e-4,
    )

    result_list.append(Nautilus.fit(model=model, analysis=analysis))

"""
__Results__

Checkout the output folder, you should see three new sets of results corresponding to our 3 datasets.

The `result_list` allows us to plot the median PDF value and 3.0 confidence intervals of the `slope` estimate 
from the model-fit to each dataset.
"""
samples_list = [result.samples for result in result_list]

mp_instances = [samps.median_pdf() for samps in samples_list]
ue3_instances = [samp.errors_at_upper_sigma(sigma=3.0) for samp in samples_list]
le3_instances = [samp.errors_at_lower_sigma(sigma=3.0) for samp in samples_list]

mp_slopes = [instance.lenses.lens.bulge.slope for instance in mp_instances]
ue3_slopes = [instance.lenses.lens.bulge.slope for instance in ue3_instances]
le3_slopes = [instance.lenses.lens.bulge.slope for instance in le3_instances]

print(f"Median PDF inferred slope values")
print(mp_slopes)
print()

"""
__Overall Gaussian Parent Distribution__

Fit the inferred `slope`'s from the fits performed above with a Gaussian distribution, in order to 
estimate the mean and scatter of the Gaussian from which the Sersic indexes were drawn.

We first extract the inferred median PDF Sersic index values and their 1 sigma errors below, which will be the inputs
to our fit for the parent Gaussian.
"""
ue1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_slopes = [instance.lenses.lens.bulge.slope for instance in ue1_instances]
le1_slopes = [instance.lenses.lens.bulge.slope for instance in le1_instances]

error_list = [ue1 - le1 for ue1, le1 in zip(ue1_slopes, le1_slopes)]

"""
The `Analysis` class below fits a Gaussian distribution to the inferred `slope` values from each of the fits above,
where the inferred error values are used as the errors.
"""


class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, errors: np.ndarray):
        super().__init__()

        self.data = np.array(data)
        self.errors = np.array(errors)

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Fits a set of 1D data points with a 1D Gaussian distribution, in order to determine from what Gaussian
        distribution the analysis classes `data` were drawn.

        In this example, this function determines from what parent Gaussian disrtribution the inferred slope
        of each lens were drawn.
        """
        log_likelihood_term_1 = np.sum(
            -np.divide(
                (self.data - instance.median) ** 2,
                2 * (instance.scatter**2 + self.errors**2),
            )
        )
        log_likelihood_term_2 = -np.sum(
            0.5 * np.log(instance.scatter**2 + self.errors**2)
        )

        return log_likelihood_term_1 + log_likelihood_term_2


"""
The `ParentGaussian` class is the model-component which used to fit the parent Gaussian to the inferred `slope` values.
"""


class ParentGaussian:
    def __init__(self, median: float = 0.0, scatter: float = 0.01):
        """
        A model component which represents a parent Gaussian distribution, which can be fitted to a 1D set of
        measurments with errors in order to determine the probabilty they were drawn from this Gaussian.

        Parameters
        ----------
        median
            The median value of the parent Gaussian distribution.
        scatter
            The scatter (E.g. the sigma value) of the Gaussian.
        """

        self.median = median
        self.scatter = scatter

    def probability_from_values(self, values: np.ndarray) -> float:
        """
        For a set of 1D values, determine the probability that they were random drawn from this parent Gaussian
        based on its `median` and `scatter` attributes.

        Parameters
        ----------
        values
            A set of 1D values from which we will determine the probability they were drawn from the parent Gaussian.
        """
        values = np.sort(np.array(values))
        transformed_values = np.subtract(values, self.median)

        return np.multiply(
            np.divide(1, self.scatter * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_values, self.scatter))),
        )


"""
__Model__

The `ParentGaussian` is the model component we fit in order to determine the probability the inferred Sersic indexes 
were drawn from the distribution.

This will be fitted via a non-linear search and therefore is created as a model component using `af.Model()` as per 
usual in **PyAutoFit**.
"""
model = af.Model(ParentGaussian)

model.median = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.scatter = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

"""
__Analysis + Search__

We now create the Analysis class above which fits a parent 1D gaussian and create a Nautilus search in order to fit
it to the 1D inferred list of `slope`'s.
"""
analysis = Analysis(data=mp_slopes, errors=error_list)
search = af.Nautilus(n_live=150)

result = search.fit(model=model, analysis=analysis)

"""
The results of this fit tell us the most probable values for the `median` and `scatter` of the 1D parent Gaussian fit.
"""
samples = result.samples

median = samples.median_pdf().median

u1_error = samples.values_at_upper_sigma(sigma=1.0).median
l1_error = samples.values_at_lower_sigma(sigma=1.0).median

u3_error = samples.values_at_upper_sigma(sigma=3.0).median
l3_error = samples.values_at_lower_sigma(sigma=3.0).median

print(
    f"Inferred value of the hierarchical median via simple fit to {total_datasets} datasets: \n "
)
print(f"{median} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{median} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")
print()

scatter = samples.median_pdf().scatter

u1_error = samples.values_at_upper_sigma(sigma=1.0).scatter
l1_error = samples.values_at_lower_sigma(sigma=1.0).scatter

u3_error = samples.values_at_upper_sigma(sigma=3.0).scatter
l3_error = samples.values_at_lower_sigma(sigma=3.0).scatter

print(
    f"Inferred value of the hierarchical scatter via simple fit to {total_datasets} datasets: \n "
)
print(f"{scatter} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{scatter} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")
print()

"""
We can compare these values to those inferred in `tutorial_4_hierarchical_model`, which fits all datasets and the
hierarchical values of the parent Gaussian simultaneously.,
 
The errors for the fit performed in this tutorial are much larger. This is because of how in a graphical model
the "datasets talk to one another", which is described fully in that tutorials subsection "Benefits of Graphical Model".
"""
