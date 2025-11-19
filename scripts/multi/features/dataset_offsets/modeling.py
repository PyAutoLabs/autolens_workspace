"""
Modeling Features: Dataset Offsets
==================================

Multi-wavelength datasets often have offsets between their images, which are due to the different telescope pointings
during the observations.

These offsets are often accounted for during the data reduction process, which aligns the images, however:

 - Certain data reduction pipelines may not perfectly align the images, and the scientist may be unsure what the
 true offset between the images are.

 - Even if the reduction process does align the images, there is still a small uncertainty in the offset due to the
   precision of the telescope pointing which for detailed lens models must be accounted for.

This script shows how to include an offset in the model, which is two free parameters, the y and x offsets, for every
additional dataset after the first dataset. The offset therefore describes the offset of each dataset in the
multi-wavelength dataset relative to the first dataset.

To apply the offset, the code simply subtracts the offset from the grids aligned to the dataset pixels before performing
lensing calculations. This means that the light and mass model centres do not change when the offset is applied, only
the coordinates of the image pixels which are input into these profiles to compute the images.

__Advantages__

If one fits a lens model to one dataset and applies it to other datasets, it is common to see the lens model fit
and source reconsturction degrade due to small offsets between the datasets. The same issue persists for simultaneous
fits to multiple datasets, even when care has been taken to align the datasets.

The advantage is therefore simple, for most multi-wavelength lens modeling, accounting for offsets in this way
is the only way to ensure the lens model is accurate and the source reconstruction is reliable.

__Disadvantages__

Each offset introduces two additional free parameters into the model for each dataset after the first dataset. For
4 datasets, this is 6 additional free parameters. This increases the dimensionality of the non-linear parameter space
and therefore the computational run-time of the model-fit.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a an MGE bulge.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a an MGE.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).

The strings are used for load each dataset.
"""
color_list = ["g", "r"]

"""
__Pixel Scales__

Every multi-wavelength dataset can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.12]

"""
__Dataset__

Load and plot each multi-wavelength strong lens dataset, using a list of their waveband colors.

The plotted images show that the datasets have a small offset between them, half a pixel based on the resolution of
the second image.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "dataset_offsets"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

dataset_list = [
    al.Imaging.from_fits(
        data_path=Path(dataset_path) / f"{color}_data.fits",
        psf_path=Path(dataset_path) / f"{color}_psf.fits",
        noise_map_path=Path(dataset_path) / f"{color}_noise_map.fits",
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.

For multi-wavelength lens modeling, we use the same mask for every dataset whenever possible. This is not
absolutely necessary, but provides a more reliable analysis.

The small offset between datasets means that the mask may not contain the exact same area of the image for every
dataset. 

For this dataset's offset of half a pixel (and anything of order a few pixels) this is fine and wont impact the analysis. 
However, for larger offsets the mask may need to be adjusted to ensure the same image area is masked out.
"""
mask_radius = 3.0

mask_list = [
    al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    for dataset in dataset_list
]

dataset_list = [
    dataset.apply_mask(mask=mask) for imaging, mask in zip(dataset_list, mask_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Analysis__

We create an `Analysis` object for every dataset.
"""
analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

"""
__Model__

We compose a lens model where:

 - Parameters which shift the second dataset's image (y_offset_0, x_offset_0) relative to the first dataset's image
 are included via the `DatasetModel` object [2 parameters].

 - The lens galaxy's light is an MGE with 1 x 20 Gaussians [6 parameters].

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's light is an MGE with 1 x 20 Gaussians [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=23.
"""
bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=True,
)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

dataset_model = af.Model(al.DatasetModel)

model = af.Collection(
    dataset_model=dataset_model, galaxies=af.Collection(lens=lens, source=source)
)

"""
The `DatasetModel` can apply a shift to each dataset, however we do not want this to be applied to both
datasets as they will then both have free parameters for the same offset, duplicating free parameters.

The default prior on a `DatasetModel`'s offsets are actually not a prior, but fixed values of (0.0, 0.0),
meaning that if we do not update the model the shift will not be applied to the datasets.

We therefore update the `DatasetModel` below, to only apply a shift to the second dataset, which is the r-band image.

If we add more datasets, the code will apply the shift to each one after the first dataset, which are all shifted
relative to the first dataset, making it the reference point.
"""
analysis_factor_list = []

for i, analysis in enumerate(analysis_list):
    model_analysis = model.copy()

    if i > 0:
        model_analysis.dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
            lower_limit=-1.0, upper_limit=1.0
        )
        model_analysis.dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
            lower_limit=-1.0, upper_limit=1.0
        )

    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=Path("multi", "modeling"),
    name="dataset_offsets",
    unique_tag=dataset_name,
    n_live=100,
)

"""
__Model-Fit__
"""
result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result object returned by this model-fit is a list of `Result` objects, because we used a factor graph.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.
"""
print(result_list[0].max_log_likelihood_instance)
print(result_list[1].max_log_likelihood_instance)

"""
Plotting each result's tracer shows that the source appears different, owning to its different intensities.
"""
for result in result_list:
    tracer_plotter = aplt.TracerPlotter(
        tracer=result.max_log_likelihood_tracer, grid=result.grids.lp
    )
    tracer_plotter.subplot_tracer()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
The `Samples` object still has the dimensions of the overall non-linear search (in this case N=15). 

Therefore, the samples is identical in every result object.
"""
for result in result_list:
    plotter = aplt.NestPlotter(samples=result.samples)
    plotter.corner_anesthetic()

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results in **PyAutoLens**.
"""
