"""
Adaptive Pixelization (Group)
=============================

This script demonstrates how adaptive pixelization features work for group-scale strong lenses.

Adaptive pixelizations adapt the mesh density and regularization strength to the source galaxy's unlensed
morphology. More source pixels are placed where the source is brightest, and regularization is relaxed
in bright regions to preserve detail while remaining strong in faint regions to suppress noise.

For group-scale lenses, correct lens-light subtraction from ALL galaxies (main + extra) is critical
for producing good adapt_data, which drives the adaptive mesh. If the lens light subtraction is poor,
the adaptive mesh may concentrate pixels on residuals rather than genuine source emission.

This script uses search chaining: an initial parametric fit establishes the lens model and source
morphology, then subsequent searches use adaptive pixelization features.

__Contents__

**Dataset & Mask:** Standard set up of the group dataset and 7.5" mask.
**Galaxy Centres:** Load centres for main lens and extra galaxies.
**Search 1:** Fit a parametric model to establish the lens model and source morphology.
**Search 2:** Introduce a pixelization with constant regularization.
**Search 3:** Use adaptive mesh and regularization driven by adapt_data from search 2.
**Adapt Images:** How adapt_data is constructed from the lens-subtracted source image.

__Adaptive Features__

Two key adaptive classes are used:

 - `RectangularAdaptImage` mesh: adapts the rectangular source-pixel upsampling to the source's unlensed
   morphology. More rectangular pixels are placed where the source is located, even in low magnification
   regions.

 - `Adapt` regularization: adapts the regularization coefficient to the source's unlensed morphology.
   Bright regions are regularized less (preserving detail), faint regions are regularized more
   (suppressing noise).

For group lenses, the adapt_data is the lens-subtracted image, where the light of ALL group galaxies
has been removed. This means accurate modeling of every galaxy's light profile is essential.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset `simple`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Galaxy Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Paths__
"""
path_prefix = Path("group") / "features" / "pixelization" / "adaptive"

"""
__Model (Search 1)__

Search 1 fits a parametric group model to establish the lens mass model and source morphology.

 - Main lens galaxy: MGE light + Isothermal mass + ExternalShear.
 - Extra galaxies: MGE light + IsothermalSph mass (fixed centres, bounded Einstein radii).
 - Source: MGE light profile.
"""
# Main Lens Galaxies:

lens_models = []

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

    lens_models.append(lens)

# Extra Galaxies:

extra_galaxies_list = []

for centre in extra_galaxies_centres:

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_fixed=centre
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source (parametric):

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Overall Lens Model:

lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
lens_dict["source"] = source

model_1 = af.Collection(
    galaxies=af.Collection(**lens_dict),
    extra_galaxies=extra_galaxies,
)

print(model_1.info)

search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__parametric",
    unique_tag=dataset_name,
    n_live=100,
    n_like_max=500,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Mesh Shape__
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Model (Search 2)__

Search 2 introduces a pixelization with constant regularization. The lens mass model is taken from
search 1 as a model (with priors from the previous result).
"""
lens_dict_2 = {}
for i, _ in enumerate(main_lens_centres):
    lens_dict_2[f"lens_{i}"] = getattr(result_1.model.galaxies, f"lens_{i}")

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source_2 = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
lens_dict_2["source"] = source_2

model_2 = af.Collection(
    galaxies=af.Collection(**lens_dict_2),
    extra_galaxies=result_1.model.extra_galaxies,
)

search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__pixelization_setup",
    unique_tag=dataset_name,
    n_live=100,
    n_like_max=500,
)

analysis_2 = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[
        result_1.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Adaptive Pixelization (Search 3)__

Search 3 uses the adaptive pixelization classes:

 - `RectangularAdaptImage` mesh: adapts pixel density to the source morphology.
 - `Adapt` regularization: adapts smoothing strength to the source brightness.

The lens mass is fixed from search 2 to ensure the adaptation is performed quickly.

__Adapt Images__

The `adapt_images` are constructed from the lens-subtracted source image of search 2. For group lenses,
this means the light of ALL galaxies (main + extra) has been subtracted, leaving only the lensed source.

If the lens light subtraction is poor (e.g. because an extra galaxy's light was not modeled), the
adapt_data will contain residuals that confuse the adaptive mesh.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result_2)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

# Fix the lens mass model from search 2.
lens_dict_3 = {}
for i, _ in enumerate(main_lens_centres):
    lens_dict_3[f"lens_{i}"] = getattr(result_2.instance.galaxies, f"lens_{i}")

pixelization_3 = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularAdaptImage(shape=mesh_shape),
    regularization=al.reg.Adapt,
)

source_3 = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization_3)
lens_dict_3["source"] = source_3

model_3 = af.Collection(
    galaxies=af.Collection(**lens_dict_3),
    extra_galaxies=result_2.instance.extra_galaxies,
)

search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]__adaptive_pixelization",
    unique_tag=dataset_name,
    n_live=75,
    n_like_max=500,
)

analysis_3 = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Result (Search 3)__

The adaptive pixelization result should show significantly higher likelihood than search 2, with
the source reconstruction concentrating pixels in bright source regions.
"""
print(result_3.info)

aplt.subplot_fit_imaging(fit=result_3.max_log_likelihood_fit)

"""
__Search 4: Free Mass Model__

Finally, we refit the lens mass model with the adaptive pixelization fixed from search 3.
"""
lens_models_4 = []
for i, _ in enumerate(main_lens_centres):
    bulge_i = getattr(result_2.instance.galaxies, f"lens_{i}").bulge

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge_i,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )
    lens_models_4.append(lens)

extra_galaxies_4_list = []
for centre in extra_galaxies_centres:
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    extra_galaxies_4_list.append(extra_galaxy)

extra_galaxies_4 = af.Collection(extra_galaxies_4_list)

source_4 = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=result_3.instance.galaxies.source.pixelization,
)

lens_dict_4 = {f"lens_{i}": m for i, m in enumerate(lens_models_4)}
lens_dict_4["source"] = source_4

model_4 = af.Collection(
    galaxies=af.Collection(**lens_dict_4),
    extra_galaxies=extra_galaxies_4,
)

search_4 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[4]__adapt_free_mass",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_4 = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
)

result_4 = search_4.fit(model=model_4, analysis=analysis_4)

"""
__Wrap Up__

This script demonstrated adaptive pixelization for group-scale lenses.

Key points:
 - Adaptive pixelizations are set up via search chaining: parametric fit -> constant pixelization -> adaptive.
 - The adapt_data is the lens-subtracted image, so accurate light modeling of ALL group galaxies is essential.
 - `RectangularAdaptImage` concentrates source pixels where the source is brightest.
 - `Adapt` regularization varies smoothing based on source brightness.
 - The SLaM pipeline (see `group/features/pixelization/slam.py`) automates this entire process.
"""
