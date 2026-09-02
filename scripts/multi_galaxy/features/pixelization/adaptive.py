"""
Adaptive Pixelization (Multi Galaxy)
====================================

This script fits a multi-galaxy strong lens with an **adaptive** pixelized source, where both the mesh and the
regularization adapt to the source's own reconstructed morphology rather than being applied uniformly.

The adaptive schemes need an estimate of the source's surface brightness before they can adapt to it, which a
standalone script has no way of knowing up front. This script therefore runs four searches in sequence, each
using the previous one's result: the earlier searches produce the estimate, the later ones consume it.

__Contents__

- **Adaptive Features:** The two adaptive classes used and what each adapts.
- **Dataset:** Load the multi-galaxy dataset that is fitted.
- **Extra Galaxies Noise Scaling:** Scale the contaminating galaxy's light out of the fit.
- **Mask, Centres & Over Sampling:** Standard set up, over-sampled at every deflector centre.
- **Paths:** Where the four searches write their output.
- **Model (Search 1):** A parametric fit that establishes the mass model and source morphology.
- **Mesh Shape:** The resolution of the source-plane mesh.
- **Model (Search 2):** Introduce a pixelization with constant regularization.
- **Adapt Images:** How the adapt images are built from search 2's result.
- **Model (Search 3):** The adaptive mesh and adaptive regularization.
- **Model (Search 4):** Refit the mass model with the adaptive source fixed.
- **Wrap Up:** Where to go next.

__Adaptive Features__

Two adaptive classes are used, replacing the `RectangularBilinearAdaptDensity` mesh and `Constant` regularization of
`multi_galaxy/features/pixelization/modeling.py`:

 - `RectangularBilinearAdaptImage` mesh: places more source pixels where the source's adapt image is brighter, so
   resolution follows the source's light rather than the magnification pattern.

 - `Adapt` regularization: varies the smoothing strength across the source, regularizing bright regions less
   (preserving detail) and faint regions more (suppressing noise). It contributes two sampled parameters
   instead of `Constant`'s one.

Both read from **adapt images**: one image per galaxy, derived from an earlier fit. The source's adapt image is
what the two classes above adapt to.

__What Changes For Multiple Deflectors__

The source's adapt image is the data with every other galaxy's light model subtracted. At galaxy scale that is
one light model; here it is one per co-dominant deflector, plus the noise scaling applied to the faint
contaminant. Every deflector's light therefore has to be modelled well enough to subtract, which is why search 1
below fits an MGE to each deflector before any pixelization is introduced.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/features/pixelization/modeling.py` for the non-adaptive
version of this model, and `imaging/features/pixelization/adaptive.py` for the galaxy-scale walkthrough.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `simple` multi-galaxy dataset, the same co-dominant pair fitted by `multi_galaxy/modeling.py`.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Extra Galaxies Noise Scaling__

Scale the faint contaminant out of the fit, as `multi_galaxy/modeling.py` explains.

This is done before the adapt images are built, so the contaminant's flux is not present in the source's adapt
image and cannot draw source pixels towards it.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.
"""
mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Paths__

All four searches write to the same folder, one sub-folder per search.
"""
path_prefix = Path("multi_galaxy") / "features" / "pixelization" / "adaptive"

"""
__Model (Search 1)__

Search 1 fits a fully parametric model, with an MGE source in place of a pixelization. Its job is to produce a
lens model good enough that search 2's pixelized fit starts from sensible priors, and light models good enough to
subtract when the adapt images are built.

The composition is the standard multi-galaxy one from `multi_galaxy/modeling.py`: one `lens_i` per deflector with
its mass centre fixed to the deflector's measured centre, and the shear in its own `shear_galaxy`.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

# External Shear:

shear_galaxy = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

# Source (parametric):

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model_1 = af.Collection(
    galaxies=af.Collection(**lens_dict, shear_galaxy=shear_galaxy, source=source)
)

print(model_1.info)

search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__parametric",
    unique_tag=dataset_name,
    n_live=100,
    n_like_max=500,
)

analysis_1 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Mesh Shape__

The shape of the source-plane mesh, fixed rather than fitted for the reason given in
`multi_galaxy/features/pixelization/modeling.py`.
"""
mesh_shape = (28, 28)

"""
__Model (Search 2)__

Search 2 swaps the parametric source for a pixelization with `Constant` regularization, which needs no adapt
images. Each deflector is passed through from search 1 as a model, so its priors are centred on search 1's
result.

The `positions_likelihood_from` method takes the multiple images from search 1's result rather than requiring
them as input, which matters more here than at galaxy scale: two deflectors produce more multiple images in more
complex configurations than one.
"""
lens_dict_2 = {}

for i, _ in enumerate(main_lens_centres):
    lens_dict_2[f"lens_{i}"] = getattr(result_1.model.galaxies, f"lens_{i}")

pixelization_2 = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source_2 = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization_2)

model_2 = af.Collection(
    galaxies=af.Collection(
        **lens_dict_2,
        shear_galaxy=result_1.model.galaxies.shear_galaxy,
        source=source_2,
    )
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
    use_jax=True,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Adapt Images__

`galaxy_name_image_dict_via_result_from` returns one image per galaxy in the model — here `lens_0`, `lens_1`,
`shear_galaxy` and `source` — taken from search 2's maximum log likelihood fit. `AdaptImages` wraps that
dictionary into the object the adaptive classes read.

The entry the adaptive mesh and regularization use is `source`, which is the data with both deflectors' light
models subtracted.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result_2)

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

"""
__Model (Search 3)__

Search 3 introduces the adaptive mesh and adaptive regularization, with every deflector fixed as an instance
from search 2. Fixing them keeps this search cheap: the only free parameters are the source's regularization,
so the fit is measuring how the adaptation behaves rather than re-measuring the mass model.
"""
lens_dict_3 = {}

for i, _ in enumerate(main_lens_centres):
    lens_dict_3[f"lens_{i}"] = getattr(result_2.instance.galaxies, f"lens_{i}")

pixelization_3 = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularBilinearAdaptImage(shape=mesh_shape),
    regularization=al.reg.Adapt,
)

source_3 = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization_3)

model_3 = af.Collection(
    galaxies=af.Collection(
        **lens_dict_3,
        shear_galaxy=result_2.instance.galaxies.shear_galaxy,
        source=source_3,
    )
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
    use_jax=True,
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Result (Search 3)__

The source reconstruction plotted below is the adaptive one: its pixels are concentrated where the source's
adapt image is brightest, rather than spread by magnification alone.
"""
print(result_3.info)

aplt.subplot_fit_imaging(fit=result_3.max_log_likelihood_fit)

"""
__Model (Search 4)__

Search 4 refits the mass model with the adaptive source fixed from search 3. Each deflector's light is fixed to
its search 2 instance and its mass is freed, with `unfix_mass_centre=True` converting the centre fixed in search
1 into a free parameter with a prior around it.

This is the search that uses the adaptive source for what it is for: the mass model is now being constrained by
a source reconstruction that resolves the arcs' structure, rather than by one smoothed uniformly.
"""
lens_dict_4 = {}

for i, _ in enumerate(main_lens_centres):

    lens_instance = getattr(result_2.instance.galaxies, f"lens_{i}")
    lens_model = getattr(result_2.model.galaxies, f"lens_{i}")

    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.Isothermal),
        mass_result=lens_model.mass,
        unfix_mass_centre=True,
    )

    lens_dict_4[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=lens_instance.bulge,
        mass=mass,
    )

source_4 = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=result_3.instance.galaxies.source.pixelization,
)

model_4 = af.Collection(
    galaxies=af.Collection(
        **lens_dict_4,
        shear_galaxy=result_2.model.galaxies.shear_galaxy,
        source=source_4,
    )
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
    positions_likelihood_list=[
        result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    use_jax=True,
)

result_4 = search_4.fit(model=model_4, analysis=analysis_4)

print(result_4.info)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/pixelization/modeling.py` — the non-adaptive pixelized fit this script starts from.
 - `multi_galaxy/features/pixelization/delaunay.py` — the Delaunay meshes, which support the split
   regularization schemes the rectangular meshes do not.
 - `multi_galaxy/slam.py` — the SLaM pipeline, which automates this chain of searches.
 - `imaging/features/pixelization/adaptive.py` — the galaxy-scale walkthrough of the same adaptive classes.
"""
