"""
Extra Galaxies: SLaM
=====================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for fitting a
lens model where extra galaxies surrounding the lens are included in the lens model.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that
guide before working through this example.

This example only provides documentation specific to the extra galaxies, describing how the pipeline
differs from the standard SLaM pipelines described in the SLaM start here guide.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
**Group SLaM:** This SLaM pipeline is designed for the regime where one is modeling galaxy scale lenses with nearby.
**This Script:** Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this.
**SOURCE PIX PIPELINE 1:** Unlike `slam_start_here.py`, this pipeline does not use a `source_lp` pipeline before the pixelized.
**SOURCE PIX PIPELINE 2:** Identical to `slam_start_here.py`, using adapt images from `source_pix_result_1` to improve the.
**MASS TOTAL PIPELINE:** Identical to `slam_start_here.py`, except no lens light model is included as interferometer data.
**Extra Galaxies Centres:** This is the same API as described in the `features/extra_galaxies.ipynb` example, where the centres.
**Sparse Operators:** The `pixelization/modeling` example describes how the sparse operator formalism speeds up.
**Position Likelihood:** Load the multiple image positions used for the position likelihood, which resamples bad mass models.
**Settings:** Disable the default position only linear algebra solver so the source reconstruction can have.
**Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
**Redshifts:** The redshifts of the lens and source galaxies.
**Mesh Shape:** As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before.
**Extra Galaxies:** Build the extra galaxies model: each extra galaxy has an `IsothermalSph` mass profile with its.
**SLaM Pipeline:** The code below calls the full SLaM PIPELINE.
**Output:** The `start_here.ipynb` example describes how results can be output to hard-disk after the SLaM.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines
  and how they integrate into strong-lens modeling.

- **Extra Galaxies** (`features/extra_galaxies.ipynb`):
    How we include extra galaxies in the lens model, by using the centres of the galaxies
    which have been determined beforehand.

You can still run the script without fully understanding the guide, but reviewing it later will
make the structure and choices of the SLaM workflow clearer.

__Group SLaM__

This SLaM pipeline is designed for the regime where one is modeling galaxy scale lenses with nearby surrounding
extra galaxies.

However, these systems can often become close to the group scale lensing regime, for which PyAutoLens has a dedicated
package for modeling (`autolens_workspace/*/group`) and its own dedicated SLaM pipelines.

The main difference between this SLaM pipeline and the group SLaM pipelines is that in the latter, the masses of
the extra galaxies are modeled using scaling relations tied to their light profiles. The group SLaM pipeline has
additional searches in the SOURCE LP PIPELINE to measure the luminosities of the extra galaxies for this purpose.

Which SLaM pipeline you should use depends on your particular strong lens, but as a rule of thumb if you are
including a lot of extra galaxies (e.g. more than 5) and your model complexity is increasing significantly, you should
consider using the group SLaM pipelines.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script  fits `Imaging` dataset  of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.
 - Two extra galaxies are included in the model, each with their mass as a `IsothermalSph` profile.

This modeling script uses the SLaM pipelines:

 `source_pix`
 `light_lp`
 `mass_total`

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/slam_start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__SOURCE PIX PIPELINE 1__

Unlike `slam_start_here.py`, this pipeline does not use a `source_lp` pipeline before the pixelized source
pipeline. This is because fitting light profiles to interferometer datasets with many visibilities is slow.

The search therefore uses a `Constant` regularization (not adaptive) as there is no adapt image available.

The `extra_galaxies` are included in the model, each with an `IsothermalSph` mass profile whose centre is fixed
to the centre of the extra galaxy.
"""


def source_pix_1(
    settings_search: af.SettingsSearch,
    dataset,
    redshift_lens: float,
    redshift_source: float,
    positions_likelihood,
    mesh_shape,
    settings,
    extra_galaxies,
    n_batch: int = 20,
) -> af.Result:
    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        positions_likelihood_list=[positions_likelihood],
        settings=settings,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=None,
                disk=None,
                mass=af.Model(al.mp.Isothermal),
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
                    regularization=al.reg.Constant,
                ),
            ),
        ),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Identical to `slam_start_here.py`, using adapt images from `source_pix_result_1` to improve the source
pixelization and regularization.

The extra galaxies are passed from `source_pix_result_1` as fixed instances.

Note that the LIGHT LP PIPELINE from `slam_start_here.py` is omitted here, as interferometer data does not
contain lens light emission.
"""


def source_pix_2(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mesh_shape,
    settings,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1, use_model_images=True
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        settings=settings,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_pix_result_1.instance.galaxies.lens.redshift,
                bulge=source_pix_result_1.instance.galaxies.lens.bulge,
                disk=source_pix_result_1.instance.galaxies.lens.disk,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_pix_result_1.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        extra_galaxies=source_pix_result_1.instance.extra_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Identical to `slam_start_here.py`, except no lens light model is included as interferometer data does not
contain lens light emission.

The extra galaxies are passed from `source_pix_result_1` as free model parameters, so their masses are
updated during this search.
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    source_pix_result_2: af.Result,
    settings,
    n_batch: int = 20,
) -> af.Result:
    # Total mass model for the lens galaxy.
    mass = af.Model(al.mp.PowerLaw)

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1, use_model_images=True
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_pix_result_1.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        settings=settings,
    )

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_pix_result_1.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    source = al.util.chaining.source_from(result=source_pix_result_2)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_pix_result_1.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=mass,
                shear=source_pix_result_1.model.galaxies.lens.shear,
            ),
            source=source,
        ),
        extra_galaxies=source_pix_result_1.model.extra_galaxies,
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset + Masking__

Load the `Interferometer` data, define the visibility and real-space masks.
"""
dataset_name = "extra_galaxies"
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256), pixel_scales=0.1, radius=mask_radius
)

dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/extra_galaxies/simulator.py"],
        check=True,
    )

# dataset_name = "alma"

# if dataset_name == "alma":
#
#     real_space_mask = al.Mask2D.circular(
#         shape_native=(800, 800),
#         pixel_scales=0.01,
#         radius=mask_radius,
#     )


dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

"""
__Extra Galaxies Centres__

This is the same API as described in the `features/extra_galaxies.ipynb` example, where the centres of the extra
galaxies are loaded from a `.json` file.
"""
extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "extra_galaxies_centres.json"))
)

print(extra_galaxies_centres)

"""
__Sparse Operators__

The `pixelization/modeling` example describes how the sparse operator formalism speeds up interferometer
pixelized source modeling, especially for many visibilities.

We use a try / except to load the pre-computed curvature preload, which is necessary to use
the sparse operator formalism. If this file does not exist (e.g. you have not made it manually via
the `many_visibilities_preparartion` example it is made here.
"""
try:
    nufft_precision_operator = np.load(
        file=dataset_path / "nufft_precision_operator.npy",
    )
except FileNotFoundError:
    nufft_precision_operator = None

dataset = dataset.apply_sparse_operator(
    nufft_precision_operator=nufft_precision_operator, use_jax=True, show_progress=True
)

"""
__Position Likelihood__

Load the multiple image positions used for the position likelihood, which resamples bad mass
models and prevents demagnified solutions being inferred.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "positions.json"))
)

positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
__Settings__

Disable the default position only linear algebra solver so the source reconstruction can have
negative pixel values.
"""
settings = al.Settings(use_positive_only_solver=False)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("interferometer") / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Mesh Shape__

As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before modeling.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Extra Galaxies__

Build the extra galaxies model: each extra galaxy has an `IsothermalSph` mass profile with its centre fixed
to the known extra galaxy centre and a free `einstein_radius`.
"""
extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    extra_galaxy.mass.centre = extra_galaxy_centre

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

"""
__SLaM Pipeline__

The code below calls the full SLaM PIPELINE. See the documentation string above each Python function for
a description of each pipeline step.
"""
source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
    positions_likelihood=positions_likelihood,
    mesh_shape=mesh_shape,
    settings=settings,
    extra_galaxies=extra_galaxies,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    mesh_shape=mesh_shape,
    settings=settings,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    source_pix_result_2=source_pix_result_2,
    settings=settings,
)

"""
__Output__

The `start_here.ipynb` example describes how results can be output to hard-disk after the SLaM pipelines have been run.
Checkout that script for a complete description of the output of this script.
"""
