"""
Pixelization: SLaM
==================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for pixelized source modeling.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that
guide before working through this example.

Because the SLaM pipelines are designed around pixelized source modeling, the example `slam_start_here` fully
describes all design choices and modeling decisions made in this script. This script therefore does not repeat
that documentation, therefore `slam_start_here` should be read first.

The interferometer SLaM pipeline has one different from the imaging SLaM pipeline, it omits the `source_lp`
pipeline and does not fit a model with a light profile source. This is because fitting light profiles
to datasets with many visibilities is slow, whereas pixelized sources are fast. This has two consequences:

- You must provide the multiple image locations used for the position likelihoods manually, whereas for the imaging
  SLaM pipeline they are estimated via the lens model fit in the `source_lp` pipeline.

- `source_pix[1]` does not have an adapt image and therefore uses a regularization which is not adaptive.

Other than that, the interferometer SLaM pipeline is identical to the imaging SLaM pipeline.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
**Interferometer SLaM Description:** The `slam_start_here` notebook provides a detailed description of the SLaM pipelines, but it does.
**High Resolution Dataset:** A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts.
**SOURCE PIX PIPELINE 1:** Unlike `slam_start_here.py`, this pipeline does not use a `source_lp` pipeline before the pixelized.
**SOURCE PIX PIPELINE 2:** Identical to `slam_start_here.py`, using adapt images from `source_pix_result_1` to improve the.
**MASS TOTAL PIPELINE:** Identical to `slam_start_here.py`, except no lens light model is included as interferometer data.
**Sparse Operators:** The `pixelization/modeling` example describes how the sparse operator formalism speeds up.
**Position Likelihood:** Load the multiple image positions used for the position likelihood, which resamples bad mass models.
**Settings:** Disable the default position only linear algebra solver so the source reconstruction can have.
**Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
**Redshifts:** The redshifts of the lens and source galaxies.
**Mesh Shape:** As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before.
**SLaM Pipeline:** The code below calls the full SLaM PIPELINE.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines
  and how they integrate into strong-lens modeling.

You can still run the script without fully understanding the guide, but reviewing it later will
make the structure and choices of the SLaM workflow clearer.

__Interferometer SLaM Description__

The `slam_start_here` notebook provides a detailed description of the SLaM pipelines, but it does this using CCD
imaging data.

There is no dedicated example which provides full descriptions of the SLaM pipelines using interferometer data, however,
the concepts and API described in the `slam_start_here` are identical to what is required for interferometer data.

Therefore, by reading the `slam_start_here` example you will fully understand everything required to use this
interferometer SLaM script.

__High Resolution Dataset__

A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts large files which
are too big to include in the main `autolens_workspace` repository:

https://github.com/Jammy2211/autolens_workspace_large_files

After downloading the file, place it in the directory:

`autolens_workspace/dataset/interferometer/alma`

You can then perform modeling using this high-resolution dataset by uncommenting the relevant line of code
below.
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
"""


def source_pix_1(
    settings_search: af.SettingsSearch,
    dataset,
    redshift_lens: float,
    redshift_source: float,
    positions_likelihood,
    mesh_shape,
    settings,
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
dataset_name = "simple"
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
        [sys.executable, "scripts/interferometer/simulator.py"],
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
Finish.
"""
