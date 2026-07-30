"""
Simulator: Sample (Multi Galaxy)
================================

This script illustrates how to simulate a sample of `Imaging` datasets of 'multi-galaxy' strong lenses, which can
easily be used to simulate hundreds or thousands of them.

A multi-galaxy lens has two or more galaxies of comparable mass which both contribute significantly to the lensing of
a single background source. Simulating a *sample* of them is especially useful, because the two properties that
define the regime — the **mass ratio** of the pair and their **separation** — are exactly the population properties
you want to vary when testing how well a multi-deflector model can be constrained.

Each lens and source galaxy is set up by drawing parameters from distributions, so the parameters of each simulated
strong lens vary across the sample. This script uses the signal-to-noise based light profiles described in
`imaging/features/simulator_manual_signal_to_noise_ratio.ipynb`, to make it straight forward to ensure the lens and
source galaxies are visible in each image.

__Co-Dominance Is A Constraint, Not An Accident__

The important difference from `imaging/simulator_sample.py` is that the two deflectors cannot be drawn
independently. If each galaxy's Einstein radius were drawn from the same broad distribution, many draws would
produce one galaxy several times more massive than the other — which is a *galaxy-scale lens with a perturber*, not a
multi-galaxy lens, and belongs in `imaging/`.

We therefore draw the first galaxy's Einstein radius, then draw the second as a **ratio** of the first within a band
that keeps them comparable. Likewise the separation is drawn small relative to the combined Einstein radius, so the
lensed arcs wrap around the pair as a whole rather than forming two independent rings.

__Contents__

- **Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a name.
- **Simulate:** Simulate the image using a (y,x) grid with the adaptive over sampling scheme.
- **Sample Truth Distributions:** Draw random co-dominant pairs plus a source.
- **Sample Instances:** Generate and output each dataset in a for loop.

__Model__

This script simulates a sample of `Imaging` data of 'multi-galaxy' strong lenses where:

 - Each lens galaxy's light profile is a `Sersic`.
 - Each lens galaxy's total mass distribution is an `Isothermal`, with comparable Einstein radii.
 - The system has a single overall `ExternalShear`, held at the system centre rather than on either galaxy.
 - The source galaxy's light profile is a `Sersic`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `multi_galaxy/simulator.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.
"""
dataset_label = "samples"
dataset_type = "multi_galaxy"
dataset_sample_name = "simple"

"""
The path where the dataset will be output.
"""
dataset_path = Path("dataset", dataset_type, dataset_label, dataset_sample_name)

"""
__Simulate__

Simulate the image using a (y,x) grid with the adaptive over sampling scheme.

Unlike the single-galaxy sample simulator, the adaptive over-sampling centres cannot be fixed at (0.0", 0.0"),
because each simulated lens has its galaxies in different places. We therefore build the over-sampling grid *inside*
the loop below, once each pair's centres are known.

The 0.05"/pixel resolution matches Hubble Space Telescope ACS imaging.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.08, pixel_scales=grid.pixel_scales
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=900.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Sample Truth Distributions__

To simulate a sample, we draw random instances of lens and source galaxies. Each parameter
is sampled directly from a numpy ``Generator`` and used to construct concrete light/mass
profile instances — there is no ``af.Model`` involved here because we are generating
*truths* for synthetic data, not fitting a model.

The bulges use ``al.lp_snr.Sersic`` so each lens/source hits a target signal-to-noise
ratio in the data — SNR is a property of the data, not a fitted parameter.

The co-dominance constraints described at the top of this script are enforced in
``_random_multi_galaxy_lens`` below:

 - ``einstein_radius_0`` is drawn broadly, then ``einstein_radius_1`` is set by a ratio drawn in [0.6, 1.0], so the
   two deflectors are always within a factor of ~1.7 of each other in mass.
 - The pair's separation is drawn as a fraction of the combined Einstein radius, so both galaxies always sit well
   inside the lensed images.
 - The two galaxies are placed symmetrically about the origin at a random position angle, keeping the system centred
   on (0.0", 0.0") — which is where the external shear acts.
"""
rng = np.random.default_rng()


def _clipped_ell_comp() -> float:
    return float(np.clip(rng.normal(0.0, 0.2), -1.0, 1.0))


def _random_multi_galaxy_lens():
    """
    Draw a random co-dominant pair of deflectors, an external shear and a source galaxy.

    Returns the two lens galaxies, a galaxy holding the system's external shear, the source
    galaxy, and the list of the two lens centres (needed to build the over-sampling grid).
    """
    einstein_radius_0 = float(rng.uniform(0.6, 1.4))
    einstein_radius_1 = einstein_radius_0 * float(rng.uniform(0.6, 1.0))

    einstein_radius_combined = einstein_radius_0 + einstein_radius_1

    # Separation is a fraction of the combined Einstein radius, so both galaxies sit inside the arcs.
    separation = einstein_radius_combined * float(rng.uniform(0.15, 0.45))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))

    offset_y = 0.5 * separation * float(np.sin(angle))
    offset_x = 0.5 * separation * float(np.cos(angle))

    centre_0 = (offset_y, offset_x)
    centre_1 = (-offset_y, -offset_x)

    lens_galaxy_0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_snr.Sersic(
            centre=centre_0,
            ell_comps=(_clipped_ell_comp(), _clipped_ell_comp()),
            effective_radius=float(rng.uniform(0.3, 1.0)),
            sersic_index=float(np.clip(rng.normal(4.0, 0.5), 0.8, 5.0)),
            signal_to_noise_ratio=float(rng.uniform(20.0, 60.0)),
        ),
        mass=al.mp.Isothermal(
            centre=centre_0,
            ell_comps=(_clipped_ell_comp(), _clipped_ell_comp()),
            einstein_radius=einstein_radius_0,
        ),
    )

    lens_galaxy_1 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_snr.Sersic(
            centre=centre_1,
            ell_comps=(_clipped_ell_comp(), _clipped_ell_comp()),
            effective_radius=float(rng.uniform(0.3, 1.0)),
            sersic_index=float(np.clip(rng.normal(4.0, 0.5), 0.8, 5.0)),
            signal_to_noise_ratio=float(rng.uniform(20.0, 60.0)),
        ),
        mass=al.mp.Isothermal(
            centre=centre_1,
            ell_comps=(_clipped_ell_comp(), _clipped_ell_comp()),
            einstein_radius=einstein_radius_1,
        ),
    )

    shear_galaxy = al.Galaxy(
        redshift=0.5,
        shear=al.mp.ExternalShear(
            gamma_1=float(rng.normal(0.0, 0.05)),
            gamma_2=float(rng.normal(0.0, 0.05)),
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_snr.Sersic(
            centre=(float(rng.normal(0.0, 0.1)), float(rng.normal(0.0, 0.1))),
            ell_comps=(_clipped_ell_comp(), _clipped_ell_comp()),
            effective_radius=float(rng.uniform(0.05, 0.5)),
            sersic_index=float(np.clip(rng.normal(2.0, 0.5), 0.8, 5.0)),
            signal_to_noise_ratio=float(rng.uniform(10.0, 30.0)),
        ),
    )

    return (
        lens_galaxy_0,
        lens_galaxy_1,
        shear_galaxy,
        source_galaxy,
        [centre_0, centre_1],
    )


"""
__Sample Instances__

Within a for loop, we will now generate instances of the lens and source galaxies.
This loop will run for `total_datasets` iterations, which sets the number of lenses
that are simulated.

Each iteration of the for loop will then create a tracer and use this to simulate the
imaging dataset.
"""
total_datasets = 3

for sample_index in range(total_datasets):
    dataset_sample_path = Path(dataset_path, f"dataset_{sample_index}")

    (
        lens_galaxy_0,
        lens_galaxy_1,
        shear_galaxy,
        source_galaxy,
        main_lens_centres,
    ) = _random_multi_galaxy_lens()

    """
    __Over Sampling__

    Build the adaptive over-sampling grid for this lens, centred on both of its deflectors. This has to happen
    inside the loop because each sample lens places its galaxies differently.
    """
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=main_lens_centres,
    )

    grid_over_sampled = grid.apply_over_sampling(over_sample_size=over_sample_size)

    """
    __Ray Tracing__

    Use the sample's lens galaxies, shear and source galaxy to setup a tracer, which will generate the image for the
    simulated `Imaging` dataset.

    The steps below are expanded on in other `multi_galaxy/simulator` scripts, so check them out if anything below is
    unclear.
    """
    tracer = al.Tracer(
        galaxies=[lens_galaxy_0, lens_galaxy_1, shear_galaxy, source_galaxy]
    )

    aplt.plot_array(array=tracer.image_2d_from(grid=grid_over_sampled), title="Image")

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid_over_sampled)

    aplt.subplot_imaging_dataset(dataset=dataset)

    """
    __Output__

    Output the simulated dataset to the dataset path as .fits files.

    This uses the updated `dataset_sample_path` which outputs this sample lens to a unique folder.
    """
    aplt.fits_imaging(
        dataset=dataset,
        data_path=Path(dataset_sample_path, "data.fits"),
        psf_path=Path(dataset_sample_path, "psf.fits"),
        noise_map_path=Path(dataset_sample_path, "noise_map.fits"),
        overwrite=True,
    )

    """
    __Visualize__

    Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png
    files.
    """
    aplt.subplot_imaging_dataset(dataset=dataset)
    aplt.plot_array(array=dataset.data, title="Data")

    aplt.subplot_tracer(
        tracer=tracer,
        grid=grid_over_sampled,
        output_path=dataset_sample_path,
        output_format="png",
    )
    aplt.subplot_galaxies_images(
        tracer=tracer,
        grid=grid_over_sampled,
        output_path=dataset_sample_path,
        output_format="png",
    )

    """
    __Tracer json__

    Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and
    galaxies are safely stored and available to check how the dataset was simulated in the future.

    This can be loaded via the method `tracer = al.from_json()`.
    """
    al.output_to_json(
        obj=tracer,
        file_path=Path(dataset_sample_path, "tracer.json"),
    )

    """
    __Centre JSON Files__

    Save the centres of the two main lens galaxies, exactly as `multi_galaxy/simulator.py` does. The modeling scripts
    load this file to initialize each deflector's centre priors, so a sample lens is not modelable without it.
    """
    al.output_to_json(
        obj=al.Grid2DIrregular(main_lens_centres),
        file_path=Path(dataset_sample_path, "main_lens_centres.json"),
    )

"""
The datasets can be viewed in the folder `autolens_workspace/dataset/multi_galaxy/samples/simple/dataset_*`.

Where to go next:

- `autolens_workspace/*/multi_galaxy/simulator`: the fully documented single-dataset simulator.
- `autolens_workspace/*/multi_galaxy/modeling`: fitting a lens model to one of these datasets.
- `autolens_workspace/*/imaging/simulator_sample`: the galaxy-scale version of this script.
"""
