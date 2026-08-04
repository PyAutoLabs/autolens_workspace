"""
Simulator: Multi Galaxy Extra Galaxies
======================================

Simulates the dataset for the `multi_galaxy/features/extra_galaxies` feature: the co-dominant pair of
`multi_galaxy/simulator.py`, plus two extra galaxies near the lens which have **mass as well as light**.

The mass is the whole point of this dataset, and it is what the package's main `simple` dataset deliberately
withholds. There, the single extra galaxy is given a light profile only, so the lensed arcs stay clean and every
other example can load it as a pure two-deflector lens; the core scripts then demonstrate the
`__Extra Galaxies Noise Scaling__` step, which removes its light and never has to reason about its mass.

That is the right default, but it means the core scripts only ever show one of the two levers. Once an extra
galaxy has mass, noise-scaling its light is no longer sufficient — the mass is still bending the light of the
source, and it will still be there after every contaminated pixel has been scaled away. The companion
`modeling.py` demonstrates the other lever: carrying the extra galaxies in the model.

__The Tier Question__

`multi_galaxy/simulator.py` puts the judgement plainly: an extra galaxy is a contaminant, a main lens galaxy is
a co-dominant deflector, "and telling them apart is the first judgement you make about a multi-galaxy field — if
in doubt, the test is whether it contributes significantly to the lensing."

This dataset is built so the companion `modeling.py` can make that test concrete rather than rhetorical. The two
extra galaxies are given Einstein radii an order of magnitude below the main pair's: large enough that ignoring
their mass degrades the fit, small enough that promoting them to main lens galaxies would be modeling noise.
They sit in the band where the judgement is actually made.

__Contents__

- **Dataset Paths / Grid / PSF / Simulator:** Standard imaging simulation setup.
- **Main Lens Galaxies:** The co-dominant pair (identical to `multi_galaxy/simulator.py`).
- **Extra Galaxies:** Two perturbers with light AND mass.
- **Shear / Source:** The external shear and the lensed source.
- **Dataset:** Simulate and write the imaging dataset.
- **Mask Extra Galaxies:** Write `mask_extra_galaxies.fits` for the noise-scaling comparison.
- **Records:** The tracer, and the two centres JSONs — one per tier.

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
__Dataset Paths / Grid / PSF / Simulator__
"""
dataset_type = "multi_galaxy"
dataset_name = "extra_galaxies"

dataset_path = Path("dataset", dataset_type, dataset_name)

grid = al.Grid2D.uniform(
    shape_native=(300, 300),
    pixel_scales=0.05,
)

main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

extra_galaxy_0_centre = (2.2, 1.6)
extra_galaxy_1_centre = (-2.4, 1.9)

extra_galaxies_centres = [extra_galaxy_0_centre, extra_galaxy_1_centre]

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres + extra_galaxies_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

psf = al.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Main Lens Galaxies__

The co-dominant pair, identical to `multi_galaxy/simulator.py`: two comparable early-type deflectors whose
Einstein radii are of the same order, which is what makes this a multi-galaxy lens rather than a galaxy-scale
lens with a companion.
"""
main_lens_galaxies = []

for centre, einstein_radius, intensity in zip(
    main_lens_centres, [0.9, 0.8], [1.0, 0.8]
):
    main_lens_galaxies.append(
        al.Galaxy(
            redshift=0.5,
            bulge=al.lp.Sersic(
                centre=centre,
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
                intensity=intensity,
                effective_radius=0.6,
                sersic_index=4.0,
            ),
            mass=al.mp.Isothermal(
                centre=centre,
                einstein_radius=einstein_radius,
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            ),
        )
    )

"""
__Extra Galaxies__

Two perturbers near the co-dominant pair, each with an `ExponentialSph` light profile **and** an `IsothermalSph`
mass profile.

Their Einstein radii of 0.08" and 0.10" sit roughly an order of magnitude below the main pair's 0.9" and 0.8".
That ratio is deliberate, and it is what places them in the extra-galaxies tier rather than either neighbouring
tier:

 - **Not main lens galaxies.** A main lens galaxy is co-dominant — it changes the image configuration, and its
   mass must be free to move. At a tenth of the Einstein radius these two do not; they perturb a lens
   configuration that the pair alone already sets.
 - **Not scaling galaxies either.** The scaling tier (`features/scaling_relation`) is for populations far from
   the lens whose collective contribution is a weak correction and which are too numerous to model individually.
   These two are near the lens, few, and individually resolvable — so they get an individual free parameter
   each, not a shared relation.

The first sits at `(2.2, 1.6)`, the same position as the single massless contaminant in
`multi_galaxy/simulator.py`, so the two datasets can be compared directly: same galaxy, same place, now with
mass.

Note their redshift matches the main pair's, so this stays single-plane ray tracing. Extra galaxies at a
different redshift are supported and trigger multi-plane tracing automatically.
"""
extra_galaxies = [
    al.Galaxy(
        redshift=0.5,
        light=al.lp.ExponentialSph(
            centre=extra_galaxy_0_centre, intensity=1.0, effective_radius=0.3
        ),
        mass=al.mp.IsothermalSph(centre=extra_galaxy_0_centre, einstein_radius=0.08),
    ),
    al.Galaxy(
        redshift=0.5,
        light=al.lp.ExponentialSph(
            centre=extra_galaxy_1_centre, intensity=0.8, effective_radius=0.35
        ),
        mass=al.mp.IsothermalSph(centre=extra_galaxy_1_centre, einstein_radius=0.10),
    ),
]

"""
__Shear / Source__

The external shear is carried by a single galaxy rather than attached to every deflector, matching
`multi_galaxy/simulator.py` — giving a shear to each would be a redundant parameterization.
"""
shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.02, gamma_2=0.03),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.4,
        sersic_index=1.5,
    ),
)

"""
__Dataset__

Use all galaxies to setup a tracer, which generates the image for the simulated `Imaging` dataset. Because every
deflector is at the same redshift, the main pair's, the extra galaxies' and the shear's deflection fields simply
add.
"""
tracer = al.Tracer(
    galaxies=main_lens_galaxies + extra_galaxies + [shear_galaxy, source_galaxy]
)

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Mask Extra Galaxies__

Write a `mask_extra_galaxies.fits` covering both extra galaxies, so `modeling.py` can run the noise-scaling
comparison — fitting the same data with their light scaled away but their mass unaccounted for — and show
directly why that is not sufficient once an extra galaxy has mass.

Each circle is sized to ~3x the galaxy's `effective_radius`. The geometry derives from the same centres and
radii defined above, so it stays in sync with any future tweak. `Mask2D.circular` honours
`PYAUTO_SMALL_DATASETS=1`, so the mask shrinks with the image and never goes out of bounds.
"""
extra_galaxies_mask = np.zeros(dataset.shape_native, dtype=bool)

for centre, radius in [
    (extra_galaxy_0_centre, 3.0 * 0.3),
    (extra_galaxy_1_centre, 3.0 * 0.35),
]:
    circle = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        centre=centre,
        radius=radius,
        invert=True,  # True inside the circle (i.e. the scaled region)
    )
    extra_galaxies_mask = np.logical_or(extra_galaxies_mask, circle.native)

mask_extra_galaxies = al.Mask2D(
    mask=extra_galaxies_mask,
    pixel_scales=dataset.pixel_scales,
)

aplt.fits_array(
    array=mask_extra_galaxies,
    file_path=dataset_path / "mask_extra_galaxies.fits",
    overwrite=True,
)

"""
__Records__

Save the tracer and both centres files.

The split into two files is the tier assignment made concrete: `main_lens_centres.json` initializes the free
centre priors of the co-dominant pair, while `extra_galaxies_centres.json` **fixes** the perturbers' profile
centres. Which file a galaxy lands in is the decision `modeling.py` discusses.
"""
al.output_to_json(
    obj=tracer,
    file_path=dataset_path / "tracer.json",
)

al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=dataset_path / "main_lens_centres.json",
)

al.output_to_json(
    obj=al.Grid2DIrregular(extra_galaxies_centres),
    file_path=dataset_path / "extra_galaxies_centres.json",
)
