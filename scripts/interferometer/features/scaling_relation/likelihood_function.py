"""
__Log Likelihood Function: Scaling Relation (Interferometer)__

Describes the one step of the interferometer `log_likelihood` computation that a scaling relation changes: how the
real-space deflection field is composed when most of the foreground galaxies' Einstein radii are derived from the main
lens's rather than sampled.

This script does NOT repeat the steps shared with the standard interferometer likelihood (real-space grid, lens light,
the Fourier transform, chi-squared, noise normalisation). Those are in the prerequisite and are entirely unaffected by
the relation.

__Prerequisites__

 - `autolens_workspace/scripts/interferometer/likelihood_function.py` — the canonical uv-plane log-likelihood
   walkthrough, including the Fourier transform step.
 - `autolens_workspace/scripts/interferometer/features/scaling_relation/modeling.py` — the search-based version of
   the composition demonstrated here.

__What Changes For A Scaling Relation__

The interferometer likelihood evaluates a model image in real space, then Fourier transforms it to the uv-plane to
compare with the visibilities. The relation touches only the first half of that, and only the deflection field:

    alpha(theta) = alpha_anchor(theta) + sum_j alpha_scaling_j(theta)

    where einstein_radius_j = einstein_radius_anchor * (L_j / L_anchor) ** 0.5

The transform, the chi-squared against the visibilities and the noise normalisation are untouched. So the tier costs
likelihood *time* — one extra deflection evaluation per member per call — and no parameters.

__Where The Luminosities Come From__

Not from this dataset. Foreground light is rarely detected at interferometer wavelengths, so there is no foreground
emission here to fit and nothing to integrate; the `L_j` are measurements from ancillary optical/NIR imaging. This is
the substantive difference from the CCD-imaging version of this feature, whose `slam.py` measures its own.

__Contents__

- **Real Space Mask / Dataset:** Standard set up (auto-simulating if absent).
- **Centres + Luminosities:** External inputs.
- **The Relation:** The per-companion Einstein radius.
- **Galaxies:** Mass-only lens plane plus an analytic source.
- **Deflection Field:** Each tier member's contribution.
- **Manual Ray-Tracing:** Hand-compute the source-plane grid and confirm it matches `Tracer`.
- **Real-Space Image:** The image that is about to be transformed.
- **Likelihood:** `FitInterferometer.log_likelihood`.
- **CSV Interface.**
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Real Space Mask / Dataset__
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

dataset_name = "scaling_relation"
dataset_path = Path("dataset") / "interferometer" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/interferometer/features/scaling_relation/simulator.py",
        ],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
__Centres + Luminosities__

Both external. Explicit lists here; the CSV equivalent is at the end.
"""
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

luminosity_anchor = 31.0962

scaling_galaxies_luminosities = [1.4939, 1.0865, 0.7696, 0.4980, 0.2716]

"""
__The Relation__
"""
einstein_radius_anchor = 1.6
scaling_exponent = 0.5


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the main lens.
    """
    return einstein_radius_anchor * (luminosity / luminosity_anchor) ** scaling_exponent


"""
__Galaxies__

Mass-only lens plane (as every interferometer example is) plus an analytic source. All profiles **untruncated** —
truncation encodes tidal stripping by a host halo, which a galaxy-scale lens does not have.
"""
anchor = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=einstein_radius_anchor,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

scaling_galaxies = [
    al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(
            centre=tuple(centre), einstein_radius=einstein_radius_from(luminosity)
        ),
    )
    for centre, luminosity in zip(
        scaling_galaxies_centres, scaling_galaxies_luminosities
    )
]

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer(galaxies=[anchor] + scaling_galaxies + [source])

"""
__Deflection Field__

Every lens-plane mass profile contributes; recomputing the sum by hand makes the tier explicit.
"""
masked_grid = dataset.grid

alpha_anchor = anchor.mass.deflections_yx_2d_from(grid=masked_grid)
alpha_scaling = [
    galaxy.mass.deflections_yx_2d_from(grid=masked_grid) for galaxy in scaling_galaxies
]

alpha_total = alpha_anchor + sum(alpha_scaling)

print(f"alpha_anchor             (first coord): {alpha_anchor[0]}")
print(f"alpha_scaling (tier sum) (first coord): {sum(alpha_scaling)[0]}")
print(f"alpha_total   (all)      (first coord): {alpha_total[0]}")

for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    centre_str = f"({float(centre[0]):5.2f}, {float(centre[1]):5.2f})"
    print(
        f"  scaling galaxy @ {centre_str}: einstein_radius = "
        f"{einstein_radius_anchor:.3f} * ({luminosity:.4f} / {luminosity_anchor:.4f}) ** {scaling_exponent} "
        f"= {einstein_radius_from(luminosity):.4f}"
    )

"""
__Manual Ray-Tracing__

The source-plane grid is the image-plane grid minus the total deflection. The relation changes the values fed into
that sum, not the sum itself.
"""
grid_source_manual = masked_grid - alpha_total

traced_grid_list = tracer.traced_grid_2d_list_from(grid=masked_grid)
grid_source_tracer = traced_grid_list[1]

print(f"\nsource-plane grid (first coord, manual): {grid_source_manual[0]}")
print(f"source-plane grid (first coord, tracer): {grid_source_tracer[0]}")

assert np.allclose(np.asarray(grid_source_manual), np.asarray(grid_source_tracer))

"""
__Real-Space Image__

The image the transform will act on. Everything the tier does to the likelihood has already happened by this point —
it perturbed the deflection field, which moved the lensed emission.
"""
model_image = tracer.image_2d_from(grid=masked_grid)

aplt.plot_array(array=model_image, title="Real-space model image, before the transform")

"""
__Likelihood__

The image above is Fourier transformed to the uv-plane and compared with the visibilities via the standard
interferometer chi-squared, documented in the prerequisite. `FitInterferometer` performs the transform as part of
fitting.
"""
fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

aplt.subplot_fit_interferometer(fit=fit)

print(f"\nLog likelihood: {fit.log_likelihood}")

"""
__CSV Interface__

`al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV and returns a `GalaxyTable` with `.centres`,
`.luminosities` and `.redshifts`:

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

Nothing downstream changes — the likelihood never sees where the numbers came from.
"""
scaling_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

print(f"Tier luminosities from CSV: {list(scaling_table.luminosities)}")

"""
__Wrap Up__

The relation changes one thing: some companions' `einstein_radius` values are set by the anchor's value and an
externally measured luminosity rather than sampled. The deflection sum, ray-tracing, Fourier transform, chi-squared
and noise normalisation are all shared with the standard interferometer likelihood.

The cost is one extra deflection evaluation per member per likelihood call — and zero extra parameters.
"""
