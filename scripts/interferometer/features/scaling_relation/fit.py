"""
Features (Interferometer): Scaling Relation Fit
==============================================

Fits the interferometer `scaling_relation` dataset at the simulator's truth values, so the tied tier can be inspected
without a non-linear search in the way.

The point is the **deflection sum**: a scaling relation does not change how ray-tracing works, only where each
member's `einstein_radius` comes from. Below, the relation is evaluated explicitly, each companion's deflection field
is computed on its own, and the sum is checked against what the `Tracer` produces internally — after which the whole
lot is Fourier transformed to the uv-plane, exactly as an ordinary interferometer fit would be.

__Prerequisites__

 - `autolens_workspace/scripts/interferometer/fit.py` — the standard `FitInterferometer` walkthrough.
 - `autolens_workspace/scripts/interferometer/features/scaling_relation/modeling.py` — the search-based version.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/fit.py` — the CCD-imaging equivalent, which also
   models companion light.

__Mass Only__

Foreground light is rarely detected at interferometer wavelengths, so lens and companions alike are mass-only here.
The luminosities driving the relation are therefore external inputs from ancillary optical/NIR imaging — see
`modeling.py`. All profiles are **untruncated**; truncated `dPIEMass` members belong to the group and cluster
workflows.

__Contents__

- **Real Space Mask / Dataset:** Standard interferometer set up (auto-simulating if absent).
- **Centres + Luminosities:** From ancillary imaging, not from this dataset.
- **The Relation:** One function, evaluated per companion.
- **Galaxies:** Anchor, scaling tier, source — all at simulator truth.
- **Tracer & Fit:** Build the `Tracer` and fit the visibilities.
- **The Relation, Evaluated:** Each tied Einstein radius as the arithmetic that produced it.
- **Deflection Sum:** Per-galaxy deflections, summed by hand and checked against the tracer.
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

aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Centres + Luminosities__

Both come from ancillary imaging: a visibility dataset contains neither the companions' positions nor their light.
Given as explicit lists here; the CSV equivalent is at the end.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

luminosity_anchor = 31.0962

scaling_galaxies_luminosities = [1.4939, 1.0865, 0.7696, 0.4980, 0.2716]

"""
__The Relation__

The anchor's Einstein radius is a fixed number here (simulator truth) rather than a free parameter, so the relation
evaluates to a plain float per companion. In `modeling.py` the identical expression multiplies the model's free
`einstein_radius` instead — same algebra, different object.
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

All at the simulator's truth values: the anchor's `Isothermal`, five `IsothermalSph` companions with radii from the
relation, and the source's `SersicCore`.
"""
main_lens = al.Galaxy(
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

"""
__Tracer & Fit__

`FitInterferometer` performs the Fourier transform as part of fitting, so the tier enters exactly where it does for
CCD imaging — in the real-space deflection field, before the transform. Nothing about the uv-plane comparison changes.
"""
tracer = al.Tracer(galaxies=[main_lens] + scaling_galaxies + [source])

fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

aplt.subplot_fit_interferometer(fit=fit)
aplt.subplot_fit_dirty_images(fit=fit)

print(f"Log likelihood of the truth fit: {fit.log_likelihood}")

"""
__The Relation, Evaluated__
"""
print(
    f"\nAnchor: einstein_radius = {einstein_radius_anchor:.4f}, L = {luminosity_anchor:.4f}"
)

for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    centre_str = f"({float(centre[0]):5.2f}, {float(centre[1]):5.2f})"
    print(
        f"  scaling galaxy @ {centre_str}: "
        f"{einstein_radius_anchor:.3f} * ({luminosity:.4f} / {luminosity_anchor:.4f}) ** {scaling_exponent} "
        f"= {einstein_radius_from(luminosity):.4f}"
    )

"""
__Deflection Sum__

The real-space total deflection is the sum of every mass profile's contribution, computed here explicitly and checked
against the tracer.
"""
grid = dataset.grid

alpha_anchor = main_lens.mass.deflections_yx_2d_from(grid=grid)
alpha_scaling = [g.mass.deflections_yx_2d_from(grid=grid) for g in scaling_galaxies]

print(f"\nalpha_anchor             (first coord): {alpha_anchor[0]}")
print(f"alpha_scaling (tier sum) (first coord): {sum(alpha_scaling)[0]}")

alpha_total_summed = alpha_anchor + sum(alpha_scaling)

traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
alpha_total_tracer = grid - traced_grids[1]

print(f"\nalpha_total (summed by hand, first 3): {alpha_total_summed[:3]}")
print(f"alpha_total (from tracer,    first 3): {alpha_total_tracer[:3]}")

assert np.allclose(np.asarray(alpha_total_summed), np.asarray(alpha_total_tracer))

"""
An isothermal's deflection magnitude is constant and equal to its Einstein radius, so these companions deflect by
0.15-0.35" everywhere — not negligible next to the anchor's 1.6". A nearly uniform deflection is degenerate with the
source position, though, so what matters is the *differential* deflection across the lensed emission: a shear of
roughly `theta_E / 2d`, a few percent for these members.

For an interferometer that differential is measured in the uv-plane rather than from an image, which if anything makes
it easier to detect — visibilities constrain the arcs' shape directly, without the PSF blurring that limits CCD
imaging.

__CSV Interface__

For larger populations, `al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV and returns a `GalaxyTable` with
`.centres`, `.luminosities` and `.redshifts`:

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

Since both quantities come from the same external photometric catalogue for this regime, one file for both is the
natural representation.
"""
scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")

print(f"\nTier luminosities from CSV: {list(scaling_table.luminosities)}")

"""
__Wrap Up__

Ray-tracing is unchanged by the relation — the real-space deflection field is a plain sum over mass profiles, as the
assertion confirms, and the Fourier transform that follows is untouched. The relation only fixes what each member's
`einstein_radius` is set to.

Next: `modeling.py` fits this with a search, and `slam.py` runs the same relation through a SLaM pipeline.
"""
