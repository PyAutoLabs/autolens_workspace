"""
COOLEST
=======

COOLEST (COde-independent Organized LEns STandard, https://github.com/aymgal/COOLEST) is a standard for storing
and exchanging strong lens models between different lens modeling software, for example lenstronomy, herculens
and GLEE.

**PyAutoLens** can export a lens model's analytic profile parameters to a COOLEST JSON template file and import
a template (including one produced by another code) back as a `Tracer`.

This requires the optional `coolest` package:

`pip install autolens[coolest]`

__Contents__

- **Conventions:** How PyAutoLens conventions map to COOLEST conventions.
- **Lens Model:** A simple lens model (power-law + shear lens, Sersic source) to export.
- **Export:** Writing the model to a COOLEST `.json` template.
- **Import:** Reading a COOLEST template back as a `Tracer`.
- **Round Trip:** Verifying the exported and imported models are numerically identical.
- **NFW Profiles:** The critical surface density COOLEST's NFW normalization requires.

__Conventions__

The conversion handles the following differences automatically, so you do not need to apply any factors yourself:

- Position angles: PyAutoLens measures counter-clockwise from the positive x-axis; COOLEST measures
  counter-clockwise from the positive y-axis ("East-of-North") in the interval (-90, +90] degrees.

- Ellipticity: PyAutoLens profiles use elliptical components `ell_comps`; COOLEST uses the axis ratio `q` and
  position angle `phi`.

- Radii: COOLEST characteristic radii (e.g. the Einstein radius `theta_E`) use the intermediate axis
  r = sqrt(a * b) of the elliptical contours. For a power-law profile this means
  `theta_E = sqrt(q) * (2 / (1 + q))^(1 / (gamma - 1)) * einstein_radius`, which for an isothermal profile
  reduces to the familiar `2 sqrt(q) / (1 + q)` factor. The Sersic `effective_radius` is already an
  intermediate-axis radius and converts with no factor.

The exported `theta_E` is the mass profile's parameter in the COOLEST convention. It is not a curve-based
Einstein radius (e.g. the effective radius of the area within the tangential critical curve, used by the
Euclid DR1 catalogue), which depends on all profiles in the model including external shear.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from os import path

import autolens as al

"""
__Lens Model__

A simple lens model: a power-law mass profile with external shear lensing a Sersic source.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
        einstein_radius=1.6,
        slope=2.1,
    ),
    shear=al.mp.ExternalShear(gamma_1=0.02, gamma_2=-0.03),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.5,
        sersic_index=2.0,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

"""
__Export__

`to_coolest` writes the model to a COOLEST `.json` template file and returns the written path.

Each galaxy becomes a COOLEST `Galaxy` lensing entity; `ExternalShear` and `MassSheet` profiles are written as
COOLEST `MassField` entities, as the standard requires. The model cosmology's H0 and Om0 are stored in the
template.
"""
file_path = al.interop.coolest.to_coolest(
    galaxies=tracer, file_path=path.join("output", "coolest_template")
)

print(f"COOLEST template written to: {file_path}")

"""
__Import__

`from_coolest` reads a COOLEST template — one written by **PyAutoLens** or by any other code — and returns a
`Tracer` built from its profiles, with all parameters converted back to PyAutoLens conventions.
"""
tracer_via_coolest = al.interop.coolest.from_coolest(file_path=file_path)

print(tracer_via_coolest.galaxies)

"""
__Round Trip__

The imported model is numerically identical to the exported one, which we verify by comparing deflection angles
on a grid.
"""
grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.1)

deflections = tracer.deflections_yx_2d_from(grid=grid)
deflections_via_coolest = tracer_via_coolest.deflections_yx_2d_from(grid=grid)

print(
    "Max deflection difference: "
    f"{abs(deflections.array - deflections_via_coolest.array).max()}"
)

"""
__NFW Profiles__

COOLEST parameterises the NFW profile by a physical characteristic density `rho_c`, whereas the PyAutoLens
`NFW` uses the dimensionless `kappa_s`. The conversion therefore uses the critical surface mass density between
the galaxy's redshift and the model's highest redshift, computed automatically from the model's cosmology
(pass `cosmology=` to `to_coolest` / `from_coolest` to control it; a template only stores H0 and Om0, so supply
the same cosmology to both directions for an exact round trip).

__Supported Profiles__

Light: `Sersic` / `SersicSph`. Mass: `Isothermal` / `IsothermalSph` (SIE), `PowerLaw` / `PowerLawSph` (PEMD),
`NFW` / `NFWSph`, `ExternalShear` and `MassSheet` (ConvergenceSheet). Converting an unsupported profile raises
an error naming the profile.

Fin.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
