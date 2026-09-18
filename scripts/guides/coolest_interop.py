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
- **Lens Model:** A simple lens model (power-law lens + an external-shear `MassField`, Sersic source) to export.
- **Export:** Writing the model to a COOLEST `.json` template.
- **Import:** Reading a COOLEST template back as a `Tracer`.
- **Round Trip:** Verifying the exported and imported models are numerically identical.
- **Mass Fields:** How a PyAutoLens `MassField` maps 1:1 onto a COOLEST `MassField` entity.
- **NFW Profiles:** The critical surface density COOLEST's NFW normalization requires.
- **Unsupported Profiles:** Exporting a model containing components COOLEST cannot represent.

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

import json

from os import path

import autolens as al

"""
__Lens Model__

A simple lens model: a power-law mass profile lensing a Sersic source, with an external shear beside them.

The shear describes the tidal field of everything *outside* the modelled system, so it is not a property of the
lens galaxy: it is held in an `al.MassField` (a redshift plus a bag of mass profiles, carrying no light) and
passed to the `Tracer` via its `fields` argument. This mirrors COOLEST, whose standard also treats external
fields as their own entity rather than as part of a galaxy.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
        einstein_radius=1.6,
        slope=2.1,
    ),
)

field = al.MassField(
    redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.02, gamma_2=-0.03)
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

tracer = al.Tracer(galaxies=[lens, source], fields=[field])

"""
__Export__

`to_coolest` writes the model to a COOLEST `.json` template file and returns the written path.

Each galaxy becomes a COOLEST `Galaxy` lensing entity; each `al.MassField` becomes a COOLEST `MassField`
entity, a 1:1 mapping of the two standards' external-field containers. (An `ExternalShear` or `MassSheet` still
attached to a galaxy is peeled off into a COOLEST `MassField` too, so the older galaxy-attached form exports
identically.) The model cosmology's H0 and Om0 are stored in the template.

COOLEST's plotting API evaluates a model on the pixel grid of the observation, so the template's `observation`
block must record the field of view and number of pixels of the data. These are written from `shape_native`
(the data's `(y_pixels, x_pixels)` shape) and `pixel_size` (its pixel scale in arcseconds); passing
`dataset=dataset` instead takes both from an `al.Imaging` object directly. If neither is given the observation
grid is written as zeros, a warning is raised and COOLEST cannot plot the template.
"""
file_path = al.interop.coolest.to_coolest(
    galaxies=tracer,
    file_path=path.join("output", "coolest_template"),
    shape_native=(100, 100),
    pixel_size=0.1,
)

print(f"COOLEST template written to: {file_path}")

with open(file_path) as f:
    print(f"Observation pixel grid: {json.load(f)['observation']['pixels']}")

"""
__Import__

`from_coolest` reads a COOLEST template — one written by **PyAutoLens** or by any other code — and returns a
`Tracer` built from its profiles, with all parameters converted back to PyAutoLens conventions.

Each COOLEST `Galaxy` entity comes back in `tracer.galaxies` and each COOLEST `MassField` entity comes back in
`tracer.fields` as an `al.MassField`. The external shear therefore returns as a field, not bolted onto the lens
galaxy — the round trip preserves *which entity* holds the shear, not only its parameter values.

The COOLEST standard does not store the *names* PyAutoLens gives a profile, so a returned field's components are
named `mass_0`, `mass_1`, ... rather than `shear` or `mass_sheet`. Select them by class instead:
"""
tracer_via_coolest = al.interop.coolest.from_coolest(file_path=file_path)

print(tracer_via_coolest.galaxies)
print(tracer_via_coolest.fields)

shear_via_coolest = tracer_via_coolest.fields[0].cls_list_from(cls=al.mp.ExternalShear)[
    0
]

print(f"Shear returned as a field: {shear_via_coolest}")

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
__Mass Fields__

The mapping is 1:1 in both directions: a `Tracer` built with `fields=[...]` exports one COOLEST `MassField`
entity per `al.MassField`, and reading that template back gives the same fields again.

Below a mass sheet is added beside the shear — shear + sheet at one redshift are *one* field, exactly as a
bulge + disk are one galaxy — and the export/import round trip is repeated to show both profiles returning in
the single field entity.
"""
field_with_sheet = al.MassField(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.02, gamma_2=-0.03),
    mass_sheet=al.mp.MassSheet(centre=(0.0, 0.0), kappa=0.05),
)

tracer_with_sheet = al.Tracer(galaxies=[lens, source], fields=[field_with_sheet])

file_path_field = al.interop.coolest.to_coolest(
    galaxies=tracer_with_sheet,
    file_path=path.join("output", "coolest_template_field"),
    shape_native=(100, 100),
    pixel_size=0.1,
)

tracer_field_via_coolest = al.interop.coolest.from_coolest(file_path=file_path_field)

field_via_coolest = tracer_field_via_coolest.fields[0]

print(f"Fields returned: {len(tracer_field_via_coolest.fields)}")
print(f"Shear: {field_via_coolest.cls_list_from(cls=al.mp.ExternalShear)[0]}")
print(f"Mass sheet: {field_via_coolest.cls_list_from(cls=al.mp.MassSheet)[0]}")

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

__Unsupported Profiles__

Many lens models contain components the COOLEST standard has no analytic representation for, for example a
multi-Gaussian expansion (`Basis` of Gaussians) lens light or a pixelized source (`Pixelization`). Exporting
such a model raises an error by default, which would make COOLEST unusable for these models.

Passing `on_unsupported="skip"` instead exports everything COOLEST can represent and records what it left out
in the template's `meta` block, under `skipped_profiles`. The template therefore still describes the mass model
and any supported light profiles, whilst stating explicitly which components of the model it does not contain.
"""
lens_mge = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_basis.Basis(
        profile_list=[
            al.lp.GaussianSph(intensity=1.0, sigma=0.1),
            al.lp.GaussianSph(intensity=0.5, sigma=0.3),
        ]
    ),
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6),
)

source_pixelized = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        mesh=al.mesh.Delaunay(pixels=500),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

file_path_skip = al.interop.coolest.to_coolest(
    galaxies=[lens_mge, source_pixelized],
    file_path=path.join("output", "coolest_template_skip"),
    shape_native=(100, 100),
    pixel_size=0.1,
    on_unsupported="skip",
)

with open(file_path_skip) as f:
    print(f"Skipped profiles: {json.load(f)['meta']['skipped_profiles']}")

"""
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
