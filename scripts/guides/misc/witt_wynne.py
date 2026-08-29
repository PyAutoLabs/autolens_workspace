"""
Guide: Witt-Wynne (SIEP) Projection and the isit4or2or1 Solver
==============================================================

The singular isothermal elliptical *potential* (SIEP) is the one strong-lens model whose lens
equation reduces to a quartic with a closed-form solution. Given a source position it returns, in
microseconds and without any iteration, the number of images (4, 2 or 1), their positions, their
signed magnifications and their time lags. That speed is what makes it usable inside a transient
broker: when a supernova alert lands near a known quad, the question "is this a fourth image or a
foreground star?" has to be answered before the object fades.

Paul Schechter's `isit4or2or1` implements exactly that check (Schechter, Lu & Hernandez 2026). This
guide does three things:

1. Ports the shipped C++ solver to pure numpy, so no build step is needed, and reproduces the
   SN 2025wny example distributed with it to the precision of its printed output.
2. Projects a **PyAutoLens** ``Tracer`` (an isothermal or power-law mass profile plus external
   shear, and any source) onto SIEP parameters. The projection throws away everything the SIEP
   cannot represent: source shape, non-isothermality, secondary perturbers, and the components of
   ellipticity and shear perpendicular to the direction of their sum.
3. Measures how much that projection costs, by comparing the SIEP prediction against the full
   ``PointSolver`` and ``time_delays_from`` on simulated SIE + shear quads.

Two projections are implemented. The **caustic-matched** one fits the SIEP astroid to the tracer's
own tangential caustic. The **vector-sum** one follows Schechter's literal prescription: add the
ellipticity and shear as vectors in the 2-theta plane and keep the magnitude and angle of the sum.
The validation section below reports how they compare.

__Attribution and References__

The solver ported here is `isit4or2or1` v1.0 (Schechter, Lu & Hernandez), archived at Zenodo,
DOI 10.5281/zenodo.20086659, and released under CC-BY-4.0. This port is a derivative work: it
follows the structure and the conventions of `SIEP_CLI.v1.0.cpp` line for line, and the same
CC-BY-4.0 attribution applies to it. Please cite the Zenodo record and the paper below if you use
it.

- Schechter, Lu & Hernandez 2026, arXiv:2605.11090 -- SN 2025wny and the LSST alert protocol.
- Witt 1996, ApJ 472, L1 -- the hyperbola on which the image positions lie.
- Wynne & Schechter 2018, arXiv:1808.06151 -- the ellipse that intersects it.
- Schechter & Wynne 2019, arXiv:1901.08517 -- the resulting quartic.
- Falor & Schechter 2022, arXiv:2205.06269 -- the asymptotically circular lens equation (ACLE) and
  the 4/2/1 root count used here.
- Schechter 2026, Galaxies 14, 20, arXiv:2604.11908 -- SIEP with parallel shear.

__Conventions__

The C++ follows Keeton's `gravlens` conventions, and this port reproduces them exactly:

- Coordinates are ``(x, y)`` in arcseconds, position angle ``phi`` is degrees East of North. In
  the `gravlens` frame the x-axis points West and the y-axis North, which is the orientation a
  North-up / East-left image has when read into a **PyAutoLens** ``Grid2D``. **PyAutoLens** grids
  are ordered ``(y, x)``, so the only change of variables needed is the swap of the tuple order.
- A position angle ``phi`` maps to a **PyAutoLens** angle (degrees counter-clockwise from the
  positive x-axis) as ``angle = phi - 90``, and back as ``phi = angle + 90``. Getting this
  backwards reflects every predicted image about the potential's minor axis, which is why the
  validation section below checks positions and not just image counts.
- ``e`` is the ellipticity of the *potential*, ``e = 1 - q_psi``, not of the density. It is not the
  same quantity as **PyAutoLens**'s ``ell_comps`` magnitude ``(1 - q)/(1 + q)``, which describes
  the density.
- Distances are angular diameter distances in ``h^-1 Mpc``, with ``D_H = 3000 h^-1 Mpc``.
- The original hardcodes ``h = 0.7`` in its time constant while taking distances in ``h^-1 Mpc``,
  so its lags are on an ``h = 0.7`` scale whatever cosmology produced the distances. The port
  exposes ``h`` as an argument defaulting to ``0.7`` so the regression below is exact, and passes
  the tracer's own ``h`` when comparing against ``tracer.time_delays_from``.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

import autolens as al
import autogalaxy as ag

"""
__The Quartic__

``solve_quartic`` returns the four (generally complex) roots of ``a x^4 + b x^3 + c x^2 + d x + e``
using the closed-form general quartic formula. It is a direct transcription of the C++ so that the
roots come back in the same order, which is what fixes the A/B/C/D image labelling of the reference
output. ``numpy.roots`` would give the same set in a different order.
"""


def solve_quartic(a: float, b: float, c: float, d: float, e: float) -> List[complex]:
    a, b, c, d, e = (complex(v) for v in (a, b, c, d, e))

    p = (8.0 * a * c - 3.0 * b * b) / (8.0 * a * a)
    q = (b**3 - 4.0 * a * b * c + 8.0 * a * a * d) / (8.0 * a**3)

    delta_0 = c * c - 3.0 * b * d + 12.0 * a * e
    delta_1 = (
        2.0 * c**3
        - 9.0 * b * c * d
        + 27.0 * b * b * e
        + 27.0 * a * d * d
        - 72.0 * a * c * e
    )

    big_q = (0.5 * (delta_1 + np.sqrt(delta_1 * delta_1 - 4.0 * delta_0**3))) ** (
        1.0 / 3.0
    )
    s = 0.5 * np.sqrt(-2.0 / 3.0 * p + (big_q + delta_0 / big_q) / (3.0 * a))

    k_1 = np.sqrt(-4.0 * s * s - 2.0 * p + q / s)
    k_2 = np.sqrt(-4.0 * s * s - 2.0 * p - q / s)

    return [
        -0.25 * b / a - s + 0.5 * k_1,
        -0.25 * b / a - s - 0.5 * k_1,
        -0.25 * b / a + s + 0.5 * k_2,
        -0.25 * b / a + s - 0.5 * k_2,
    ]


"""
__The Asymptotically Circular Lens Equation__

Witt's hyperbola ``(x - p)(y - q) = p q`` and the unit circle ``x^2 + y^2 = 1`` intersect in the
image positions of the scaled problem. Eliminating ``y`` gives

    x^4 - 2 p x^3 + (p^2 + q^2 - 1) x^2 + 2 p x - p^2 = 0,   y = q x / (x - p)

and the number of *real* roots is the 4/2/1 verdict (Falor & Schechter 2022, section 2.3). A root
counts as real when its imaginary part is below ``threshold``; if a system you expect to be a quad
returns two images, raise it.

The ACLE degenerates when ``q = 0``, i.e. when the source lies exactly on the potential's major
axis: the quartic factorises as ``(x^2 - 1)(x - p)^2`` and the double root at ``x = p`` makes
``y = q x / (x - p)`` indeterminate. The original has the same behaviour and no guard, so neither
does this port -- perturb such a source off the axis rather than trusting the output.
"""


def find_intersections(
    p: float, q: float, threshold: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    roots = solve_quartic(1.0, -2.0 * p, p * p + q * q - 1.0, 2.0 * p, -p * p)

    x_list, y_list = [], []

    for root in roots:
        if abs(root.imag) < threshold:
            x = root.real
            x_list.append(x)
            y_list.append(q * x / (x - p))

    return np.asarray(x_list), np.asarray(y_list)


"""
__Registered Coordinates__

"Registered" coordinates are centred on the potential with the x-axis along its major axis. The
rotation is a proper one (determinant +1), so handedness is preserved and the inverse is the
transpose.
"""


def to_registered(
    x: np.ndarray, y: np.ndarray, pa_deg: float, centre: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.radians(pa_deg)
    sin, cos = np.sin(phi), np.cos(phi)
    return (
        sin * (x - centre[0]) - cos * (y - centre[1]),
        cos * (x - centre[0]) + sin * (y - centre[1]),
    )


def to_detector(
    x: np.ndarray, y: np.ndarray, pa_deg: float, centre: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.radians(pa_deg)
    sin, cos = np.sin(phi), np.cos(phi)
    return sin * x + cos * y + centre[0], -cos * x + sin * y + centre[1]


"""
__Images From a Source__

The source position is scaled into the ACLE parameters

    p = x'_s (1 - e)^2 / (b e (2 - e)),    q = -y'_s (1 - e) / (b e (2 - e))

solved, and mapped back with ``x = b x_hat + x'_s``, ``y = b y_hat / (1 - e) + y'_s``. The signed
magnification follows from inverting ``mu^-1 = I - H(psi)`` (Falor & Schechter 2022, equation F1),
and ``angle`` is the position angle of the eigenvector of the smaller eigenvalue, i.e. the
direction along which the image is stretched.
"""


def images_from_source(
    x_s: float,
    y_s: float,
    e: float,
    b: float,
    pa_deg: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    threshold: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = np.radians(pa_deg)

    x_reg_s, y_reg_s = to_registered(x_s, y_s, pa_deg=pa_deg, centre=centre)

    p = x_reg_s * (1.0 - e) ** 2 / (b * (2.0 - e) * e)
    q = -y_reg_s * (1.0 - e) / (b * (2.0 - e) * e)

    x_hat, y_hat = find_intersections(p=p, q=q, threshold=threshold)

    x = b * x_hat + x_reg_s
    y = b * y_hat / (1.0 - e) + y_reg_s

    x_image, y_image = to_detector(x, y, pa_deg=pa_deg, centre=centre)

    m = (1.0 - e) ** 2 * (x * x + y * y / (1.0 - e) ** 2) ** 1.5
    magnification = m / (m - b * (x * x + y * y))

    numerator = 2.0 * b * (1.0 - e) ** 2 * x * y
    denominator = b * (1.0 - e) ** 2 * (x - y) * (x + y) - np.sqrt(
        b * b * (1.0 - e) ** 4 * (x * x + y * y) ** 2
    )
    angle = np.arctan(numerator / denominator) + phi - np.pi / 2.0
    angle = angle - np.floor((angle + np.pi / 2.0) / np.pi) * np.pi

    return x_image, y_image, magnification, np.degrees(angle)


"""
__Source From an Image__

The inverse problem: given one image position, where is the source? Intersecting Wynne's ellipse
with Witt's hyperbola at the known image gives a closed form, so a single detected image of a
candidate transient fixes the source and hence the other three images.
"""


def source_from_image(
    x_i: float,
    y_i: float,
    e: float,
    b: float,
    pa_deg: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    x_reg, y_reg = to_registered(x_i, y_i, pa_deg=pa_deg, centre=centre)

    x_hat = x_reg / b
    y_hat = y_reg / b

    y_s = y_hat - y_hat / (1.0 - e) * np.sqrt(
        1.0 / ((1.0 - e) ** 2 * x_hat * x_hat + y_hat * y_hat)
    )
    x_s = x_hat * e * (2.0 - e) + (1.0 - e) ** 2 * x_hat * y_s / y_hat

    return to_detector(x_s * b, y_s * b, pa_deg=pa_deg, centre=centre)


"""
__Time Lags__

The Fermat potential of the SIEP gives the arrival-time surface directly. Delays are referred to
the leading image, so the earliest arrival has a lag of zero and every other lag is positive and in
days.

``d_ol`` and ``d_ls`` are angular diameter distances in ``h^-1 Mpc``; the effective distance uses
comoving distances ``chi = D (1 + z)``. The Hubble time enters as ``9.78 / h`` Gyr, and the
original fixes ``h = 0.7`` here regardless of the distances it was given, so ``h`` defaults to
``0.7`` to reproduce it.
"""

D_H = 3000.0  # Hubble distance in h^-1 Mpc.


def time_lags_days(
    x_image: np.ndarray,
    y_image: np.ndarray,
    x_s: float,
    y_s: float,
    e: float,
    b: float,
    pa_deg: float,
    d_ol: float,
    d_ls: float,
    z_lens: float,
    z_source: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    h: float = 0.7,
) -> np.ndarray:
    time_constant = 9.78e9 * (1.0 / h) * 365.0  # 1/H0 in days.

    chi_ol = (d_ol / D_H) * (1.0 + z_lens)
    chi_ls = (d_ls / D_H) * (1.0 + z_source)
    d_eff = chi_ol * (chi_ol + chi_ls) / chi_ls

    arcsec_to_radians = np.pi / (3600.0 * 180.0)

    x_reg_s, y_reg_s = to_registered(x_s, y_s, pa_deg=pa_deg, centre=centre)
    x_reg_s, y_reg_s = x_reg_s * arcsec_to_radians, y_reg_s * arcsec_to_radians

    x_reg, y_reg = to_registered(x_image, y_image, pa_deg=pa_deg, centre=centre)
    x_reg, y_reg = x_reg * arcsec_to_radians, y_reg * arcsec_to_radians

    b_radians = b * arcsec_to_radians

    shapiro = -d_eff * b_radians * np.sqrt(x_reg**2 + y_reg**2 / (1.0 - e) ** 2)
    geometric = 0.5 * d_eff * ((x_reg - x_reg_s) ** 2 + (y_reg - y_reg_s) ** 2)

    delays = (shapiro + geometric) * time_constant

    return delays - delays.min()


"""
__Regression: SN 2025wny__

The Zenodo record ships two worked examples for SN 2025wny, one predicting from the source position
and one from a single image. Both are reproduced below and compared against the numbers in the
distributed ``.out`` files. Those files are written with the C++ default of six significant
figures, so the comparison is made at ``rtol = 1e-5, atol = 1e-3``, which is the printed precision.
"""

WNY = dict(
    centre=(6.593, 6.295),
    source=(7.019908, 6.481430),
    image_a=(5.4450, 7.6350),
    e=2.848743e-01,
    b=1.685944,
    pa_deg=2.271276e01,
    d_ol=745.89,
    d_ls=867.04,
    z_lens=0.375,
    z_source=2.008,
)

# Columns: xpos ypos mag angle lags, from 2025wny_source.out.
WNY_SOURCE_OUT = np.array(
    [
        [4.84040, 5.58275, 2.75755, -67.8834, 125.383],
        [5.54144, 7.53257, -1.26685, 40.3545, 152.259],
        [6.74068, 4.69140, -1.23602, 5.26186, 153.516],
        [9.19257, 7.39661, 1.74532, -67.0344, 0.0],
    ]
)

# Columns: xpos ypos mag angle lags, from 2025wny_imageA.out (source at 6.93244 6.59150).
WNY_IMAGE_OUT = np.array(
    [
        [4.68748, 5.91044, 2.76318, -78.5903, 114.580],
        [5.44500, 7.63500, -1.55821, 40.5872, 133.573],
        [6.80686, 4.81970, -0.973223, 8.24825, 159.014],
        [9.06068, 7.60014, 1.76825, -62.1260, 0.0],
    ]
)


def solve_wny(x_s: float, y_s: float) -> np.ndarray:
    """Run the ported solver on the SN 2025wny model and return the .out columns."""
    x, y, magnification, angle = images_from_source(
        x_s=x_s,
        y_s=y_s,
        e=WNY["e"],
        b=WNY["b"],
        pa_deg=WNY["pa_deg"],
        centre=WNY["centre"],
    )
    lags = time_lags_days(
        x_image=x,
        y_image=y,
        x_s=x_s,
        y_s=y_s,
        e=WNY["e"],
        b=WNY["b"],
        pa_deg=WNY["pa_deg"],
        d_ol=WNY["d_ol"],
        d_ls=WNY["d_ls"],
        z_lens=WNY["z_lens"],
        z_source=WNY["z_source"],
        centre=WNY["centre"],
        h=0.7,
    )
    return np.stack([x, y, magnification, angle, lags], axis=1)


def print_regression(title: str, got: np.ndarray, expected: np.ndarray):
    print(f"\n{title}")
    print(
        f"  {'':2} {'xpos':>10} {'ypos':>10} {'mag':>10} {'angle':>10} {'lags':>10}"
        f"   {'max|delta|':>10}"
    )
    for i, (row, ref) in enumerate(zip(got, expected)):
        label = "ABCD"[i]
        print(
            f"  {label}: "
            + " ".join(f"{v:10.5f}" for v in row)
            + f"   {np.abs(row - ref).max():10.2e}"
        )
    print(
        f"  n_images = {len(got)}, max |delta| over all columns = {np.abs(got - expected).max():.2e}"
    )
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-3)


source_columns = solve_wny(x_s=WNY["source"][0], y_s=WNY["source"][1])
print_regression(
    "SN 2025wny, predicting from the source (2025wny_source.out)",
    source_columns,
    WNY_SOURCE_OUT,
)

x_s, y_s = source_from_image(
    x_i=WNY["image_a"][0],
    y_i=WNY["image_a"][1],
    e=WNY["e"],
    b=WNY["b"],
    pa_deg=WNY["pa_deg"],
    centre=WNY["centre"],
)
print(
    f"\nSource recovered from image A: ({x_s:.5f}, {y_s:.5f})  [reference 6.93244 6.59150]"
)
assert np.allclose([x_s, y_s], [6.93244, 6.59150], rtol=1e-5, atol=1e-3)

image_columns = solve_wny(x_s=x_s, y_s=y_s)
print_regression(
    "SN 2025wny, predicting from image A (2025wny_imageA.out)",
    image_columns,
    WNY_IMAGE_OUT,
)

"""
__The SIEP Astroid__

The 4/2/1 boundary is the astroid ``|p|^(2/3) + |q|^(2/3) = 1``, which in the source plane has
semi-axes

    a_long  = b e (2 - e) / (1 - e)^2      along the potential's major axis
    a_short = b e (2 - e) / (1 - e)        perpendicular to it

so the astroid is always elongated *along* the major axis, with axis ratio ``1 / (1 - e)``. Two
numbers, one free parameter ``e`` once ``b`` is fixed: the projection below picks the ``e`` that
matches both as well as it can.
"""


def siep_astroid_semi_axes(e: float, b: float) -> Tuple[float, float]:
    scale = b * e * (2.0 - e)
    return scale / (1.0 - e) ** 2, scale / (1.0 - e)


def ellipticity_from_caustic(b: float, a_long: float, a_short: float) -> float:
    """The potential ellipticity whose astroid best matches the two caustic semi-axes."""

    def cost(e):
        long_axis, short_axis = siep_astroid_semi_axes(e=e, b=b)
        return (long_axis / a_long - 1.0) ** 2 + (short_axis / a_short - 1.0) ** 2

    return float(minimize_scalar(cost, bounds=(1e-6, 0.9), method="bounded").x)


"""
__Projecting a PyAutoLens Model__

``WittWynne`` holds the nine numbers the ``.in`` file needs, in the solver's own ``(x, y)``
convention, plus the ``h`` the distances were computed with.
"""


@dataclass(frozen=True)
class WittWynne:
    centre: Tuple[float, float]  # (x, y) of the potential, arcsec.
    source: Tuple[float, float]  # (x, y) of the source, arcsec.
    e: float  # Potential ellipticity, 1 - q_psi.
    b: float  # Einstein radius, arcsec.
    pa_deg: float  # Position angle of the major axis, degrees East of North.
    d_ol: float  # Angular diameter distance to the lens, h^-1 Mpc.
    d_ls: float  # Angular diameter distance lens to source, h^-1 Mpc.
    z_lens: float
    z_source: float
    h: float

    def zeroed(self) -> "WittWynne":
        """The same model with the potential at the origin, so no sky coordinates are divulged."""
        return replace(
            self,
            centre=(0.0, 0.0),
            source=(self.source[0] - self.centre[0], self.source[1] - self.centre[1]),
        )

    def images(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return images_from_source(
            x_s=self.source[0],
            y_s=self.source[1],
            e=self.e,
            b=self.b,
            pa_deg=self.pa_deg,
            centre=self.centre,
        )

    def lags(self, x_image: np.ndarray, y_image: np.ndarray) -> np.ndarray:
        return time_lags_days(
            x_image=x_image,
            y_image=y_image,
            x_s=self.source[0],
            y_s=self.source[1],
            e=self.e,
            b=self.b,
            pa_deg=self.pa_deg,
            d_ol=self.d_ol,
            d_ls=self.d_ls,
            z_lens=self.z_lens,
            z_source=self.z_source,
            centre=self.centre,
            h=self.h,
        )


def caustic_semi_axes_from(
    caustic: np.ndarray, centre_yx: Tuple[float, float]
) -> Tuple[float, float, float]:
    """
    Semi-axes and long-axis orientation of a tangential caustic.

    The furthest point from the centre is a major cusp, which fixes the long axis; the short
    semi-axis is then the largest excursion perpendicular to it. Returns
    ``(a_long, a_short, angle_deg)`` with ``angle_deg`` counter-clockwise from the positive x-axis.
    """
    caustic = np.asarray(caustic)
    delta_y = caustic[:, 0] - centre_yx[0]
    delta_x = caustic[:, 1] - centre_yx[1]

    angle = np.arctan2(delta_y, delta_x)[np.argmax(np.hypot(delta_y, delta_x))]

    long_axis = delta_x * np.cos(angle) + delta_y * np.sin(angle)
    short_axis = -delta_x * np.sin(angle) + delta_y * np.cos(angle)

    return (
        float(np.abs(long_axis).max()),
        float(np.abs(short_axis).max()),
        float(np.degrees(angle) % 180.0),
    )


def _mass_and_shear_from(tracer):
    profiles = tracer.cls_list_from(cls=al.mp.MassProfile)
    shear = next((p for p in profiles if isinstance(p, al.mp.ExternalShear)), None)
    mass = next((p for p in profiles if not isinstance(p, al.mp.ExternalShear)), None)
    return mass, shear


def _distances_from(tracer) -> Tuple[float, float, float, float, float]:
    z_lens, z_source = tracer.plane_redshifts[0], tracer.plane_redshifts[-1]
    h = float(tracer.cosmology.H0) / 100.0
    d_ol = tracer.cosmology.angular_diameter_distance_to_earth_in_kpc_from(
        redshift=z_lens
    )
    d_ls = tracer.cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=z_lens, redshift_1=z_source
    )
    # kpc -> Mpc -> h^-1 Mpc.
    return float(d_ol) / 1e3 * h, float(d_ls) / 1e3 * h, z_lens, z_source, h


def witt_wynne_from_tracer(
    tracer,
    grid,
    source_centre: Tuple[float, float],
    centre: Tuple[float, float] = None,
    caustic_pixel_scale: float = 0.05,
) -> WittWynne:
    """
    Project a tracer onto SIEP parameters by matching its tangential caustic.

    ``b`` is the effective Einstein radius (the radius of the circle enclosing the same area as the
    tangential critical curve), the position angle is the orientation of the caustic's long axis,
    and ``e`` is chosen so the SIEP astroid matches both caustic semi-axes as closely as it can.
    The caustic is a property of the whole tracer, so secondary perturbers and external shear are
    folded in automatically rather than being modelled term by term.

    ``source_centre`` is the source-plane coordinate in **PyAutoLens** ``(y, x)`` order.
    """
    mass, _ = _mass_and_shear_from(tracer)
    centre_yx = mass.centre if centre is None else (centre[1], centre[0])

    lens_calc = al.LensCalc.from_tracer(tracer=tracer)

    b = float(
        lens_calc.einstein_radius_from(grid=grid, pixel_scale=caustic_pixel_scale)
    )
    caustic = np.asarray(
        lens_calc.tangential_caustic_list_from(
            grid=grid, pixel_scale=caustic_pixel_scale
        )[0]
    )

    a_long, a_short, angle_deg = caustic_semi_axes_from(
        caustic=caustic, centre_yx=centre_yx
    )

    d_ol, d_ls, z_lens, z_source, h = _distances_from(tracer)

    return WittWynne(
        centre=(centre_yx[1], centre_yx[0]),
        source=(source_centre[1], source_centre[0]),
        e=ellipticity_from_caustic(b=b, a_long=a_long, a_short=a_short),
        b=b,
        pa_deg=(angle_deg + 90.0) % 180.0,
        d_ol=d_ol,
        d_ls=d_ls,
        z_lens=z_lens,
        z_source=z_source,
        h=h,
    )


"""
__The Vector-Sum Projection__

Schechter's literal prescription is to add the ellipticity and the shear as vectors and throw away
the components perpendicular to their sum. Both quantities live in the 2-theta plane, so the sum is
well defined -- but three conversions are needed first.

**Density to potential.** ``ell_comps`` describe the *density*; the SIEP's ``e`` describes the
*potential*. Expanding both to first order in ellipticity, an isothermal density with axis ratio
``q`` has a relative quadrupole ``(1 - q) / 2`` in convergence, while a potential ellipticity ``e``
gives ``3 e / 2``. Matching them gives the familiar factor of three, ``e ~ (1 - q) / 3``.

**Shear to potential ellipticity.** At the Einstein radius the SIEP quadrupole ``b e r cos 2 theta``
and the shear quadrupole ``gamma r^2 cos 2 theta`` are equal in amplitude when ``e = gamma``, so
shear enters the sum with unit weight.

**Sign.** In **PyAutoGalaxy**'s convention a mass ellipticity at angle ``theta`` produces a
tangential caustic elongated *along* ``theta``, whereas an external shear at angle ``theta_gamma``
produces one elongated at ``theta_gamma + 90``. The shear therefore enters the 2-theta sum with a
minus sign, i.e. as ``-(gamma_1, gamma_2)``.

Both projections use the same ``b``, so any difference between them is due to ``e`` and the
position angle alone.
"""


def witt_wynne_vector_sum(
    tracer,
    grid,
    source_centre: Tuple[float, float],
    centre: Tuple[float, float] = None,
    caustic_pixel_scale: float = 0.05,
) -> WittWynne:
    """Project a tracer onto SIEP parameters by summing the ellipticity and shear vectors."""
    mass, shear = _mass_and_shear_from(tracer)
    centre_yx = mass.centre if centre is None else (centre[1], centre[0])

    axis_ratio, angle_deg = ag.convert.axis_ratio_and_angle_from(
        ell_comps=mass.ell_comps
    )
    e_potential = (1.0 - float(axis_ratio)) / 3.0

    gamma_1 = 0.0 if shear is None else float(shear.gamma_1)
    gamma_2 = 0.0 if shear is None else float(shear.gamma_2)

    component_1 = e_potential * np.cos(2.0 * np.radians(angle_deg)) - gamma_1
    component_2 = e_potential * np.sin(2.0 * np.radians(angle_deg)) - gamma_2

    e = float(np.hypot(component_1, component_2))
    angle_sum = np.degrees(np.arctan2(component_2, component_1)) / 2.0

    b = float(
        al.LensCalc.from_tracer(tracer=tracer).einstein_radius_from(
            grid=grid, pixel_scale=caustic_pixel_scale
        )
    )
    d_ol, d_ls, z_lens, z_source, h = _distances_from(tracer)

    return WittWynne(
        centre=(centre_yx[1], centre_yx[0]),
        source=(source_centre[1], source_centre[0]),
        e=e,
        b=b,
        pa_deg=(angle_sum + 90.0) % 180.0,
        d_ol=d_ol,
        d_ls=d_ls,
        z_lens=z_lens,
        z_source=z_source,
        h=h,
    )


"""
__Writing the isit4or2or1 Input__

The ``.in`` file is seven whitespace-separated lines, read by the C++ in this order:

    x_lens   y_lens
    x_source y_source
    ellipticity
    einstein_radius
    position_angle
    D_ol D_ls
    z_lens z_source

``zero_centre=True`` (the default) translates the lens to the origin and the source with it. The
solver is translation invariant, so every predicted position and lag is unchanged and only the
absolute sky coordinates are withheld -- which is what makes it safe to share a projected model
from a proprietary survey.

The distances written here are on the cosmology's own ``h``. Feeding this file to the compiled
`isit4or2or1` reproduces the positions and magnifications printed above exactly, but its lags come
out a factor ``h / 0.7`` smaller than the ones printed here, because the C++ fixes ``h = 0.7`` in
its time constant. Scale by ``0.7 / h`` to compare the two.
"""

ISIT_CSV_HEADER = (
    "name,x_lens,y_lens,x_source,y_source,ellipticity,einstein_radius,"
    "position_angle,d_ol,d_ls,z_lens,z_source"
)


def write_isit_input(path, model: WittWynne, zero_centre: bool = True) -> Path:
    if zero_centre:
        model = model.zeroed()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{model.centre[0]:.6e} {model.centre[1]:.6e}\n"
        f"{model.source[0]:.6e} {model.source[1]:.6e}\n"
        f"{model.e:.6e}\n"
        f"{model.b:.6e}\n"
        f"{model.pa_deg:.6e}\n"
        f"{model.d_ol:.6g} {model.d_ls:.6g}\n"
        f"{model.z_lens:.6g} {model.z_source:.6g}\n"
    )
    return path


def isit_csv_row(name: str, model: WittWynne, zero_centre: bool = True) -> str:
    if zero_centre:
        model = model.zeroed()

    return ",".join(
        [
            name,
            f"{model.centre[0]:.6f}",
            f"{model.centre[1]:.6f}",
            f"{model.source[0]:.6f}",
            f"{model.source[1]:.6f}",
            f"{model.e:.6f}",
            f"{model.b:.6f}",
            f"{model.pa_deg:.6f}",
            f"{model.d_ol:.4f}",
            f"{model.d_ls:.4f}",
            f"{model.z_lens:.4f}",
            f"{model.z_source:.4f}",
        ]
    )


"""
__Validation Against the Point Solver__

The projection is only useful if the SIEP it produces answers the broker's question the same way
the full model does. The check below simulates SIE + external shear lenses, solves them properly
with ``al.PointSolver`` and ``tracer.time_delays_from``, and compares against both projections:

- the 4/2/1 **verdict** (the number of images), which is what the broker actually reads,
- the **positions**, paired to their nearest counterpart,
- the **time lags**, referred to the leading image.

One case places the source at 95% of the distance to a caustic cusp, where the image count is most
fragile, and one places it outside the caustic, where the correct verdict is two.
"""

z_lens, z_source = 0.5, 1.5
cosmology = al.cosmo.Planck15()

grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)
solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)


def tracer_from(axis_ratio: float, angle: float, gamma: float, gamma_angle: float):
    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(
        magnitude=gamma, angle=gamma_angle
    )

    lens = al.Galaxy(
        redshift=z_lens,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle),
            einstein_radius=1.2,
        ),
        shear=al.mp.ExternalShear(gamma_1=gamma_1, gamma_2=gamma_2),
    )

    return al.Tracer(galaxies=[lens, al.Galaxy(redshift=z_source)], cosmology=cosmology)


def caustic_point_at_angle(
    tracer, angle_offset_deg: float, scale: float
) -> Tuple[float, float]:
    """A source position at ``scale`` times the caustic radius, offset from the long axis."""
    caustic = np.asarray(
        al.LensCalc.from_tracer(tracer=tracer).tangential_caustic_list_from(grid=grid)[
            0
        ]
    )
    _, _, angle_long = caustic_semi_axes_from(caustic=caustic, centre_yx=(0.0, 0.0))

    azimuth = np.degrees(np.arctan2(caustic[:, 0], caustic[:, 1]))
    target = angle_long + angle_offset_deg
    index = int(np.argmin(np.abs((azimuth - target + 180.0) % 360.0 - 180.0)))

    return float(scale * caustic[index, 0]), float(scale * caustic[index, 1])


def nearest_pairing(positions_a: np.ndarray, positions_b: np.ndarray) -> np.ndarray:
    distance = np.hypot(
        positions_a[:, None, 0] - positions_b[None, :, 0],
        positions_a[:, None, 1] - positions_b[None, :, 1],
    )
    return distance.argmin(axis=1), distance.min(axis=1)


cases = []

tracer = tracer_from(axis_ratio=0.75, angle=30.0, gamma=0.05, gamma_angle=10.0)
cases.append(("q=0.75 g=0.05 quad", tracer, (0.05, 0.02)))

tracer = tracer_from(axis_ratio=0.85, angle=110.0, gamma=0.10, gamma_angle=70.0)
cases.append(("q=0.85 g=0.10 quad", tracer, (-0.03, 0.06)))

tracer = tracer_from(axis_ratio=0.70, angle=0.0, gamma=0.03, gamma_angle=120.0)
cases.append(
    (
        "q=0.70 g=0.03 near cusp",
        tracer,
        caustic_point_at_angle(tracer, angle_offset_deg=10.0, scale=0.95),
    )
)
cases.append(
    (
        "q=0.70 g=0.03 just outside",
        tracer,
        caustic_point_at_angle(tracer, angle_offset_deg=10.0, scale=1.02),
    )
)
cases.append(
    (
        "q=0.70 g=0.03 well outside",
        tracer,
        caustic_point_at_angle(tracer, angle_offset_deg=10.0, scale=1.40),
    )
)

print(
    f"\n{'case':<28} {'projection':<10} {'e':>7} {'PA':>7} {'n_true':>7} {'n_siep':>7}"
    f" {'max dpos':>9} {'max dlag':>9} {'lag span':>9}"
)
print("-" * 104)

verdict_agreement = {"caustic": 0, "vector-sum": 0}

for name, tracer, source_centre in cases:
    positions = np.asarray(
        solver.solve(tracer=tracer, source_plane_coordinate=source_centre)
    )
    lags_true = np.asarray(tracer.time_delays_from(grid=al.Grid2DIrregular(positions)))
    lags_true = lags_true - lags_true.min()
    # PyAutoLens (y, x) -> solver (x, y).
    positions_true = positions[:, ::-1]

    for label, projection in (
        ("caustic", witt_wynne_from_tracer),
        ("vector-sum", witt_wynne_vector_sum),
    ):
        model = projection(tracer=tracer, grid=grid, source_centre=source_centre)

        x, y, _, _ = model.images()
        positions_siep = np.stack([x, y], axis=1)

        if len(positions_siep) == len(positions_true) and len(positions_siep) > 0:
            index, separation = nearest_pairing(positions_true, positions_siep)
            lags_siep = model.lags(x_image=x, y_image=y)
            d_position = separation.max()
            d_lag = np.abs(lags_true - lags_siep[index]).max()
        else:
            d_position, d_lag = float("nan"), float("nan")

        if len(positions_siep) == len(positions_true):
            verdict_agreement[label] += 1

        print(
            f"{name:<28} {label:<10} {model.e:7.4f} {model.pa_deg:7.2f}"
            f" {len(positions_true):7d} {len(positions_siep):7d}"
            f" {d_position:9.4f} {d_lag:9.3f} {lags_true.max():9.3f}"
        )

print(
    f"\nverdict agreement: caustic {verdict_agreement['caustic']}/{len(cases)},"
    f" vector-sum {verdict_agreement['vector-sum']}/{len(cases)}"
)
print("(max dpos in arcsec, max dlag in days, both against the full PointSolver model)")

assert verdict_agreement["caustic"] == len(cases), (
    "The caustic-matched projection did not reproduce the PointSolver verdict on every case. "
    "Under the automated test harness this most likely means `ENV: full_datasets` was not "
    "applied: PYAUTO_SMALL_DATASETS short-circuits `PointSolver.solve` to a fixed pair of "
    "positions, so every case reports two images whatever the model."
)

"""
__Writing a Model Out__

The projection of the first validation case, written in the ``.in`` format and as a CSV row.
"""

name, tracer, source_centre = cases[0]
model = witt_wynne_from_tracer(tracer=tracer, grid=grid, source_centre=source_centre)

directory = Path(tempfile.mkdtemp(prefix="witt_wynne_"))
path = write_isit_input(
    path=directory / "projected_model.in", model=model, zero_centre=True
)

print(f"\n{path}:\n")
print(path.read_text())
print(ISIT_CSV_HEADER)
print(isit_csv_row(name="validation_case_0", model=model))

"""
__Wrap Up__

What the validation showed, on the five cases above:

- **Verdict.** Both projections reproduced the 4/2/1 answer in every case, including a source at
  95% of the distance to a cusp and one only 2% outside the caustic. This is the number the broker
  reads, and the projection did not cost it.
- **Positions.** Predicted images land 0.07 to 0.16 arcsec from the true ones -- roughly a tenth of
  the image separation. That is close enough to associate a transient with a predicted image
  unambiguously, and nowhere near good enough to use as an astrometric constraint.
- **Lags.** Agreement is 0.9 to 6.7 days on lag spans of 17 to 71 days, i.e. 5 to 13%. The
  caustic-matched projection is clearly better where the shear is largest and the lag span short
  (1.8 vs 3.6 days at ``gamma = 0.1``); elsewhere the two are indistinguishable.

Prefer the caustic-matched projection. It reads the tracer's own tangential caustic, so extra
deflectors, a non-isothermal slope and multiple mass components are folded in without any per-term
conversion, whereas the vector sum depends on the density-to-potential factor of three and on
getting the shear's sign right -- both of which are approximations that only hold at small
ellipticity.

The limits of what was tested: a single deflector, ``gamma <= 0.1``, ``q >= 0.7``, an isothermal
slope, and a source described by its centroid alone. Stronger shear, a nearby perturber or a
significantly non-isothermal slope were not checked, and there is no reason to assume the same
accuracy holds there. If the projection is being used to decide whether a real transient is a
fourth image, re-run the full ``PointSolver`` on the candidate before believing anything more than
the verdict.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

This guide computes tangential caustics and solves the lens equation on a 200x200
grid; SMALL_DATASETS caps grids to 16x16 at 0.6" pixels, which destroys both.

ENV: full_datasets
"""
