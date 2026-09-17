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
   ``PointSolver`` and ``time_delays_from`` on a grid of simulated SIE + shear lenses, inside and
   outside the caustic.

Two projections are implemented. The **caustic-matched** one fits the SIEP astroid to the tracer's
own tangential caustic. The **vector-sum** one follows Schechter's literal prescription: add the
ellipticity and shear as vectors in the 2-theta plane and keep the magnitude and angle of the sum.
The validation section below reports how they compare, and the Wrap Up says which to use.

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
- **The two orders are mixed in the signatures below, deliberately.** ``centre`` is ``(x, y)`` --
  the solver's order, because it is written straight into the ``.in`` file -- while
  ``source_centre`` is ``(y, x)``, **PyAutoLens**'s order, because it comes from a tracer. Every
  function states which it takes; getting them the wrong way round is the easiest mistake to make
  here and the hardest to see, because the output stays finite and plausible.
- A position angle ``phi`` maps to a **PyAutoLens** angle (degrees counter-clockwise from the
  positive x-axis) as ``angle = phi - 90``, and back as ``PA_isit = (angle_ccw + 90) mod 180``.
  Getting this backwards -- writing the mirrored ``90 - angle`` -- reflects every predicted image
  about the potential's minor axis, which moves them by **about 1 arcsec** on the lenses validated
  below (0.80 to 1.07 arcsec, against 0.12 arcsec for the correct map, and identical only at
  ``angle = 0``). That is why the validation section compares positions and not just image counts.
- ``e`` is the ellipticity of the *potential*, ``e = 1 - q_psi``, not of the density. It is not the
  same quantity as **PyAutoLens**'s ``ell_comps`` magnitude ``(1 - q)/(1 + q)``, which describes
  the density. To first order in ellipticity the density-to-potential map is ``e ~ (1 - q) / 3``.
- In **PyAutoGalaxy**'s convention a mass ellipticity at angle ``theta`` elongates the tangential
  caustic *along* ``theta``, whereas an external shear at ``theta_gamma`` elongates it at
  ``theta_gamma + 90``. The shear therefore enters the 2-theta vector sum **with a minus sign**, as
  ``-(gamma_1, gamma_2)``.
- Distances are angular diameter distances in ``h^-1 Mpc``, with ``D_H = 3000 h^-1 Mpc``.
- The original hardcodes ``h = 0.7`` in its ``TIMECONSTANT`` while taking distances in
  ``h^-1 Mpc``, so its lags are on an ``h = 0.7`` scale whatever cosmology produced the distances.
  This port **exposes** ``h`` as an argument defaulting to ``0.7``, so the regression below is
  exact and the tracer's own ``h`` can be passed when comparing against ``tracer.time_delays_from``.
- The time constant keeps the C++'s own rounded literals -- ``D_H = 3000`` (+0.07% against
  ``c / H0``), ``9.78e9 / h`` yr (+0.02%) and ``365.0`` days per year rather than the Julian 365.25
  (-0.07%). Together they make the lags **0.12% low** against an exact
  ``(1 + z_l) D_l D_s / (c D_ls)``. That is kept deliberately, so the lags are like-for-like with
  `isit4or2or1`'s own output; it is negligible against the 5 to 13% the projection itself costs.

__Degeneracies__

The quartic has places where it returns a finite, plausible and *wrong* answer, and the functions
below screen them rather than trusting the output:

- A source **on a potential axis or at the lens centre** (``min(|p|, |q|) < P_Q_MINIMUM``), an
  ``e`` outside ``(0, 1)``, a non-positive ``b``, or any non-finite output gives a **NaN row**
  whose verdict, read with ``n_images_from``, is the sentinel ``N_IMAGES_SENTINEL = -1``.
- Half the real roots of the quartic sit on the **wrong branch** of the square root that maps back
  to the image plane. They are filtered by their lens-equation residual, which is what lets the
  "1" of `isit4or2or1` actually fire.
- A projection that cannot be made at all -- no isothermal-family mass profile, no tangential
  caustic, an ellipticity and a shear that cancel -- returns a ``WittWynne`` with ``valid=False``
  and a ``reason``, never an exception and never a plausible wrong model.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

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

A root counts as real when its imaginary part is below ``threshold`` (Falor & Schechter 2022,
section 2.3).

**Not every real root is an image.** The map back to the image plane takes the square root of
``G = sqrt(x^2 + y^2 / (1 - e)^2)``, and half the real roots can sit on its wrong branch: the
position recovered from such a root solves the lens equation of a *negative* mass. Counting real
roots alone therefore returns only ever 4 or 2 images, and the "1" of `isit4or2or1` never fires.

``find_intersections`` drops those roots. The recovered image must satisfy the SIEP lens equation
to ``residual_threshold * max(1, p^2 + q^2)`` in units of ``b``, equivalently
``sign(x_hat) = sign(x)`` and ``sign(y_hat) = sign(y)``. The two branches are separated by fifteen
orders of magnitude, so the threshold is not a tuning knob: measured out to eight times the caustic
scale, a genuine root's residual is below ``1.3e-9`` and a wrong-branch root's is about ``2``. It
is scaled by ``p^2 + q^2`` only because the quartic's own conditioning degrades with its
coefficients -- a genuine root's residual grows as roughly ``8e-11 (p^2 + q^2)``, so a fixed
``1e-10`` would start discarding the real images of far sources.

A root that misses the lens equation by more than the tolerance but by far less than a wrong-branch
root's order unity means the *quartic* is ill-conditioned rather than the *root* unphysical. **No
roots are returned at all** in that case, so ``images_from_source`` reports the degeneracy sentinel
instead of a verdict one image short.

The branch test needs the potential ellipticity ``e``: it is not fixed by ``(p, q)`` alone, because
the 2-image / 1-image boundary moves with ``e`` (a circular lens has none -- it images every source
twice, only inside its Einstein radius). That is why ``e`` is an argument here.

The ACLE itself degenerates when ``q = 0`` (source exactly on the potential's major axis) or
``p = 0`` (minor axis): the quartic acquires a double root and ``y = q x / (x - p)`` is
indeterminate. Both are screened by ``images_from_source`` before this function is reached.
"""

# Below this, a source lies on a potential axis (or at the centre) to within the quartic's ability
# to tell, and `images_from_source` returns the sentinel row.
P_Q_MINIMUM = 1e-6

# The verdict returned for a sentinel (all-NaN) row.
N_IMAGES_SENTINEL = -1

# A wrong-branch quartic root misses the lens equation by order unity in units of `b` (measured
# minimum 1.49 over a grid out to eight times the caustic scale). A root that misses it by more
# than the tolerance but by much less than this is neither: the quartic is ill-conditioned, which
# happens in a narrow band just outside the on-axis screen of `P_Q_MINIMUM`. `find_intersections`
# then returns nothing at all, so the verdict is the sentinel rather than an image short.
AMBIGUOUS_RESIDUAL = 1e-2


def find_intersections(
    p: float,
    q: float,
    e: float,
    threshold: float = 1e-5,
    residual_threshold: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """The scaled image positions ``(x_hat, y_hat)`` of the ACLE, wrong-branch roots removed."""
    roots = solve_quartic(1.0, -2.0 * p, p * p + q * q - 1.0, 2.0 * p, -p * p)

    scale = e * (2.0 - e)
    axis_ratio_squared = (1.0 - e) ** 2
    tolerance = residual_threshold * max(1.0, p * p + q * q)

    x_list, y_list = [], []

    for root in roots:
        if abs(root.imag) >= threshold:
            continue

        x_hat = float(root.real)
        if x_hat == p:
            continue

        y_hat = q * x_hat / (x_hat - p)
        if not (np.isfinite(x_hat) and np.isfinite(y_hat)):
            continue

        # The recovered image position in the registered frame, in units of b:
        # x = b x_hat + x_s and y = b y_hat / (1 - e) + y_s.
        x_over_b = x_hat + p * scale / axis_ratio_squared
        y_over_b = (y_hat - q * scale) / (1.0 - e)

        g = np.sqrt(x_over_b**2 + y_over_b**2 / axis_ratio_squared)
        if not np.isfinite(g) or g <= 0.0:
            continue

        # The lens equation, divided by b: x_hat = x / G and y_hat = y / ((1 - e) G), which only
        # the physical branch satisfies.
        residual = max(
            abs(x_hat - x_over_b / g),
            abs(y_hat - y_over_b / ((1.0 - e) * g)),
        )

        if not np.isfinite(residual):
            continue

        if residual > tolerance:
            if residual < AMBIGUOUS_RESIDUAL:
                # Neither a solution nor the wrong branch: the quartic is too ill-conditioned here
                # to say which, so the whole solve is reported degenerate rather than silently
                # short of an image.
                return np.asarray([], dtype=float), np.asarray([], dtype=float)
            continue

        x_list.append(x_hat)
        y_list.append(y_hat)

    return np.asarray(x_list, dtype=float), np.asarray(y_list, dtype=float)


"""
__Registered Coordinates__

"Registered" coordinates are centred on the potential with the x-axis along its major axis. The
rotation is a proper one (determinant +1), so handedness is preserved and the inverse is the
transpose. Both functions take ``centre`` as ``(x, y)``, the solver's order.
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

``images_from_source`` returns four arrays of the same length, ``(x_image, y_image, magnification,
angle_deg)``, in the solver's ``(x, y)`` order; ``centre`` is the potential's centre as ``(x, y)``,
*not* **PyAutoLens**'s ``(y, x)``. The number of images is the length of those arrays, read with
``n_images_from``.

**The solve is degenerate** -- and all four arrays come back as a single NaN, verdict
``N_IMAGES_SENTINEL`` (-1) -- when

- ``min(|p|, |q|) < P_Q_MINIMUM`` (1e-6): the source lies on a potential axis or at the lens
  centre. This is the case worth guarding hardest. The quartic does not blow up there; it returns
  four finite positions of which two are wrong by **0.6 to 0.8 arcsec**, with finite
  magnifications and finite lags and no warning at all. The original C++ has the same behaviour;
  this port screens it instead of documenting it;
- ``e`` is outside ``(0, 1)``, ``b`` is not positive, or any input is non-finite;
- no quartic root survives the lens-equation filter of ``find_intersections``;
- any returned position, magnification or angle is non-finite (a source exactly on a fold or cusp).

It never raises. A sentinel row is all-NaN, so it can never be confused with a genuine 1-image
verdict, whose position is finite.
"""


def _nan_row() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The sentinel returned for a degenerate solve: one all-NaN image row."""
    nan = np.array([np.nan])
    return nan.copy(), nan.copy(), nan.copy(), nan.copy()


def n_images_from(x_image: np.ndarray, y_image: Optional[np.ndarray] = None) -> int:
    """The 4 / 2 / 1 verdict, or ``N_IMAGES_SENTINEL`` (-1) when the solve was degenerate."""
    x_image = np.asarray(x_image, dtype=float)

    if x_image.size == 0 or not np.all(np.isfinite(x_image)):
        return N_IMAGES_SENTINEL

    if y_image is not None:
        y_image = np.asarray(y_image, dtype=float)
        if y_image.size != x_image.size or not np.all(np.isfinite(y_image)):
            return N_IMAGES_SENTINEL

    return int(x_image.size)


def _is_solvable(e: float, b: float, x_s: float, y_s: float, centre) -> bool:
    """Whether the SIEP parameters are inside the solver's domain at all."""
    values = [float(e), float(b), float(x_s), float(y_s), *[float(c) for c in centre]]

    if not all(np.isfinite(values)):
        return False

    return 0.0 < float(e) < 1.0 and float(b) > 0.0


def images_from_source(
    x_s: float,
    y_s: float,
    e: float,
    b: float,
    pa_deg: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    threshold: float = 1e-5,
    residual_threshold: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not _is_solvable(e=e, b=b, x_s=x_s, y_s=y_s, centre=centre):
        return _nan_row()

    e, b = float(e), float(b)
    phi = np.radians(pa_deg)

    x_reg_s, y_reg_s = to_registered(x_s, y_s, pa_deg=pa_deg, centre=centre)

    with np.errstate(all="ignore"):
        p = x_reg_s * (1.0 - e) ** 2 / (b * (2.0 - e) * e)
        q = -y_reg_s * (1.0 - e) / (b * (2.0 - e) * e)

        if not (np.isfinite(p) and np.isfinite(q)):
            return _nan_row()

        if min(abs(float(p)), abs(float(q))) < P_Q_MINIMUM:
            return _nan_row()

        x_hat, y_hat = find_intersections(
            p=p,
            q=q,
            e=e,
            threshold=threshold,
            residual_threshold=residual_threshold,
        )

        if x_hat.size == 0:
            return _nan_row()

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
        angle = np.degrees(angle)

    for values in (x_image, y_image, magnification, angle):
        if not np.all(np.isfinite(values)):
            return _nan_row()

    return x_image, y_image, magnification, angle


"""
__Source From an Image__

The inverse problem: given one image position, where is the source? Intersecting Wynne's ellipse
with Witt's hyperbola at the known image gives a closed form, so a single detected image of a
candidate transient fixes the source and hence the other three images.

``centre`` is ``(x, y)``, the solver's order. The function returns ``(nan, nan)`` rather than
raising when ``e`` is outside ``(0, 1)``, ``b`` is not positive or an input is non-finite; the
image plane's own degeneracy -- an image exactly on the potential's major axis, ``y_reg = 0`` --
returns the lens centre.
"""


def source_from_image(
    x_i: float,
    y_i: float,
    e: float,
    b: float,
    pa_deg: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    if not _is_solvable(e=e, b=b, x_s=x_i, y_s=y_i, centre=centre):
        return float("nan"), float("nan")

    x_reg, y_reg = to_registered(x_i, y_i, pa_deg=pa_deg, centre=centre)

    with np.errstate(all="ignore"):
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
comoving distances ``chi = D (1 + z)``. ``centre`` is ``(x, y)``, the solver's order.

The time constant keeps the C++'s rounded literals -- ``D_H = 3000``, ``9.78e9 / h`` yr and
``365.0`` days per year -- which together run **0.12% low** against the exact
``(1 + z_l) D_l D_s / (c D_ls)``. That is deliberate: it makes these lags directly comparable with
`isit4or2or1`'s own. The original fixes ``h = 0.7`` here regardless of the distances it was given,
so ``h`` defaults to ``0.7``; pass the tracer's own ``h`` to compare against
``tracer.time_delays_from``, whose lags are a factor ``h / 0.7`` different.
"""

D_H = 3000.0  # Hubble distance in h^-1 Mpc (the C++'s own rounded literal).


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

    with np.errstate(all="ignore"):
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

The ``.out`` rows are hard-coded here, so this regression is self-contained: it does not need the
Zenodo archive on disk. What it therefore does *not* re-check is the ``.in`` round trip -- whether
the compiled C++ reads the file this guide writes in the order it writes it. That was verified once
when the guide was first written, but ``SIEP_CLI.v1.0.cpp`` is not present in this workspace and
the check could not be re-run for the corrections made here. To redo it:

1. Download ``SIEP_CLI.v1.0.cpp`` from the Zenodo record, DOI 10.5281/zenodo.20086659.
2. ``g++ -std=c++17 -O2 -o isit4or2or1 SIEP_CLI.v1.0.cpp``.
3. Run it on the ``.in`` file written at the end of this guide, and on one written from ``WNY``.
4. Diff its ``xpos ypos mag angle lags`` columns against the rows printed below, remembering that
   its lags are a factor ``h / 0.7`` different when the distances are not on ``h = 0.7``.
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
        f"  n_images = {n_images_from(got[:, 0])}, "
        f"max |delta| over all columns = {np.abs(got - expected).max():.2e}"
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
__The Degeneracy Sentinel__

The screens described above are worth seeing fire, because every one of these cases used to come
back as a finite number. A source on the potential's major axis, on its minor axis and at the lens
centre all return the sentinel verdict -1 and an all-NaN row, whereas a source a hundredth of an
arcsecond off the axis is solved normally.
"""

degenerate_cases = [
    ("on the major axis (q=0)", dict(x_s=6.593, y_s=6.295 + 0.30, e=0.2, b=1.2)),
    ("on the minor axis (p=0)", dict(x_s=6.593 + 0.30, y_s=6.295, e=0.2, b=1.2)),
    ("at the lens centre", dict(x_s=6.593, y_s=6.295, e=0.2, b=1.2)),
    ("e = 0 (circular)", dict(x_s=6.593 + 0.20, y_s=6.295 + 0.10, e=0.0, b=1.2)),
    ("e = 1 (degenerate)", dict(x_s=6.593 + 0.20, y_s=6.295 + 0.10, e=1.0, b=1.2)),
    ("b <= 0", dict(x_s=6.593 + 0.20, y_s=6.295 + 0.10, e=0.2, b=0.0)),
    (
        '0.01" off the major axis',
        dict(x_s=6.593 + 0.01, y_s=6.295 + 0.30, e=0.2, b=1.2),
    ),
]

print(f"\n{'degenerate case':<28} {'n_images':>9} {'x_image[0]':>12}")
print("-" * 51)

for label, parameters in degenerate_cases:
    x, y, _, _ = images_from_source(pa_deg=0.0, centre=(6.593, 6.295), **parameters)
    print(f"{label:<28} {n_images_from(x, y):9d} {x[0]:12.5f}")

"""
__The SIEP Astroid__

The 4/2/1 boundary is the astroid ``|p|^(2/3) + |q|^(2/3) = 1``, which in the source plane has
semi-axes

    a_long  = b e (2 - e) / (1 - e)^2      along the potential's major axis
    a_short = b e (2 - e) / (1 - e)        perpendicular to it

so the astroid is always elongated *along* the major axis, with axis ratio ``1 / (1 - e)``. Two
numbers, one free parameter ``e`` once ``b`` is fixed: the projection below picks the ``e`` that
matches both as well as it can.

The fit is bounded to ``(1e-6, 0.9)``. It is a genuine compromise at large ellipticity, because a
real isothermal caustic is not an astroid: at ``q = 0.5`` the best fit is 10% short on the long
semi-axis and 11% long on the short one, and by ``q <= 0.35`` the residuals reach -18% / +16%.
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

``valid`` is False on a **sentinel** model -- one the projection could not produce -- whose ``e``,
``b``, ``pa_deg`` and ``centre`` are NaN and whose ``reason`` says why. The source and the
distances are kept even then, so a catalogue row for a lens that could not be projected still says
which lens it was. A *valid* model may also carry a ``reason``: it records anything the projection
had to choose, such as which of two admissible mass profiles was used.
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
    valid: bool = True
    reason: str = ""

    def zeroed(self) -> "WittWynne":
        """The same model with the potential at the origin, so no sky coordinates are divulged."""
        return replace(
            self,
            centre=(0.0, 0.0),
            source=(self.source[0] - self.centre[0], self.source[1] - self.centre[1]),
        )

    def images(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(x_image, y_image, magnification, angle_deg)``; a NaN row on a sentinel model."""
        return images_from_source(
            x_s=self.source[0],
            y_s=self.source[1],
            e=self.e,
            b=self.b,
            pa_deg=self.pa_deg,
            centre=self.centre,
        )

    def n_images(self) -> int:
        """The 4 / 2 / 1 verdict, or ``N_IMAGES_SENTINEL`` (-1)."""
        x_image, y_image, _, _ = self.images()
        return n_images_from(x_image, y_image)

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
    semi-axis is then the largest excursion perpendicular to it. ``caustic`` and ``centre_yx`` are
    both in **PyAutoLens** ``(y, x)`` order. Returns ``(a_long, a_short, angle_deg)`` with
    ``angle_deg`` counter-clockwise from the positive x-axis.
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


"""
__Picking the Mass Profile__

The mass profile is selected **by class**, and this is not fussiness. ``al.lp_basis.Basis`` -- the
container an MGE lens light is built from -- *subclasses* ``MassProfile``, so the natural-looking
rule "any ``MassProfile`` that is not an ``ExternalShear``" picks the **lens light** as the lens
mass on the commonest Euclid-pipeline model shape (MGE light + SIE + shear). Nothing raises, no
field is NaN: the caustic projection simply comes back with the position angle of the light,
**84 degrees** away from the truth, and the vector sum reads the light's ``ell_comps``.

The projection is only defined for an isothermal-like potential anyway, so the admissible family is
named explicitly and anything else returns a sentinel rather than a plausible wrong answer. With
more than one admissible mass profile -- a main deflector and a perturber -- the first is used and
the choice is recorded in ``WittWynne.reason``; note that the caustic-matched projection still
folds the perturber in through the caustic, while the vector sum ignores it.
"""


def _mass_classes():
    """The mass profile classes the SIEP projection accepts, by class rather than by elimination."""
    return (
        ag.mp.Isothermal,
        ag.mp.IsothermalSph,
        ag.mp.PowerLaw,
        ag.mp.PowerLawSph,
    )


def _is_admissible_mass(profile) -> bool:
    if isinstance(profile, (ag.lp_basis.Basis, ag.LightProfile)):
        return False

    if isinstance(profile, ag.mp.ExternalShear):
        return False

    return isinstance(profile, _mass_classes())


def _admissible_mass_list_from(tracer) -> list:
    """Every Isothermal/PowerLaw-family mass profile in the tracer, in order."""
    return [
        profile
        for profile in tracer.cls_list_from(cls=ag.mp.MassProfile)
        if _is_admissible_mass(profile)
    ]


def _shear_from(tracer):
    return next(
        (
            profile
            for profile in tracer.cls_list_from(cls=ag.mp.MassProfile)
            if isinstance(profile, ag.mp.ExternalShear)
        ),
        None,
    )


def _mass_and_shear_from(tracer):
    """
    The lens's mass profile and its external shear, both **selected by class**.

    Returns ``(mass, shear)``, either of which may be ``None``: ``mass`` when the tracer holds no
    Isothermal/PowerLaw-family profile, ``shear`` when there is no ``ExternalShear``.
    """
    mass_list = _admissible_mass_list_from(tracer)
    return (mass_list[0] if mass_list else None), _shear_from(tracer)


def _distances_from(tracer) -> Tuple[float, float, float, float, float]:
    """``(d_ol, d_ls, z_lens, z_source, h)``, the distances in ``h^-1 Mpc``."""
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


def _mass_note_from(mass_list: list) -> str:
    if len(mass_list) < 2:
        return ""

    names = ", ".join(type(mass).__name__ for mass in mass_list)
    return (
        f"{len(mass_list)} admissible mass profiles ({names}); projected the "
        f"first, {type(mass_list[0]).__name__}"
    )


def _sentinel_from(
    reason: str,
    source_centre: Tuple[float, float],
    distances: Optional[Tuple[float, float, float, float, float]] = None,
) -> WittWynne:
    """An invalid ``WittWynne``: NaN parameters, ``valid=False`` and ``reason``."""
    d_ol, d_ls, z_lens, z_source, h = distances or (np.nan,) * 5

    return WittWynne(
        centre=(np.nan, np.nan),
        source=(source_centre[1], source_centre[0]),
        e=np.nan,
        b=np.nan,
        pa_deg=np.nan,
        d_ol=d_ol,
        d_ls=d_ls,
        z_lens=z_lens,
        z_source=z_source,
        h=h,
        valid=False,
        reason=reason,
    )


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
    folded in automatically rather than being modelled term by term. This is the projection to
    prefer -- see the validation below.

    ``source_centre`` is the source-plane coordinate in **PyAutoLens** ``(y, x)`` order; ``centre``,
    if given, overrides the mass profile's centre and is in the solver's ``(x, y)`` order.

    Returns a **sentinel** ``WittWynne`` (``valid=False``) rather than raising when the tracer holds
    no admissible mass profile (a shear-only or light-only galaxy) or no tangential caustic (a
    sub-critical lens).
    """
    distances = _distances_from(tracer)

    mass_list = _admissible_mass_list_from(tracer)

    if not mass_list:
        return _sentinel_from(
            reason="no Isothermal/PowerLaw-family mass profile in the tracer",
            source_centre=source_centre,
            distances=distances,
        )

    mass = mass_list[0]
    centre_yx = mass.centre if centre is None else (centre[1], centre[0])

    lens_calc = al.LensCalc.from_tracer(tracer=tracer)

    b = float(
        lens_calc.einstein_radius_from(grid=grid, pixel_scale=caustic_pixel_scale)
    )

    caustic_list = lens_calc.tangential_caustic_list_from(
        grid=grid, pixel_scale=caustic_pixel_scale
    )

    if len(caustic_list) == 0 or np.asarray(caustic_list[0]).size == 0:
        return _sentinel_from(
            reason="no tangential caustic: the lens is sub-critical on this grid",
            source_centre=source_centre,
            distances=distances,
        )

    a_long, a_short, angle_deg = caustic_semi_axes_from(
        caustic=np.asarray(caustic_list[0]), centre_yx=centre_yx
    )

    if not np.isfinite(b) or b <= 0.0 or min(a_long, a_short) <= 0.0:
        return _sentinel_from(
            reason=(
                f"degenerate caustic fit: b={b:.4g}, semi-axes "
                f"({a_long:.4g}, {a_short:.4g})"
            ),
            source_centre=source_centre,
            distances=distances,
        )

    d_ol, d_ls, z_lens, z_source, h = distances

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
        reason=_mass_note_from(mass_list),
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

**Sign.** The shear enters the 2-theta sum with a minus sign, i.e. as ``-(gamma_1, gamma_2)``: a
mass ellipticity at angle ``theta`` elongates the tangential caustic *along* ``theta``, whereas an
external shear at ``theta_gamma`` elongates it at ``theta_gamma + 90``. The decisive test is a
nearly round lens with a strong aligned shear, where the sign flips the predicted position angle by
a full 90 degrees; with the minus sign the vector sum and the caustic match agree.

Both projections use the same ``b``, so any difference between them is due to ``e`` and the
position angle alone.

The two quantities can also **cancel**, which is what ``E_MINIMUM`` guards. It happens exactly when
``e_potential = gamma`` and the two are aligned -- ``q = 0.7`` with ``gamma = 0.10`` is on the nose
-- and it is not a rare corner: at ``q = 0.85``, ``gamma = 0.06`` the sum survives but collapses to
``e = 0.01`` and the position angle flips by 90 degrees. Since ``images_from_source`` divides by
``e``, an unguarded cancellation returns image positions of millions of arcseconds without
complaint. The caustic-matched projection has no such failure mode; it reads the caustic the
cancelling model actually has.
"""

# Below this the ellipticity and the shear have cancelled and the SIEP is effectively circular.
E_MINIMUM = 1e-3


def witt_wynne_vector_sum(
    tracer,
    grid,
    source_centre: Tuple[float, float],
    centre: Tuple[float, float] = None,
    caustic_pixel_scale: float = 0.05,
) -> WittWynne:
    """
    Project a tracer onto SIEP parameters by summing the ellipticity and shear vectors.

    ``source_centre`` is in **PyAutoLens** ``(y, x)`` order; ``centre``, if given, is in the
    solver's ``(x, y)`` order. Unlike the caustic-matched projection this one reads the mass
    profile's own ``ell_comps``, so picking the wrong profile silently mis-projects it -- the
    profile is selected by class, and a tracer with none returns a sentinel, as does an ellipticity
    and a shear that cancel to ``e < E_MINIMUM``.
    """
    distances = _distances_from(tracer)

    mass_list = _admissible_mass_list_from(tracer)
    shear = _shear_from(tracer)

    if not mass_list:
        return _sentinel_from(
            reason="no Isothermal/PowerLaw-family mass profile in the tracer",
            source_centre=source_centre,
            distances=distances,
        )

    mass = mass_list[0]
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

    if not np.isfinite(e) or e < E_MINIMUM:
        return _sentinel_from(
            reason=(
                f"ellipticity and shear cancel: vector-sum e = {e:.3g}, below "
                f"{E_MINIMUM:g}"
            ),
            source_centre=source_centre,
            distances=distances,
        )

    b = float(
        al.LensCalc.from_tracer(tracer=tracer).einstein_radius_from(
            grid=grid, pixel_scale=caustic_pixel_scale
        )
    )

    if not np.isfinite(b) or b <= 0.0:
        return _sentinel_from(
            reason=f"degenerate Einstein radius: b={b:.4g}",
            source_centre=source_centre,
            distances=distances,
        )

    d_ol, d_ls, z_lens, z_source, h = distances

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
        reason=_mass_note_from(mass_list),
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

Neither writer checks ``model.valid`` -- a sentinel model writes ``nan`` fields. Check it first.
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
the full model does. The checks below simulate SIE + external shear lenses, solve them properly
with ``al.PointSolver`` and ``tracer.time_delays_from``, and compare against both projections:

- the 4/2/1 **verdict** (the number of images), which is what the broker actually reads,
- the **positions**, paired to their nearest counterpart,
- the **time lags**, referred to the leading image.

Five named cases come first, as a readable table: two quads, a source at 95% of the distance to a
caustic cusp where the image count is most fragile, and the same lens with the source 2% and 40%
outside the caustic, where the correct verdict is two. A grid of 72 cases then follows, because
five cases are not enough to say anything about where the projection breaks down -- and, as that
grid shows, these five are not representative of the outside-caustic regime.
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


def caustic_of(tracer, grid) -> np.ndarray:
    """The tracer's tangential caustic, in **PyAutoLens** ``(y, x)`` order."""
    return np.asarray(
        al.LensCalc.from_tracer(tracer=tracer).tangential_caustic_list_from(grid=grid)[
            0
        ]
    )


def caustic_point_at_angle(
    caustic: np.ndarray, angle_offset_deg: float, scale: float
) -> Tuple[float, float]:
    """A source position at ``scale`` times the caustic radius, offset from the long axis."""
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


def point_solver_truth(tracer, source_centre: Tuple[float, float], solver):
    """The ``PointSolver`` positions in solver ``(x, y)`` order, and their lags."""
    positions = np.asarray(
        solver.solve(tracer=tracer, source_plane_coordinate=source_centre)
    )
    lags = np.asarray(tracer.time_delays_from(grid=al.Grid2DIrregular(positions)))

    return positions[:, ::-1], lags - lags.min()


def compare(model: WittWynne, positions_true: np.ndarray, lags_true: np.ndarray):
    """``(n_siep, max position error, max lag error)`` of a projection against the truth."""
    x, y, _, _ = model.images()
    n_siep = n_images_from(x, y)

    if n_siep != len(positions_true):
        return n_siep, float("nan"), float("nan")

    index, separation = nearest_pairing(positions_true, np.stack([x, y], axis=1))
    lags_siep = model.lags(x_image=x, y_image=y)

    return (
        n_siep,
        float(separation.max()),
        float(np.abs(lags_true - lags_siep[index]).max()),
    )


run_start = time.time()

cases = []

tracer = tracer_from(axis_ratio=0.75, angle=30.0, gamma=0.05, gamma_angle=10.0)
cases.append(("q=0.75 g=0.05 quad", tracer, (0.05, 0.02)))

tracer = tracer_from(axis_ratio=0.85, angle=110.0, gamma=0.10, gamma_angle=70.0)
cases.append(("q=0.85 g=0.10 quad", tracer, (-0.03, 0.06)))

tracer = tracer_from(axis_ratio=0.70, angle=0.0, gamma=0.03, gamma_angle=120.0)
caustic = caustic_of(tracer, grid=grid)
cases.append(
    (
        "q=0.70 g=0.03 near cusp",
        tracer,
        caustic_point_at_angle(caustic, angle_offset_deg=10.0, scale=0.95),
    )
)
cases.append(
    (
        "q=0.70 g=0.03 just outside",
        tracer,
        caustic_point_at_angle(caustic, angle_offset_deg=10.0, scale=1.02),
    )
)
cases.append(
    (
        "q=0.70 g=0.03 well outside",
        tracer,
        caustic_point_at_angle(caustic, angle_offset_deg=10.0, scale=1.40),
    )
)

print(
    f"\n{'case':<28} {'projection':<10} {'e':>7} {'PA':>7} {'n_true':>7} {'n_siep':>7}"
    f" {'max dpos':>9} {'max dlag':>9} {'lag span':>9}"
)
print("-" * 104)

verdict_agreement = {"caustic": 0, "vector-sum": 0}

for name, tracer, source_centre in cases:
    positions_true, lags_true = point_solver_truth(tracer, source_centre, solver)

    for label, projection in (
        ("caustic", witt_wynne_from_tracer),
        ("vector-sum", witt_wynne_vector_sum),
    ):
        model = projection(tracer=tracer, grid=grid, source_centre=source_centre)

        n_siep, d_position, d_lag = compare(model, positions_true, lags_true)

        if n_siep == len(positions_true):
            verdict_agreement[label] += 1

        print(
            f"{name:<28} {label:<10} {model.e:7.4f} {model.pa_deg:7.2f}"
            f" {len(positions_true):7d} {n_siep:7d}"
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
__A Grid of Lenses, Inside and Outside the Caustic__

Five cases can only tell you that nothing is obviously broken. The grid below is the smallest one
that separates the regimes that matter:

- **axis ratio** ``q`` in 0.5, 0.7, 0.85, 0.95 -- from nearly round to rounder than the astroid fit
  can follow;
- **shear** ``gamma`` in 0.0, 0.06, 0.15 -- from none to stronger than any of the five cases above;
- **misalignment** between the mass and the shear position angles of 0, 45 and 90 degrees, which is
  what the vector sum's ``e`` is most sensitive to;
- the source at **0.5x** and **1.02x** the caustic scale, i.e. comfortably inside (truth: 4 images)
  and just outside it (truth: 2).

Both projections depend only on the tracer, not on where the source is, so each is computed once
per lens and the source position substituted -- which is also the pattern to use when projecting a
real catalogue at many transient positions.

The sweep runs on a smaller 6 arcsec grid than the five cases above, which halves it. That is not a
corner cut: ``b``, ``e`` and the position angle are stable to a few parts in 10,000 across grid
sizes from 60x60 at 0.05" to 400x400 at 0.025", and this sweep returns exactly the same 72 verdicts
on the 10 arcsec grid as on the 6 arcsec one. The ``PointSolver``'s own precision is set by
``pixel_scale_precision``, not by the grid it is built on.
"""

sweep_grid = al.Grid2D.uniform(shape_native=(120, 120), pixel_scales=0.05)
sweep_solver = al.PointSolver.for_grid(
    grid=sweep_grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

axis_ratio_list = [0.5, 0.7, 0.85, 0.95]
gamma_list = [0.0, 0.06, 0.15]
misalignment_list = [0.0, 45.0, 90.0]
scale_list = [0.5, 1.02]

mass_angle = 30.0

grid_rows = []

for axis_ratio in axis_ratio_list:
    for gamma in gamma_list:
        for misalignment in misalignment_list:
            tracer = tracer_from(
                axis_ratio=axis_ratio,
                angle=mass_angle,
                gamma=gamma,
                gamma_angle=mass_angle + misalignment,
            )

            caustic = caustic_of(tracer, grid=sweep_grid)

            projection_list = [
                (
                    "caustic",
                    witt_wynne_from_tracer(
                        tracer=tracer, grid=sweep_grid, source_centre=(0.0, 0.0)
                    ),
                ),
                (
                    "vector-sum",
                    witt_wynne_vector_sum(
                        tracer=tracer, grid=sweep_grid, source_centre=(0.0, 0.0)
                    ),
                ),
            ]

            for scale in scale_list:
                source_centre = caustic_point_at_angle(
                    caustic, angle_offset_deg=25.0, scale=scale
                )
                positions_true, lags_true = point_solver_truth(
                    tracer, source_centre, sweep_solver
                )

                for label, model in projection_list:
                    model = replace(model, source=(source_centre[1], source_centre[0]))
                    n_siep, d_position, d_lag = compare(
                        model, positions_true, lags_true
                    )

                    grid_rows.append(
                        dict(
                            axis_ratio=axis_ratio,
                            gamma=gamma,
                            misalignment=misalignment,
                            scale=scale,
                            projection=label,
                            n_true=len(positions_true),
                            n_siep=n_siep,
                            agree=n_siep == len(positions_true),
                            d_position=d_position,
                            d_lag=d_lag,
                        )
                    )


def select(rows, **conditions):
    return [
        row
        for row in rows
        if all(row[key] == value for key, value in conditions.items())
    ]


def agreement(rows) -> str:
    if len(rows) == 0:
        return f"{'-':>13}"
    agreed = sum(row["agree"] for row in rows)
    return f"{agreed:3d}/{len(rows):<3d} ({100.0 * agreed / len(rows):5.1f}%)"


n_cases = len(grid_rows) // 2

print(f"\n__Grid: {n_cases} cases, verdict agreement against the PointSolver__\n")
print(f"{'projection':<12} {'inside (0.5x)':>15} {'outside (1.02x)':>17} {'all':>17}")
print("-" * 64)

for label in ("caustic", "vector-sum"):
    rows = select(grid_rows, projection=label)
    print(
        f"{label:<12} {agreement(select(rows, scale=0.5)):>15}"
        f" {agreement(select(rows, scale=1.02)):>17} {agreement(rows):>17}"
    )

print(f"\n{'':<12} {'inside, by shear':>34} | {'outside, by shear':>34}")
print(f"{'projection':<12}", end="")
for where in ("inside", "outside"):
    for gamma in gamma_list:
        print(f" {'g=' + format(gamma, '.2f'):>10}", end="")
    print("   |" if where == "inside" else "", end="")
print()
print("-" * 92)

for label in ("caustic", "vector-sum"):
    print(f"{label:<12}", end="")
    for scale in scale_list:
        for gamma in gamma_list:
            rows = select(grid_rows, projection=label, scale=scale, gamma=gamma)
            agreed = sum(row["agree"] for row in rows)
            print(f" {f'{agreed}/{len(rows)}':>10}", end="")
        print("   |" if scale == 0.5 else "", end="")
    print()

print(f"\n{'':<12} {'inside, by axis ratio':>46}")
print(f"{'projection':<12}", end="")
for axis_ratio in axis_ratio_list:
    print(f" {'q=' + format(axis_ratio, '.2f'):>10}", end="")
print()
print("-" * 56)

for label in ("caustic", "vector-sum"):
    print(f"{label:<12}", end="")
    for axis_ratio in axis_ratio_list:
        rows = select(grid_rows, projection=label, scale=0.5, axis_ratio=axis_ratio)
        agreed = sum(row["agree"] for row in rows)
        print(f" {f'{agreed}/{len(rows)}':>10}", end="")
    print()

print(f"\n{'projection':<12} {'median dpos':>12} {'max dpos':>10} {'max dlag':>10}")
print("-" * 48)

for label in ("caustic", "vector-sum"):
    rows = [
        row
        for row in select(grid_rows, projection=label, scale=0.5)
        if row["agree"] and np.isfinite(row["d_position"])
    ]
    d_position = np.array([row["d_position"] for row in rows])
    d_lag = np.array([row["d_lag"] for row in rows])
    print(
        f"{label:<12} {np.median(d_position):12.4f} {d_position.max():10.4f}"
        f" {d_lag.max():10.3f}"
    )

print("(inside the caustic only, over the cases whose verdict agreed)")

print("\nhow the outside-caustic disagreements fail:")

for label in ("caustic", "vector-sum"):
    modes = {}
    for row in select(grid_rows, projection=label, scale=1.02):
        if not row["agree"]:
            key = (row["n_true"], row["n_siep"])
            modes[key] = modes.get(key, 0) + 1
    summary = ", ".join(
        f"truth {n_true} reported as {n_siep}: {count}"
        for (n_true, n_siep), count in sorted(modes.items())
    )
    print(f"  {label:<12} {summary or 'none'}")

print(
    "\ncases returning the sentinel verdict -1: "
    + ", ".join(
        f"{label} {sum(row['n_siep'] == N_IMAGES_SENTINEL for row in select(grid_rows, projection=label))}"
        for label in ("caustic", "vector-sum")
    )
)

inside_caustic = select(grid_rows, projection="caustic", scale=0.5)
inside_fraction = sum(row["agree"] for row in inside_caustic) / len(inside_caustic)

assert inside_fraction >= 0.95, (
    f"The caustic-matched projection reproduced the PointSolver verdict on only "
    f"{100.0 * inside_fraction:.1f}% of the inside-caustic grid cases, below the 95% this guide "
    f"claims. If this fires under the automated test harness, check that `ENV: full_datasets` was "
    f"applied: PYAUTO_SMALL_DATASETS short-circuits `PointSolver.solve` to a fixed pair of "
    f"positions."
)

"""
__When the Ellipticity and the Shear Cancel__

The grid above steps over ``gamma`` in 0, 0.06 and 0.15, which steps around the one configuration
that breaks the vector sum outright. At ``q = 0.7`` the potential ellipticity is
``(1 - q) / 3 = 0.1``, so an aligned shear of exactly 0.1 cancels it and the sum is zero. The
caustic-matched projection reads the caustic that model actually has; the vector sum has nothing
left to read, and without the ``E_MINIMUM`` guard would divide by it.
"""

cancelling_tracer = tracer_from(
    axis_ratio=0.7, angle=40.0, gamma=(1.0 - 0.7) / 3.0, gamma_angle=40.0
)

for label, projection in (
    ("caustic", witt_wynne_from_tracer),
    ("vector-sum", witt_wynne_vector_sum),
):
    model = projection(tracer=cancelling_tracer, grid=grid, source_centre=(0.03, 0.05))
    print(
        f"\n{label:<12} valid={model.valid}  e={model.e:.5f}  PA={model.pa_deg:.3f}"
        f"  n_images={model.n_images()}"
    )
    if model.reason:
        print(f"{'':<12} reason: {model.reason}")

"""
__A Lens With MGE Light__

The model a Euclid pipeline actually produces is not a bare SIE: it carries an MGE lens light, a
multi-Gaussian ``Basis``. Because that ``Basis`` is a ``MassProfile`` subclass, selecting the mass
by elimination picks the *light* and mis-projects the lens by tens of degrees without any warning.

The check below builds the same lens twice, with and without the MGE light, and confirms that the
class-based selection picks the ``Isothermal`` and that the two projections are identical.
"""

mge_light = al.lp_basis.Basis(
    profile_list=[
        al.lp.Gaussian(
            centre=(0.15, -0.1),
            ell_comps=(0.05, 0.05),
            intensity=1.0,
            sigma=0.1 * (i + 1),
        )
        for i in range(5)
    ]
)

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.75, angle=40.0),
    einstein_radius=1.2,
)
gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=70.0)
shear = al.mp.ExternalShear(gamma_1=gamma_1, gamma_2=gamma_2)

tracer_light_free = al.Tracer(
    galaxies=[
        al.Galaxy(redshift=z_lens, mass=mass, shear=shear),
        al.Galaxy(redshift=z_source),
    ],
    cosmology=cosmology,
)
tracer_with_light = al.Tracer(
    galaxies=[
        al.Galaxy(redshift=z_lens, bulge=mge_light, mass=mass, shear=shear),
        al.Galaxy(redshift=z_source),
    ],
    cosmology=cosmology,
)

print(f"\n{'tracer':<22} {'picked':<22} {'e':>8} {'b':>8} {'PA':>8}")
print("-" * 72)

mge_models = {}

for name, tracer in (
    ("SIE + shear", tracer_light_free),
    ("+ MGE lens light", tracer_with_light),
):
    picked = [type(profile).__name__ for profile in _mass_and_shear_from(tracer)]
    model = witt_wynne_from_tracer(tracer=tracer, grid=grid, source_centre=(0.03, 0.05))
    mge_models[name] = model
    print(
        f"{name:<22} {', '.join(picked):<22} {model.e:8.5f} {model.b:8.5f}"
        f" {model.pa_deg:8.3f}"
    )

assert mge_models["+ MGE lens light"].pa_deg == mge_models["SIE + shear"].pa_deg
assert mge_models["+ MGE lens light"].e == mge_models["SIE + shear"].e

print(
    f"\nthe MGE lens light changes nothing: the mass profile is selected by class."
    f"\nvalidation runtime: {time.time() - run_start:.1f} s"
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

The five named cases and the 72-case grid tell different stories, and the grid is the one to
believe. Five cases can only show that nothing is obviously broken; they cannot say where the
approximation runs out, and here they happened to land on its good side.

- **Verdict, inside the caustic.** The caustic-matched projection reproduced the ``PointSolver``'s
  4/2/1 answer on **36 of 36** grid cases (100%), the vector sum on **35 of 36** (97.2%), across
  ``q`` from 0.5 to 0.95, shear up to 0.15 and every misalignment. Inside the caustic -- which is
  where a quad lives, and where the broker's question is usually asked -- the projection does not
  cost the verdict.
- **Verdict, outside the caustic.** It does cost it there. Caustic-matched: **32 of 36** (88.9%).
  Vector sum: **16 of 36** (44.4%), i.e. worse than a coin toss. Every failure is the same one, a
  true 2-image system reported as a quad, because the projected astroid is larger than the caustic
  it stands in for and swallows a source that is really outside. The vector sum's failures are
  entirely shear-driven -- 12/12 correct at ``gamma = 0``, 3/12 at 0.06, 1/12 at 0.15 -- while the
  caustic match degrades only slightly (12/12, 10/12, 10/12).
- **Positions.** Over the inside-caustic cases whose verdict agreed, predicted images land a
  **median 0.142 and at most 0.404 arcsec** from the true ones (caustic-matched; 0.167 and 0.413
  for the vector sum). The five named cases' 0.07 to 0.16 arcsec is the good end of that
  distribution, not its typical value. Either way this is roughly a tenth of the image separation:
  close enough to associate a transient with a predicted image unambiguously, and nowhere near
  good enough to use as an astrometric constraint.
- **Lags.** On the same cases the worst lag error is **23.4 days** (caustic-matched) against
  **48.7 days** (vector sum). On the five named cases, whose lag spans run 17 to 71 days, the
  errors are 0.87 to 6.65 days, i.e. 5 to 13% of the span.

**Use the caustic-matched projection.** It is better or equal on every measure above, and it is
better for a structural reason: it reads the tracer's own tangential caustic, so extra deflectors,
a non-isothermal slope and multiple mass components are folded in without any per-term conversion.
The vector sum depends on the density-to-potential factor of three, on getting the shear's sign
right, and on the two not cancelling -- approximations that only hold at small ellipticity and
small shear, which is exactly what its 1/12 at ``gamma = 0.15`` is measuring.

**And check which side of the caustic the source is on.** The verdict for a source outside the
caustic is the one to distrust, from either projection; if the projected model says "2", the source
is near or outside the astroid and the answer is worth re-deriving with the full ``PointSolver``
before it is acted on.

An **independent numerical review on 2026-09-17** reached the same conclusion on a different grid
of 136 cases (``q`` in 0.5-0.95, ``gamma`` in 0-0.15, four misalignments, the same two source
scales): the caustic-matched verdict agreed **68/68 inside the caustic and 51/68 outside** it,
against **66/68 and 22/68** for the vector sum, with a median position error of 0.136 arcsec and a
maximum of 0.398. That review is also the source of the corrections in this guide: the wrong-branch
root filter that lets the 1-image verdict fire, the degeneracy sentinels, and the class-based mass
selection.

The limits of what was tested here: a single deflector, an isothermal slope, ``gamma <= 0.15``,
``q >= 0.5``, and a source described by its centroid alone. A nearby perturber or a significantly
non-isothermal slope were not checked (the review found ``b`` behaves correctly across slopes 1.8
to 2.2, but its outside-caustic verdicts were no better). The astroid fit itself is the binding
approximation at large ellipticity: at ``q = 0.5`` it is 10% short on the long caustic semi-axis
and 11% long on the short one, and by ``q <= 0.35`` those residuals reach -18% and +16%.

One check could not be repeated for this version: the ``.in`` round trip against the compiled
Zenodo C++, because ``SIEP_CLI.v1.0.cpp`` is not present in this workspace. The four-line recipe to
redo it is in the regression section above. If the projection is being used to decide whether a
real transient is a fourth image, re-run the full ``PointSolver`` on the candidate before believing
anything more than the verdict.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

This guide computes tangential caustics and solves the lens equation on a 200x200
grid; SMALL_DATASETS caps grids to 16x16 at 0.6" pixels, which destroys both.

ENV: full_datasets
"""
