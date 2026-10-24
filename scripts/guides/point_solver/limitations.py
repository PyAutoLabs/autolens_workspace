"""
Point Solver: Limitations and Assumptions
=========================================

``PointSolver`` finds the multiple images of a point source by tiling the image plane with triangles, tracing them
to the source plane, and repeatedly halving the triangles that contain the source. It is fast and works inside
``jax.jit``, but it is a numerical search, so it has limits.

This guide puts every known limit and assumption in one place, so you can check them for yourself. Each one is
short, says whether it matters at the default settings, and shows how to test it, with a demo where that is cheap.

__Contents__

- **The Oracle:** An independent way to find every image, used to check the solver.
- **Using The PointSolver:** What each setting does.
- **Assumptions:** Two deliberate choices that are not bugs.
- **Limitations:** Seven known limits, each with how to check it and its status.
- **Wrap Up:** A checklist.
"""

# from autolens import setup_notebook; setup_notebook()

import numpy as np

import autolens as al

"""
__Lens__

A cored elliptical lens with external shear. Its deflections are analytic, so everything below runs in seconds. The
small core gives the lens a faint central image, which we use later.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.dPIEMassB0(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.05), ra=0.15, rs=20.0, b0=1.6
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.02),
)
tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])

source = (0.07, 0.05)

lens_calc = al.LensCalc.from_tracer(tracer=tracer)


def traced(points):
    """Ray-trace (y,x) image-plane points to the source plane."""
    grid = al.Grid2DIrregular(np.asarray(points))
    return np.asarray(tracer.traced_grid_2d_list_from(grid=grid)[-1])


def magnifications(points):
    grid = al.Grid2DIrregular(np.asarray(points))
    return np.asarray(lens_calc.magnification_2d_via_hessian_from(grid=grid))


"""
__The Oracle__

To check a solver you need the true images, found by a method that shares none of its machinery. We call that an
oracle. Ours solves the lens equation ``r(theta) = beta(theta) - beta = 0`` directly:

1. **Seeds.** Sample ``|r|`` on a dense lattice and take every local minimum as a seed.
2. **Newton polish.** Run a damped Newton step from each seed until ``|r|`` stops falling.
3. **Accept.** Keep a point only when ``|r|`` is at round-off (below 1e-10"), then remove duplicates.

It shares only the ray-tracing with the solver: no triangles, no containment test and no magnification filter. A
seed in a false minimum never reaches round-off and is thrown away.
"""


def oracle(beta, half_width=3.0, h=0.02, fd_step=1e-6):
    beta = np.asarray(beta)
    ys = np.arange(-half_width, half_width + h / 2, h)
    yy, xx = np.meshgrid(ys, ys, indexing="ij")
    lattice = np.stack([yy.ravel(), xx.ravel()], axis=1)
    r = np.linalg.norm(traced(lattice) - beta, axis=1).reshape(yy.shape)

    padded = np.pad(r, 1, constant_values=np.inf)
    neighbours = [
        padded[1 + dy : 1 + dy + r.shape[0], 1 + dx : 1 + dx + r.shape[1]]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    ]
    theta = lattice[(r <= np.min(neighbours, axis=0)).ravel()]

    for _ in range(50):
        res = traced(theta) - beta
        jac = np.stack(
            [
                (traced(theta + d) - traced(theta - d)) / (2 * fd_step)
                for d in ([fd_step, 0.0], [0.0, fd_step])
            ],
            axis=-1,
        )
        step = np.linalg.solve(jac, res[..., None])[..., 0]
        step *= np.minimum(1.0, 4 * h / np.linalg.norm(step, axis=1, keepdims=True))
        for _ in range(20):  # Backtrack until |r| falls.
            trial = theta - step
            better = np.linalg.norm(traced(trial) - beta, axis=1) < np.linalg.norm(
                res, axis=1
            )
            theta = np.where(better[:, None], trial, theta)
            step = np.where(better[:, None], 0.0, step / 2)

    roots = []
    for t in theta[np.linalg.norm(traced(theta) - beta, axis=1) < 1e-10]:
        if all(np.linalg.norm(t - q) > 1e-6 for q in roots):
            roots.append(t)
    return np.array(roots)


roots = oracle(beta=source)

print("Oracle images (y, x) and magnification:")
for root, mu in zip(roots, magnifications(roots)):
    print(f"  ({root[0]:+.5f}, {root[1]:+.5f})   mu = {mu:+.3f}")

"""
Now the solver, on a 0.05" grid covering +-2.5":
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

solver = al.PointSolver.for_grid(grid=grid, pixel_scale_precision=0.001)

positions = np.asarray(
    solver.solve(tracer=tracer, source_plane_coordinate=source).array
)

print("\nDistance from each oracle image to the nearest solver image:")
for root, mu in zip(roots, magnifications(roots)):
    distance = np.min(np.linalg.norm(positions - root, axis=1))
    print(f'  mu = {mu:+.3f}   distance = {distance:.5f}"')

"""
The four bright images agree to well under 0.001". The central image (|mu| below 0.1) is not returned. That is
deliberate; see **Assumptions** below.

This inline oracle is a sketch. The full version is ``autolens_profiling/scripts/cluster/image_plane/_oracle.py``.
It adds sign-change seeds, refinement around cusps and folds, a re-run at half the lattice spacing, and a calibration
against analytic image positions (reproduced to 2.84e-8"). It judged the cluster sweep quoted below (RAL job
412572).

__Using The PointSolver__

``for_grid`` takes its extent and starting triangle size from a grid (``for_limits_and_scale`` takes them directly).
``solve`` returns the images of one source-plane coordinate as a ``Grid2DIrregular``:

- ``grid``: the triangles start at ``grid.pixel_scale`` (the ``scale``) and cover the grid's extent. Images outside
  it are never found.
- ``pixel_scale_precision``: the target size of the final triangles.
- ``magnification_threshold`` (default 0.1): images with ``|mu|`` at or below this are dropped.
- ``residual_threshold`` (default 10.0): candidates that trace further than
  ``residual_threshold x pixel_scale_precision`` from the source are dropped. ``None`` turns this off.
- ``neighbor_degree`` (default 1): how many rings of neighbouring triangles are kept around each containing triangle
  before the next halving.
- ``plane_redshift`` (a ``solve`` argument): the redshift of the plane the coordinate lives in, for multi-plane
  lenses. ``None`` means the last plane.
- ``use_jax``: run the solve in JAX, for use inside ``jax.jit``.

The search runs ``n_steps = ceil(log2(scale / pixel_scale_precision))`` steps: the first uses the starting
triangles and each later step halves them. The final triangle edge is ``h = scale / 2 ** (n_steps - 1)``, which
is always more than ``pixel_scale_precision`` and at most twice it.
"""
h = solver.scale / 2 ** (solver.n_steps - 1)

print(f"\nscale = {solver.scale}, n_steps = {solver.n_steps}, h = {h}")

"""
__Assumptions__

**1. Demagnified images are dropped on purpose.** Images with ``|mu|`` at or below ``magnification_threshold``
(default 0.1) are removed. They are far too faint to be seen, so a model of real data should not predict them.
Lower the threshold only to study or validate the solver, or when a central image really is observed:
"""
solver_all = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.01
)
positions_all = np.asarray(
    solver_all.solve(tracer=tracer, source_plane_coordinate=source).array
)

print(
    f"\nImages at threshold 0.1: {len(positions)}, at threshold 0.01: {len(positions_all)}"
)

"""
**2. Positions are triangle centres, not polished roots.** Each image is the centre of a final triangle, so it is
accurate to about ``h``. The comparison with the oracle above shows this. Close to a critical curve the error can be
larger, because there the source position barely changes as the image moves. Near a cusp, centres sat up to about
50 times ``pixel_scale_precision`` from the true image
(``autolens_workspace_test/scripts/point_source/solver/DUPLICATE_POLICY.md``).

__Limitations__

**a. The starting grid can miss images.** The first step keeps only starting triangles whose traced corners
enclose the source. Over a large triangle the lens mapping is far from linear, so a triangle holding an image can
fail that test, and the image is lost for good. Images close to a mass centre, and close pairs of bright images, are
most at risk. Here a 1.0" grid misses an image that a 0.5" grid finds:
"""
source_far = (0.35, 0.0)

for pixel_scale in (1.0, 0.5):
    shape = int(round(6.0 / pixel_scale))
    coarse_grid = al.Grid2D.uniform(
        shape_native=(shape, shape), pixel_scales=pixel_scale
    )
    found = al.PointSolver.for_grid(
        grid=coarse_grid, pixel_scale_precision=0.001
    ).solve(tracer=tracer, source_plane_coordinate=source_far)
    print(f'\nStarting grid {pixel_scale}": {len(found)} images found')

bright = np.abs(magnifications(oracle(beta=source_far))) > 0.1
print(f"Oracle: {bright.sum()} images with |mu| > 0.1")

"""
- *At the defaults?* Yes. In the cluster sweep the production 1.0" grid missed 5 of 300 stress-test images (0.5"
  and 0.7" missed 2); every miss was one of a close, highly magnified pair. On the autolens_visualization cluster
  model, an image with ``mu = -0.89``, 0.39" from a member's core, is missed at every starting grid down to 0.3";
  only 0.2" finds it. No single starting grid finds every image.
- *How to check:* run the oracle above on your best-fit model, or re-solve on a finer grid and compare image counts.
- *Status:* the 1.0" grid was kept as the default on 2026-10-23, and a validation check is planned. Evidence:
  ``autolens_profiling/wiki/campaigns/cluster/image_plane/cluster_pointsolver.md``.

**b. Images outside the grid are never found.** The triangles only cover the grid you pass in. A grid cropped to the
bounding box of the observed images lost 49-63 images in the cluster sweep.
"""
small_grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.05)

found = al.PointSolver.for_grid(grid=small_grid, pixel_scale_precision=0.001).solve(
    tracer=tracer, source_plane_coordinate=source
)
print(f'\n+-1.0" grid: {len(found)} images found, +-2.5" grid: {len(positions)}')

"""
- *At the defaults?* Yes, whenever the grid is too small. Make it comfortably larger than the outermost image.

**c. Spurious images at a singular mass centre.** At the centre of a mass profile with no core (for example a
``dPIEMassSph`` member with ``r_core=0``), the deflection jumps. A triangle around that centre can contain the
source at every step without containing an image. Inside a near-critical host its ``|mu|`` is about 0.1-0.3, so
the magnification filter keeps it. ``residual_threshold`` removes it: a true image traces back to the source, but
this point misses by about the member's Einstein radius. This is the fixture from
``PyAutoLens/test_autolens/point/triangles/test_solver_spurious_core.py``:
"""
host = al.Galaxy(
    redshift=0.5,
    mass=al.mp.dPIEMassSph(
        centre=(0.0, 0.0),
        sigma=480.0,
        r_core=2.0,
        r_cut=50.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)
member = al.Galaxy(
    redshift=0.5,
    mass=al.mp.dPIEMassSph(
        centre=(0.3, 0.2),
        sigma=60.0,
        r_core=0.0,
        r_cut=2.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)
core_tracer = al.Tracer(galaxies=[host, member, al.Galaxy(redshift=1.0)])
core_source = (-0.0165, -0.011)
core_grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1)

for precision in (0.001, 0.01):
    for residual_threshold in (None, 10.0):
        found = al.PointSolver.for_grid(
            grid=core_grid,
            pixel_scale_precision=precision,
            residual_threshold=residual_threshold,
        ).solve(tracer=core_tracer, source_plane_coordinate=core_source)
        print(
            f"\nprecision {precision}, residual_threshold {residual_threshold}: "
            f"{len(found)} images"
        )

"""
At precision 0.001 the filter removes the spurious point and keeps the two true images. At 0.01 the cut (0.1") is
larger than the spurious point's miss (0.067"), so it survives.

- *At the defaults?* The filter is on by default (PyAutoLens#800). It is weaker at coarse precision: at 0.005 the
  margin is about 1.3 times, and at 0.01 spurious points survive. Use 0.001 or lower ``residual_threshold``.
- *Side effect:* it also removes true demagnified central images of very elliptical, small-core profiles, which
  miss by about 11 times the precision. Their ``|mu|`` is about 0.01-0.02, so this only matters if you lower
  ``magnification_threshold`` below about 0.02.
- *Status:* see the ``residual_threshold`` docstring in ``autolens/point/solver/shape_solver.py`` and
  PyAutoLens#799.

**d. Images very close to a zero-core member can be missed.** Of 324 true images with ``|mu| > 0.1`` within 0.03" of
a ``r_core=0`` member, the triangle search found only 38-43, before any filter runs.

- *At the defaults?* Yes, but the practical impact is low. These images sit inside the member's own light, and a
  zero core is an idealisation.
- *Status:* under investigation: polishing candidates with Newton steps, or refining the triangles around singular
  centres.

**e. Duplicates, and images closer than ``h``.** The solver returns one centre per
containing triangle and never groups them. An image on the edge between two triangles can be returned twice,
depending on round-off. For an isothermal lens with shear and a source at the centre (the fixture of
``autolens_workspace_test/scripts/point_source/solver/error_audit.py``), a 0.05" starting grid returns six rows for
four images:
"""
quad_tracer = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6),
            shear=al.mp.ExternalShear(gamma_1=0.1, gamma_2=0.0),
        ),
        al.Galaxy(redshift=1.0),
    ]
)
quad_solver = al.PointSolver.for_limits_and_scale(
    y_min=-4.97,
    y_max=5.03,
    x_min=-4.97,
    x_max=5.03,
    scale=0.05,
    pixel_scale_precision=0.001,
)

# The lens centre is a lattice point, so silence NumPy's divide-by-zero warning there.
with np.errstate(divide="ignore", invalid="ignore"):
    quad_positions = quad_solver.solve(
        tracer=quad_tracer, source_plane_coordinate=(0.0, 0.0)
    )

print(f"\nFour true images, {len(quad_positions)} rows returned:")
print(np.round(np.asarray(quad_positions.array), 5))

"""
The extra rows are within ``pixel_scale_precision`` of each other. Near a cusp, NumPy returned up to 86 rows for four
true images, many of them several precisions from any image.

- *At the defaults?* Yes. The default fit, ``FitPositionsImagePairAll``, sums over every model image, so a
  duplicate costs only a small factor (see ``guides/point_solver/pairing.py``). Counting images directly is less
  safe.
- *Status:* three grouping rules were tested and none was adopted, because true images near a cusp can be closer
  than ``h`` (``autolens_workspace_test/scripts/point_source/solver/DUPLICATE_POLICY.md``, ``RESULTS.md``).

**f. JAX.** With ``use_jax=True`` the solve has a fixed output shape:

- The output is padded with ``inf`` rows. Drop them with
  ``positions[~np.isinf(positions).any(axis=1)]`` (or ``remove_infinities=True`` outside ``jax.jit``).
- Each refinement step keeps at most ``MAX_CONTAINING_SIZE`` (20, in ``autoarray.structures.triangles.array``)
  containing triangles, and drops the rest without warning. Near a cusp this lost a true image that NumPy found.
  The cluster sweep never exceeded 7.
- Gradients come from implicit differentiation of the lens equation at each returned position
  (``autolens/point/solver/implicit_diff.py``). This is exact only at true images away from critical curves. It diverges
  near a critical curve, and the likelihood jumps whenever the image count changes. A ``Tracer`` that is not
  registered for JAX falls back to a plain solve with zero gradient.

**g. Multi-plane lenses.** ``plane_redshift`` must be the redshift of the plane the source coordinate lives in.
The magnification and residual filters use the same plane, so they agree with the search (PyAutoLens#480). A redshift
that matches no plane raises a ``ValueError``; leaving it as ``None`` means the last plane.

__Wrap Up__

Before trusting the images of a model, especially a cluster:

- Is the grid larger than every image, and is the starting grid fine enough? Re-solve on a finer grid and compare.
- Does the oracle find the same bright images?
- Are you using ``pixel_scale_precision`` 0.001 or finer, so ``residual_threshold`` works?
- Are duplicates or a lowered ``magnification_threshold`` changing the image count?
- On JAX, have you removed the ``inf`` rows?

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Every demo here is a real ``PointSolver.solve``. SMALL_DATASETS short-circuits
the solve to a fixed pair of positions, so the printed numbers would no longer
match the prose. The script runs in about 15 seconds on a laptop CPU.

ENV: full_datasets
"""
