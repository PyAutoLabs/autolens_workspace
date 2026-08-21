"""
Modeling: Datacube — Delaunay Source
====================================

This script fits a datacube — a list of `Interferometer` channels — with a single shared lens model and a
per-channel **Delaunay-pixelized** source reconstruction. It is the Delaunay sibling of `modeling.py`, which
fits the same cube with a `RectangularRTUAdaptDensity` mesh.

A Delaunay mesh adapts the source-plane reconstruction to the lensed source's morphology more flexibly than a
rectangular mesh: source pixels are placed via a triangulation of (y, x) image-plane points that are ray-traced
into the source plane for each candidate lens model. This gives more pixels to the highly-magnified parts of
the source-plane, which is usually where the emission-line signal is concentrated.

For ALMA datacubes Delaunay is often the right default once the simpler rectangular fit has converged on a
sensible lens model — the canonical workflow is rectangular for the global fit, Delaunay (or a more adaptive
image-mesh like `Hilbert`) for the source-science follow-up. This script demonstrates the wiring.

The FactorGraph wiring is identical to `modeling.py`: shared lens, per-channel inversion. Only the source-plane
mesh (and its regularization) changes.

__Contents__

- **Mask:** Define the 2D real-space mask applied to every channel.
- **Dataset:** Where the per-channel cube lives on disk and how to point this script at your own.
- **Dataset Auto-Simulation:** Run `simulator.py` automatically if the cube isn't already on disk.
- **Dataset Loading:** Loop over the channel folders and load each as an `Interferometer` object.
- **Sparse Operators:** Pre-compute per-channel sparse-operator matrices used by the Delaunay inversion.
- **Positions:** Load the cube's multiple-image positions and build a shared `PositionsLH` penalty.
- **Settings:** Disable the positive-only solver so visibility-space inversions can take negative pixel values.
- **Image Mesh:** Build the image-plane mesh of (y, x) points that get ray-traced and Delaunay-triangulated.
- **Edge Zeroing:** Append a ring of edge pixels so the source-plane reconstruction zeroes at the mesh boundary.
- **Adapt Images:** Pair the image-plane mesh with the source galaxy via `al.AdaptImages`.
- **Model:** Compose the shared `Isothermal + ExternalShear` lens and Delaunay-pixelized source.
- **Per-Channel Analyses:** One `AnalysisInterferometer` per channel, with shared `adapt_images` and `PositionsLH`.
- **FactorGraph:** Wrap each analysis in an `AnalysisFactor` and combine via `af.FactorGraphModel`.
- **Search:** Configure the `Nautilus` non-linear search.
- **Model Fit:** Run the fit. Per-channel cost is comparable to the rectangular variant.
- **Wrap Up:** Pointers to `modeling.py`, `start_here.py`, and the JAX likelihood walkthrough.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import subprocess
import sys
from pathlib import Path

import autofit as af
import autolens as al

"""
__Mask__
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

The datacube lives under `dataset/interferometer/datacube/<dataset_name>/`, with one subfolder per channel.
"""
dataset_label = "datacube"
dataset_name = "sim_simple"
dataset_path = Path("dataset") / "interferometer" / dataset_label / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    subprocess.run(
        [sys.executable, "scripts/interferometer/features/datacube/simulator.py"],
        check=True,
    )

"""
__Dataset Loading__
"""
channel_paths = sorted(
    p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("channel_")
)
print(f"Loading {len(channel_paths)} channels from {dataset_path}")

dataset_list = [
    al.Interferometer.from_fits(
        data_path=channel_path / "data.fits",
        noise_map_path=channel_path / "noise_map.fits",
        uv_wavelengths_path=channel_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,
    )
    for channel_path in channel_paths
]

"""
__Sparse Operators__
"""
dataset_list = [
    dataset.apply_sparse_operator(use_jax=True, show_progress=False)
    for dataset in dataset_list
]

"""
__Positions__

Multiple-image positions + `PositionsLH` are essential for Delaunay fits — without them, the search routinely
finds demagnified-source local maxima.
"""
positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))
positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
__Settings__
"""
settings = al.Settings(use_positive_only_solver=False)

"""
__Image Mesh__

The Delaunay mesh is built by ray-tracing (y, x) coordinates from the image-plane to the source-plane and
triangulating the source-plane points. We start with an `Overlay` image-mesh: a regular grid of points spread
across the image-plane mask. This has a mild adaptive effect — regions of high lens magnification receive more
source pixels once they are ray-traced. For more aggressive adaptation, swap `Overlay` for `Hilbert` (which
weights points by the source's surface brightness).

The number of `pixels` passed to `al.mesh.Delaunay` must equal the number of points in `image_plane_mesh_grid`,
because JAX uses `pixels` to define static-shape arrays.
"""
image_mesh = al.image_mesh.Overlay(shape=(26, 26))
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=real_space_mask)

"""
__Edge Zeroing__

Pixels at the edge of the source-plane mesh are forced to zero brightness during the inversion to prevent
unphysical solutions where edge pixels reconstruct bright surface brightnesses (often by absorbing residuals).
For a Delaunay mesh we manually add a ring of edge points to the image-plane mesh and tell `al.mesh.Delaunay`
how many of those trailing points to zero.
"""
edge_pixels_total = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=real_space_mask.mask_centre,
    radius=mask_radius + real_space_mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

"""
__Adapt Images__

The image-plane mesh is passed into modeling via `al.AdaptImages`, keyed on the source galaxy's path in the
model. The same `adapt_images` object is reused for every channel because the lens model and image-plane mesh
are shared.
"""
adapt_images = al.AdaptImages(
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid,
    },
)

"""
__Model__

Shared `Isothermal + ExternalShear` lens + Delaunay-pixelized source. `ConstantSplit` is the canonical Delaunay
regularizer (split-prior on inner-vs-edge mesh pixels). The mesh `pixels` is fixed at the number of points in
`image_plane_mesh_grid`, which is `26*26 + 30` after the edge-ring append.
"""
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

mesh = af.Model(
    al.mesh.Delaunay,
    pixels=image_plane_mesh_grid.shape[0],
    zeroed_pixels=edge_pixels_total,
)
regularization = af.Model(al.reg.ConstantSplit)
pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__Per-Channel Analyses__

Each `AnalysisInterferometer` receives the same `adapt_images` (image-plane mesh is shared) and the same
`positions_likelihood` (lens model is shared via the FactorGraph).
"""
analysis_list = [
    al.AnalysisInterferometer(
        dataset=dataset,
        settings=settings,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood],
        use_jax=True,
    )
    for dataset in dataset_list
]

"""
__FactorGraph__
"""
analysis_factor_list = [
    af.AnalysisFactor(prior_model=model.copy(), analysis=analysis)
    for analysis in analysis_list
]

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

print(f"  channels in factor graph:           {len(analysis_factor_list)}")
print(
    f"  global model free parameters:       {factor_graph.global_prior_model.total_free_parameters}"
)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "datacube",
    name="delaunay",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
    iterations_per_quick_update=50000,
)

"""
__Model Fit__

Per-channel cost is dominated by the Delaunay inversion + the per-channel NUFFT — comparable to the rectangular
variant in `modeling.py`. On CPU expect this to take a few hours for the 4-channel reference cube; on GPU,
tens of minutes.
"""
print(
    """
    The non-linear search has begun running.

    Per-channel inversions multiply the per-likelihood cost — expect this to take longer than a single-channel
    interferometer fit.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Wrap Up__

For the rectangular variant, see `modeling.py`. For the parametric variant, see `modeling_parametric.py`. For
the narrative walkthrough, see `start_here.py`. For a step-by-step JAX likelihood walkthrough, see
`likelihood_function.py`.
"""
