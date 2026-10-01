"""
Modeling: Datacube (Array-Free)
===============================

This script fits the same datacube as `modeling.py` — a list of per-channel `Interferometer` datasets sharing one
lens model through an `af.FactorGraphModel` — but builds every channel as an **array-free** dataset.

An array-free `Interferometer` never holds the visibilities. Instead of loading a channel's `data`, `noise_map`
and `uv_wavelengths` into memory, you stream them in chunks into `al.Interferometer.from_stream`, which reduces
each chunk on the fly into the handful of quantities a sparse (w-tilde) pixelized inversion actually reads:

 - the dirty image and the dirty beam,
 - the NUFFT precision operator (the real-space representation of `W̃`),
 - and a few scalars (the data term `dᵀ W d`, the noise normalization, the sum of weights and `n_vis`).

These live in an `al.SparseTerms` record (`dataset.sparse_terms`), whose size is set by the real-space mask, not
by the number of visibilities. Peak memory is set by the chunk size, so memory stays **flat in N_vis**: a cube
with 2e8 visibilities — the scale of a full ALMA spectral-line cube, far too large to hold as complex arrays
alongside the transformer — costs the same memory to fit as the ~190-visibility SMA cube used here. Only the time
spent streaming the chunks grows with N_vis, and that is paid once, before the fit.

On the resulting dataset `dataset.data`, `dataset.noise_map`, `dataset.uv_wavelengths` and `dataset.transformer`
are all `None` and `dataset.is_array_free` is `True`. Everything a modeling run needs still works:

 - Fits with pixelized sources, linear light profiles and ordinary (non-linear) light profiles.
 - `jax.jit` of the likelihood (`use_jax=True`), exactly as in `modeling.py`.
 - Result output, `search.fit` resume and aggregator reload.
 - Visualization: the visualizer plots the naturally weighted dirty images (`dirty_image_natural`) and the
   dirty beam in place of the visibility-space panels, which need the visibilities and are skipped.

Quantities that genuinely need the visibilities (the residual and chi-squared visibilities of a fit, the
unweighted `dirty_image`, `amplitudes`, ...) raise a clear `DatasetException` rather than returning garbage.

Read `modeling.py` first: this script mirrors it step for step and only the dataset construction changes. It then
shows two things array-free terms make easy — a multi-frequency-synthesis (MFS) dataset formed by summing the
per-channel terms, and a phase-centre shift applied while streaming.

__Contents__

- **Mask:** The 2D real-space mask the sparse terms are accumulated on.
- **Dataset:** Where the per-channel cube lives on disk.
- **Dataset Auto-Simulation:** Run `simulator.py` automatically if the cube isn't already on disk.
- **Streaming Chunks:** A generator reading each channel's FITS files memory-mapped, two chunks at a time.
- **Array-Free Datasets:** `al.Interferometer.from_stream` per channel — no visibilities are kept.
- **Positions:** The `al.PositionsLH` penalty, as in `modeling.py`.
- **Settings:** Disable the positive-only solver, as in `modeling.py`.
- **Mesh Shape:** The same 14 x 14 `RectangularBilinearAdaptDensity` mesh.
- **Model:** The same shared lens and per-channel pixelized source.
- **Per-Channel Analyses:** One `AnalysisInterferometer` per array-free channel.
- **FactorGraph:** Combine the channels via `af.FactorGraphModel`.
- **Search:** `Nautilus`, with fewer live points than `modeling.py`.
- **Model Fit:** Run the fit, exactly as in `modeling.py`.
- **Multi-Frequency Synthesis:** Sum the per-channel `SparseTerms` into one MFS dataset and evaluate it.
- **Phase Centre:** Re-stream a channel with `phase_centre=` and show the dirty image re-centre.
- **Wrap Up:** Summary.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

import autofit as af
import autolens as al

"""
__Mask__

As in `modeling.py`, every channel uses the same `real_space_mask`. For an array-free dataset the mask matters
more than usual: the sparse terms are *accumulated on this mask* while streaming, so the dataset is tied to it.
You cannot re-mask an array-free dataset later — to change the mask, stream the chunks again. The MFS dataset
below is built with `al.Interferometer.from_sparse_terms`, which checks that the mask it is given matches the one
the terms were accumulated on.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

The same reference cube as `modeling.py`: one `channel_NNN/` folder per channel under
`dataset/interferometer/datacube/<dataset_name>/`, each holding `data.fits`, `noise_map.fits` and
`uv_wavelengths.fits` of shape `(n_vis, 2)`.
"""
dataset_label = "datacube"
dataset_name = "sim_simple"
dataset_path = Path("dataset") / "interferometer" / dataset_label / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    subprocess.run(
        [sys.executable, "scripts/interferometer/features/datacube/simulator.py"],
        check=True,
    )

channel_paths = sorted(
    p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("channel_")
)
print(f"Streaming {len(channel_paths)} channels from {dataset_path}")

"""
__Streaming Chunks__

`from_stream` takes any iterable of `(uv_wavelengths, data, noise_map)` triples, where each element holds the
same `K` visibilities: `uv_wavelengths` is a real `(K, 2)` array of baselines in wavelengths, and `data` and
`noise_map` are real `(K, 2)` arrays of (real, imaginary) columns (complex `(K,)` arrays or `al.Visibilities`
also work). The real and imaginary noise sigma of each visibility must be equal. `K` may differ between chunks.

The generator below opens each channel's FITS files memory-mapped (`memmap=True`) and yields one slice at a time,
so only the current chunk is ever read into memory. We split every channel into just 2 chunks to demonstrate the
mechanics — the SMA cube here is tiny. For a real ALMA cube pick `n_chunks` so that one chunk fits comfortably in
memory (e.g. a few million visibilities per chunk); the result does not depend on the chunking, because every
sparse term is a plain sum over visibilities.

The same pattern works for any on-disk layout: a CASA measurement set read with `casatools`, the 3D-FITS cube of
`data_preparation.py` sliced along its channel axis, or an HDF5 file — anything you can read a slice of.
"""


def channel_chunks(channel_path: Path, n_chunks: int = 2):
    """
    Yield `(uv_wavelengths, data, noise_map)` chunks of one channel, read memory-mapped from its FITS files.
    """
    with fits.open(
        channel_path / "uv_wavelengths.fits", memmap=True
    ) as uv_hdul, fits.open(
        channel_path / "data.fits", memmap=True
    ) as data_hdul, fits.open(
        channel_path / "noise_map.fits", memmap=True
    ) as noise_hdul:
        uv_wavelengths = uv_hdul[0].data
        data = data_hdul[0].data
        noise_map = noise_hdul[0].data

        n_vis = uv_wavelengths.shape[0]
        bounds = np.linspace(0, n_vis, n_chunks + 1).astype(int)

        for i0, i1 in zip(bounds[:-1], bounds[1:]):
            yield (
                np.asarray(uv_wavelengths[i0:i1], dtype="float64"),
                np.asarray(data[i0:i1], dtype="float64"),
                np.asarray(noise_map[i0:i1], dtype="float64"),
            )


"""
__Array-Free Datasets__

One `al.Interferometer.from_stream` call per channel. It consumes the generator once, accumulating the
`SparseTerms` chunk by chunk with a NUFFT (`al.TransformerNUFFT`), and returns a dataset that already carries its
sparse operator — there is no separate `apply_sparse_operator` step as in `modeling.py`.

`from_stream` accepts `use_jax=True` to accumulate on the JAX backend (worthwhile on a GPU for very large cubes);
the default NumPy accumulation is used here. The fit itself still runs through `jax.jit` below.
"""
dataset_list = [
    al.Interferometer.from_stream(
        channel_chunks(channel_path),
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,
    )
    for channel_path in channel_paths
]

for channel_path, dataset in zip(channel_paths, dataset_list):
    print(
        f"  {channel_path.name}: is_array_free={dataset.is_array_free}, "
        f"n_vis={dataset.sparse_terms.n_vis}, data={dataset.data}, "
        f"uv_wavelengths={dataset.uv_wavelengths}"
    )

"""
__Positions__

Identical to `modeling.py`: the cube's multiple-image positions wrapped in an `al.PositionsLH` penalty, which for
pixelized fits is essentially required to avoid demagnified-source local maxima.
"""
positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))
positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
__Settings__

Interferometer pixelizations disable the positive-only inversion solver, as in `modeling.py`.
"""
settings = al.Settings(use_positive_only_solver=False)

"""
__Mesh Shape__

The same 14 x 14 `RectangularBilinearAdaptDensity` mesh as `modeling.py`.
"""
mesh_pixels_yx = 14
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Model__

The same model as `modeling.py`: a shared `Isothermal` lens with an `ExternalShear` `MassField`, and a
pixelized source reconstructed independently per channel.

The source could equally be a parametric light profile — ordinary light profiles such as `al.lp.Sersic` and
linear light profiles such as `al.lp_linear.Sersic` both fit array-free datasets (see `modeling_parametric.py`
for that model; swap its dataset loading for the streaming above).
"""
# Lens:
mass = af.Model(al.mp.Isothermal)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source (pixelization, no free priors):
mesh = af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape)
regularization = af.Model(al.reg.Constant)
pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall lens model:
field = af.Model(al.MassField, redshift=0.5, shear=af.Model(al.mp.ExternalShear))

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

print(model.info)

"""
__Per-Channel Analyses__

One `AnalysisInterferometer` per array-free channel, exactly as in `modeling.py` — including `use_jax=True` and
`shared_preloads=True` (valid here for the same reason: the simulated cube has identical `uv_wavelengths` and
noise in every channel, so the curvature matrix is channel-invariant).
"""
analysis_list = [
    al.AnalysisInterferometer(
        dataset=dataset,
        settings=settings,
        positions_likelihood_list=[positions_likelihood],
        use_jax=True,
        shared_preloads=True,
    )
    for dataset in dataset_list
]

"""
__FactorGraph__

Identical to `modeling.py`: every prior is identified across the factors, so the global model has the
dimensionality of the single-channel model and the factor graph sums the per-channel log-evidences.
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

`Nautilus`, as in `modeling.py`, with its own `name` so the two fits write to separate output folders. We use
`n_live=50` rather than `modeling.py`'s 100 so this example (whose main purpose is the dataset construction)
completes in under a quarter of an hour on a laptop CPU; increase it for a production fit of your own cube.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "datacube",
    name="modeling_array_free",
    unique_tag=dataset_name,
    n_live=50,
    n_batch=20,
    iterations_per_quick_update=50000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Model Fit__

The fit is run exactly as in `modeling.py`. Because the sparse inversion reads only the `SparseTerms`, each
likelihood evaluation costs the same as on the in-memory dataset — array-free changes the memory, not the
per-likelihood runtime.

The results are output and can be reloaded with the aggregator like any other fit; the visualizer shows the
naturally weighted dirty images of each channel instead of the visibility-space panels.
"""
result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

for channel_path, result in zip(channel_paths, result_list):
    print(
        f"  {channel_path.name}: max log likelihood = {result.max_log_likelihood_fit.log_likelihood:.4f}"
    )

"""
__Multi-Frequency Synthesis__

Every field of a `SparseTerms` record is a sum over visibilities, so the terms of several channels accumulated
separately on the same mask *add up* to the terms of all their visibilities accumulated together. Summing the
per-channel terms therefore gives the multi-frequency-synthesis (MFS) dataset — the continuum image you would get
by gridding every channel's visibilities into one dataset — without ever streaming the cube a second time.

`sum(...)` works directly on a list of terms. Terms only add when they were accumulated on the same mask (shape,
pixel scales, origin), with the same NUFFT precision and the same phase centre; anything else raises.
`al.Interferometer.from_sparse_terms` then turns the summed terms into an ordinary array-free dataset.
"""
mfs_terms = sum(dataset.sparse_terms for dataset in dataset_list)

mfs_dataset = al.Interferometer.from_sparse_terms(
    mfs_terms, real_space_mask=real_space_mask
)

n_vis_channels = sum(dataset.sparse_terms.n_vis for dataset in dataset_list)

print(
    f"  MFS n_vis = {mfs_dataset.sparse_terms.n_vis} "
    f"(sum over channels = {n_vis_channels})"
)
assert mfs_dataset.sparse_terms.n_vis == n_vis_channels

"""
The MFS dataset can be fitted like any other: `al.AnalysisInterferometer(dataset=mfs_dataset, ...)` with the
model and search above gives a single-dataset fit of the whole cube's continuum. Here we keep the runtime down by
instead evaluating a `FitInterferometer` of the MFS dataset at a fixed model, and comparing it to the same
evaluation on each channel.

The fixed model is the true lens model the simulator used (read from channel 0's `tracer.json`) with the
pixelized source of the model above and a regularization coefficient of 1.0. A fixed, well-conditioned point keeps
the comparison deterministic; at very large regularization coefficients (which a search on this low
signal-to-noise cube can wander to) the regularization matrix's log determinant becomes ill-conditioned.

The MFS log evidence is not the sum of the channel log evidences: the MFS fit reconstructs *one* source from all
the visibilities, whereas the channel fits reconstruct one source per channel — which is exactly why a datacube
fit, not an MFS fit, is what captures a source whose morphology changes across the emission line.
"""
true_tracer = al.from_json(file_path=channel_paths[0] / "tracer.json")

tracer = al.Tracer(
    galaxies=[
        true_tracer.galaxies[0],
        al.Galaxy(
            redshift=1.0,
            pixelization=al.Pixelization(
                mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
                regularization=al.reg.Constant(coefficient=1.0),
            ),
        ),
    ],
    fields=true_tracer.fields,
)

channel_log_evidences = [
    al.FitInterferometer(dataset=dataset, tracer=tracer, settings=settings).log_evidence
    for dataset in dataset_list
]

mfs_log_evidence = al.FitInterferometer(
    dataset=mfs_dataset, tracer=tracer, settings=settings
).log_evidence

for channel_path, log_evidence in zip(channel_paths, channel_log_evidences):
    print(f"  {channel_path.name} log evidence = {log_evidence:.4f}")
print(f"  sum of channel log evidences = {sum(channel_log_evidences):.4f}")
print(f"  MFS log evidence             = {mfs_log_evidence:.4f}")

"""
__Phase Centre__

`from_stream` (and `sparse_terms_from_chunks`) can shift the phase centre while streaming:
`phase_centre=(y0, x0)` (arcseconds, the same `(y, x)` order as a mask `origin`) multiplies every chunk's
visibilities by `exp(+2πi (u x0 + v y0))`, so emission at `(y0, x0)` lands at the image origin. This is how you
re-centre a cube whose correlator phase centre is offset from the lens, without rewriting the visibilities on disk:
the shift is applied to each chunk as it streams past.

Only the dirty image changes. The precision operator, dirty beam and scalars depend on the baselines and noise
alone and are bit-identical to the unshifted ones. The shift is recorded as provenance in
`sparse_terms.phase_centre` (`(0.0, 0.0)` when no shift is applied).

To show the shift, we locate the brightest pixel of channel 0's naturally weighted dirty image — the peak of the
lensed arc — and re-stream channel 0 with `phase_centre` set to it, so the arc's peak moves to the image origin.
(The simulator's source centre, recorded in `cube_summary.json`, is a *source-plane* position: lensing spreads the
emission around the Einstein ring, so it is not where the image-plane emission peaks.)

The mask has an even number of pixels per side, so the origin is a pixel *corner*. To keep the shifted image
sampled on the same pixel lattice, we shift by the peak minus half a pixel: the arc's peak then lands on the pixel
centre just above and to the right of the origin, and that pixel of the shifted dirty image carries exactly the
unshifted peak value.
"""
with open(dataset_path / "cube_summary.json") as f:
    cube_summary = json.load(f)

print(
    f"  channel 0 source-plane centre (cube_summary.json): "
    f"{cube_summary.get('channel_centres', [None])[0]}"
)

image_plane_grid = np.asarray(real_space_mask.derive_grid.unmasked)
half_pixel = 0.5 * real_space_mask.pixel_scales[0]


def dirty_peak_from(dataset):
    """The (y, x) arcsec coordinate and value of the brightest pixel of the natural dirty image."""
    dirty_image = np.asarray(dataset.dirty_image_natural)
    index = int(np.argmax(dirty_image))
    return tuple(float(c) for c in image_plane_grid[index]), float(dirty_image[index])


peak_unshifted, peak_value_unshifted = dirty_peak_from(dataset_list[0])

phase_centre = (peak_unshifted[0] - half_pixel, peak_unshifted[1] - half_pixel)

dataset_shifted = al.Interferometer.from_stream(
    channel_chunks(channel_paths[0]),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
    phase_centre=phase_centre,
)

peak_shifted, peak_value_shifted = dirty_peak_from(dataset_shifted)

target_index = int(
    np.argmin(
        np.hypot(
            image_plane_grid[:, 0] - half_pixel, image_plane_grid[:, 1] - half_pixel
        )
    )
)
value_at_target = float(np.asarray(dataset_shifted.dirty_image_natural)[target_index])

print(
    f"  unshifted dirty-image peak (y, x) = {peak_unshifted}, value = {peak_value_unshifted:.6f}"
)
print(
    f"  phase_centre                      = {dataset_shifted.sparse_terms.phase_centre}"
)
print(
    f"  shifted dirty-image peak (y, x)   = {peak_shifted}, value = {peak_value_shifted:.6f}"
)
print(
    f"  shifted value at {tuple(float(c) for c in image_plane_grid[target_index])} = {value_at_target:.6f} "
    f"(the unshifted peak, moved next to the origin)"
)

assert np.isclose(value_at_target, peak_value_unshifted, rtol=1e-6)

"""
Terms with different phase centres describe images on different coordinate origins, so they cannot be summed —
including a shifted channel with an unshifted one. Shift every channel to the same `phase_centre` before forming
an MFS sum.
"""
try:
    dataset_list[0].sparse_terms + dataset_shifted.sparse_terms
except Exception as e:
    print(f"  shifted + unshifted terms refuse to sum: {type(e).__name__}")
else:
    raise AssertionError("Shifted and unshifted SparseTerms should not sum.")

"""
__Wrap Up__

This script fitted the datacube of `modeling.py` without ever holding its visibilities in memory:

 - Each channel was streamed from disk in chunks into `al.Interferometer.from_stream`, which keeps only the
   mask-sized `SparseTerms` — memory is flat in the number of visibilities.
 - The FactorGraph fit, its results and its visualization are unchanged from the in-memory version.
 - Summing the per-channel `SparseTerms` gives the MFS dataset for free, via `al.Interferometer.from_sparse_terms`.
 - `phase_centre=` re-centres a channel as it streams; shifted and unshifted terms cannot be mixed.
"""
