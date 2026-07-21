"""
Simulator: Datacube
====================

This script simulates an ALMA-style spectral-line `Interferometer` datacube of a 'galaxy-scale' strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`, identical across all channels.
 - The source galaxy's light is an `Sersic`, whose `intensity` follows a Gaussian emission-line profile in the
   channel index (peaking near the cube centre and falling off at the edges).

A "datacube" here is a Python list of `Interferometer` objects, one per spectral channel. Each channel writes its
own ``data.fits`` / ``noise_map.fits`` / ``uv_wavelengths.fits`` into a separate folder, alongside the true tracer
for that channel. The modeling examples in this folder load the cube by listing those folders.

For Phase 1 of datacube modeling we keep things deliberately simple: every channel uses the same `uv_wavelengths`
and the same noise level. Two source properties vary across channels — `intensity` (a Gaussian emission-line
profile) and `centre` (a small linear shift along the y axis, simulating the kinematic gradient that real
emission lines almost always exhibit).

__Contents__

- **Cube Configuration:** Number of channels and the emission-line shape used to drive the per-channel intensity.
- **Dataset Paths:** Where the per-channel datasets, summary JSON and overview plots are written.
- **uv_wavelengths:** Reuse the SMA `uv_wavelengths.fits` shipped with the workspace as a stand-in for ALMA coverage.
- **Real-Space Grid:** The 2D image-plane grid each channel is evaluated on before the Fourier transform.
- **Lens Galaxy:** Shared `Isothermal + ExternalShear` lens, identical for every channel.
- **Per-Channel Source:** A Gaussian emission line drives the per-channel `Sersic.intensity`; the `centre` shifts linearly along y across channels to mimic a kinematic gradient.
- **Per-Channel Simulate:** Loop over channels: build the tracer, simulate, write FITS + tracer.json to disk.
- **3D-FITS Cube:** Stack the per-channel arrays into single `(n_chan, n_vis, 2)` FITS files — the autolens-canonical post-polarisation-collapse shape.
- **4D CASA-like Cube:** Wrap the 3D cubes in a polarisation axis to produce `(n_pol, n_chan, n_vis, 2)` files matching CASA's native output. Polarisations are identical here for simplicity.
- **Multiple Images:** Compute and save the lensed multiple-image positions used by the modeling scripts' `PositionsLH` penalty.
- **Cube Summary:** Dump the emission-line parameters and per-channel intensities to `cube_summary.json`.
- **Cube Overview Plot:** Row-per-channel sanity figure (lensed image, uv-plane Re/Im, |vis| vs baseline length).
- **Spectrum Plot:** Source intensity as a function of channel index.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import autolens as al

"""
__Cube Configuration__

The cube shape is set by ``N_CHANNELS`` and the emission-line shape by the Gaussian profile parameters below. Keep
``N_CHANNELS`` small for the prototype: per-channel inversion cost grows linearly, and 4 channels are enough to
exercise the FactorGraph wiring end-to-end.
"""
N_CHANNELS = 4
PEAK_CHANNEL = 1.5
SIGMA_CHANNEL = 1.2
PEAK_INTENSITY = 0.6

# Total source-centre shift along the y axis across the whole cube, in arcsec.
# At the simulator's 0.1"/pixel grid, 0.12" is about 1.2 pixels end-to-end —
# visible in the per-channel lensed image without the source jumping outside the Einstein ring.
CENTRE_SHIFT_TOTAL = 0.12
SOURCE_CENTRE_BASE = (0.1, 0.1)

"""
__Dataset Paths__

Each channel lives in its own subfolder so the modeling scripts can iterate over them with a simple loop. The
``cube_summary.json`` records the emission-line parameters used to simulate the cube — modeling can compare the
recovered per-channel source amplitudes against those true values.
"""
dataset_type = "interferometer"
dataset_label = "datacube"
dataset_name = "sim_simple"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name
dataset_path.mkdir(parents=True, exist_ok=True)

"""
__uv_wavelengths__

Load the SMA `uv_wavelengths.fits` shipped with the workspace at ``dataset/interferometer/uv_wavelengths/sma.fits``.
SMA's coverage is small (~190 visibilities) but it makes the prototype fast to iterate on without paying the ALMA
runtime cost. To swap in a real ALMA cube, point this path at your own per-channel `uv_wavelengths.fits` files
inside the simulation loop below.
"""
uv_wavelengths_path = Path("dataset", dataset_type, "uv_wavelengths")
uv_wavelengths = al.ndarray_via_fits_from(
    file_path=uv_wavelengths_path / "sma.fits", hdu=0
)

"""
__Real-Space Grid__

For interferometer data this image is evaluated in real space and then Fourier-transformed. Interferometer
calculations don't need over-sampling — the Fourier transform is the dominant numerical cost.
"""
grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

"""
__Lens Galaxy__

The lens mass + external shear is shared across channels — the lens doesn't change with frequency.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__Per-Channel Source__

Two things vary channel-to-channel:

  - ``intensity`` follows a Gaussian profile in channel index — the simplest model of an emission line peaking at
    one velocity and falling off symmetrically.
  - ``centre`` shifts linearly along the y axis from ``-CENTRE_SHIFT_TOTAL / 2`` to ``+CENTRE_SHIFT_TOTAL / 2``
    relative to ``SOURCE_CENTRE_BASE``. This mimics a kinematic gradient (e.g. a rotating disk's centroid drifting
    across the emission line) and gives each channel a visually distinct lensed image. A real cube might also shift
    in x or rotate; one-axis is enough to demonstrate the principle.

The other shape parameters (ellipticity, effective radius, Sersic index) are held fixed.
"""


def channel_intensity(channel: int) -> float:
    return PEAK_INTENSITY * float(
        np.exp(-0.5 * ((channel - PEAK_CHANNEL) / SIGMA_CHANNEL) ** 2)
    )


def channel_centre(channel: int) -> tuple:
    """Source centre for a given channel.

    Linearly shifts along the y axis from channel 0 to channel N-1, centred on
    ``SOURCE_CENTRE_BASE``. For ``N_CHANNELS=4`` and ``CENTRE_SHIFT_TOTAL=0.12"`` the per-channel
    centres are ``(0.04, 0.1)``, ``(0.08, 0.1)``, ``(0.12, 0.1)``, ``(0.16, 0.1)``.
    """
    if N_CHANNELS == 1:
        return SOURCE_CENTRE_BASE
    fractional = (channel - (N_CHANNELS - 1) / 2.0) / (N_CHANNELS - 1)
    offset_y = CENTRE_SHIFT_TOTAL * fractional
    return (SOURCE_CENTRE_BASE[0] + offset_y, SOURCE_CENTRE_BASE[1])


def source_galaxy_for(channel: int) -> al.Galaxy:
    return al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=channel_centre(channel),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=channel_intensity(channel),
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )


"""
__Per-Channel Simulate__

For each channel we build a tracer with the shared lens + the per-channel source, run
``SimulatorInterferometer.via_tracer_from``, and write the resulting visibilities, noise map, baselines and true
tracer into ``channel_NNN/``. The `noise_seed` is incremented per channel so each channel sees an independent noise
realisation.
"""
channel_intensities = []
channel_centres = []
datasets = []
tracers = []

for channel in range(N_CHANNELS):
    intensity = channel_intensity(channel)
    centre = channel_centre(channel)
    channel_intensities.append(intensity)
    channel_centres.append(list(centre))

    channel_path = dataset_path / f"channel_{channel:03d}"
    channel_path.mkdir(parents=True, exist_ok=True)

    source_galaxy = source_galaxy_for(channel)
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=uv_wavelengths,
        exposure_time=300.0,
        noise_sigma=1000.0,
        transformer_class=al.TransformerDFT,
        noise_seed=1 + channel,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    datasets.append(dataset)
    tracers.append(tracer)

    al.output_to_fits(
        values=np.stack([dataset.data.real, dataset.data.imag], axis=-1),
        file_path=channel_path / "data.fits",
        overwrite=True,
    )
    al.output_to_fits(
        values=np.stack([dataset.noise_map.real, dataset.noise_map.imag], axis=-1),
        file_path=channel_path / "noise_map.fits",
        overwrite=True,
    )
    al.output_to_fits(
        values=dataset.uv_wavelengths,
        file_path=channel_path / "uv_wavelengths.fits",
        overwrite=True,
    )
    al.output_to_json(
        obj=tracer,
        file_path=channel_path / "tracer.json",
    )

    print(
        f"  channel {channel:03d}: intensity={intensity:.4f}, "
        f"centre={centre}, "
        f"|vis|_max={np.max(np.abs(dataset.data)):.3e}"
    )

"""
__3D-FITS Cube__

ALMA visibilities exit CASA as a single 4D FITS of shape `(n_pol, n_chan, n_vis, 2)`. After the user collapses
the polarisation axis (averaging or concatenating — see `data_preparation.py`), the canonical input shape is
`(n_chan, n_vis, 2)`. We write three additional files at the cube root in that shape so users with CASA-native
data can load the cube via `data_preparation.dataset_list_from_3d_fits` without first splitting into per-channel
folders. The per-channel `channel_NNN/` folders above are kept as well — both layouts coexist for now.
"""
visibilities_cube = np.stack(
    [np.stack([d.data.real, d.data.imag], axis=-1) for d in datasets], axis=0
)
noise_map_cube = np.stack(
    [np.stack([d.noise_map.real, d.noise_map.imag], axis=-1) for d in datasets], axis=0
)
uv_wavelengths_cube = np.stack([np.asarray(d.uv_wavelengths) for d in datasets], axis=0)

al.output_to_fits(
    values=visibilities_cube,
    file_path=dataset_path / "visibilities_cube.fits",
    overwrite=True,
)
al.output_to_fits(
    values=noise_map_cube,
    file_path=dataset_path / "noise_map_cube.fits",
    overwrite=True,
)
al.output_to_fits(
    values=uv_wavelengths_cube,
    file_path=dataset_path / "uv_wavelengths_cube.fits",
    overwrite=True,
)

print(f"  3D-FITS cubes: shape {visibilities_cube.shape} (n_chan, n_vis, 2)")

"""
__4D CASA-like Cube__

ALMA visibilities arrive from CASA as a single 4D FITS of shape `(n_pol, n_chan, n_vis, 2)` — for example
`(2, 34, 16984, 2)` for a typical narrow-line observation. Users typically run a polarisation-collapse step
(averaging or concatenating the two pols — see `data_preparation.py`) before the data is fed to autolens.

We write a 4D version of each cube here so users can practice the polarisation-collapse step against the
simulator's actual output rather than synthetic random arrays. For this synthetic simulator both polarisations
carry **identical** data (same visibilities, same noise_map, same baselines) — that's a pedagogical
simplification: real CASA data has independent noise realisations between polarisations, which is why
averaging real data reduces the effective noise by `sqrt(2)`. The baselines being identical across pols is
astronomically correct (both pols share the same antenna pairs).
"""
visibilities_4d_cube = np.stack([visibilities_cube, visibilities_cube], axis=0)
noise_map_4d_cube = np.stack([noise_map_cube, noise_map_cube], axis=0)
uv_wavelengths_4d_cube = np.stack([uv_wavelengths_cube, uv_wavelengths_cube], axis=0)

al.output_to_fits(
    values=visibilities_4d_cube,
    file_path=dataset_path / "visibilities_4d_cube.fits",
    overwrite=True,
)
al.output_to_fits(
    values=noise_map_4d_cube,
    file_path=dataset_path / "noise_map_4d_cube.fits",
    overwrite=True,
)
al.output_to_fits(
    values=uv_wavelengths_4d_cube,
    file_path=dataset_path / "uv_wavelengths_4d_cube.fits",
    overwrite=True,
)

print(
    f"  4D CASA-like cubes: shape {visibilities_4d_cube.shape} (n_pol, n_chan, n_vis, 2)"
)

"""
__Multiple Images__

Pixelized source modeling can drift toward unphysical demagnified-source local maxima — the source pixels are
reconstructed in low-magnification regions of the source plane that fit the noise rather than the lensed signal.
The `PositionsLH` penalty defends against that by reading a small set of multiple-image positions from disk and
adding a likelihood penalty for any candidate lens model whose source-plane back-projection of those positions
spreads them apart.

For simulated data we can compute the multiple-image positions automatically with `al.PointSolver`. The lens
model is channel-invariant; the source centre shifts slightly across channels, so we compute positions for the
*mean* source centre (`SOURCE_CENTRE_BASE`) — the resulting positions are within the `PositionsLH` threshold
for every individual channel. A single `positions.json` covers the entire cube.
"""
solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)
positions = solver.solve(
    tracer=tracers[0],
    source_plane_coordinate=SOURCE_CENTRE_BASE,
)

al.output_to_json(
    obj=positions,
    file_path=dataset_path / "positions.json",
)

print(f"  multiple-image positions: {len(positions)}")

"""
__Cube Summary__

A small JSON sidecar lets downstream scripts (and the user) recover the emission-line parameters used during
simulation without re-running the script.
"""
summary = {
    "n_channels": N_CHANNELS,
    "peak_channel": PEAK_CHANNEL,
    "sigma_channel": SIGMA_CHANNEL,
    "peak_intensity": PEAK_INTENSITY,
    "centre_shift_total": CENTRE_SHIFT_TOTAL,
    "source_centre_base": list(SOURCE_CENTRE_BASE),
    "channel_intensities": channel_intensities,
    "channel_centres": channel_centres,
}
with open(dataset_path / "cube_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

"""
__Cube Overview Plot__

Row-per-channel sanity figure: lensed real-space image, Re(visibilities) and Im(visibilities) in the uv-plane,
and |vis| as a function of baseline length. Useful for eyeballing whether the emission-line modulation has
propagated through the simulator end-to-end.
"""
fig, axes = plt.subplots(N_CHANNELS, 4, figsize=(16, 3.4 * N_CHANNELS), squeeze=False)

for c in range(N_CHANNELS):
    dataset = datasets[c]
    tracer = tracers[c]
    uv = np.asarray(dataset.uv_wavelengths)
    re = np.asarray(dataset.data.real)
    im = np.asarray(dataset.data.imag)
    amp = np.hypot(re, im)
    baseline = np.hypot(uv[:, 0], uv[:, 1])

    image = tracer.image_2d_from(grid=grid)
    axes[c, 0].imshow(np.asarray(image.native), origin="lower", cmap="hot")
    axes[c, 0].set_title(f"channel {c}: lensed image (I={channel_intensities[c]:.3f})")
    axes[c, 0].set_axis_off()

    sc = axes[c, 1].scatter(uv[:, 0], uv[:, 1], c=re, s=8, cmap="RdBu_r")
    axes[c, 1].set_title("Re(visibilities)")
    axes[c, 1].set_aspect("equal")
    axes[c, 1].set_xlabel("u")
    axes[c, 1].set_ylabel("v")
    plt.colorbar(sc, ax=axes[c, 1], fraction=0.046)

    sc = axes[c, 2].scatter(uv[:, 0], uv[:, 1], c=im, s=8, cmap="RdBu_r")
    axes[c, 2].set_title("Im(visibilities)")
    axes[c, 2].set_aspect("equal")
    axes[c, 2].set_xlabel("u")
    axes[c, 2].set_ylabel("v")
    plt.colorbar(sc, ax=axes[c, 2], fraction=0.046)

    axes[c, 3].scatter(baseline, amp, s=8, alpha=0.6)
    axes[c, 3].set_title("|vis| vs baseline length")
    axes[c, 3].set_xlabel(r"$\sqrt{u^2 + v^2}$")
    axes[c, 3].set_ylabel("|visibilities|")

fig.tight_layout()
fig.savefig(dataset_path / "cube_overview.png", dpi=120, bbox_inches="tight")
plt.close(fig)

"""
__Spectrum Plot__

Source intensity vs channel index — the input emission-line profile this cube was simulated from. Modeling
scripts can compare the recovered per-channel inversion magnitude against this curve to sanity-check the fit.
"""
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(N_CHANNELS), channel_intensities, "o-")
ax.set_xlabel("channel index")
ax.set_ylabel("source intensity")
ax.set_title(f"emission-line spectrum (peak={PEAK_CHANNEL}, sigma={SIGMA_CHANNEL})")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(dataset_path / "spectrum.png", dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"  cube simulated: {dataset_path}")
print(f"    channels:           {N_CHANNELS}")
print(f"    visibilities/chan:  {uv_wavelengths.shape[0]}")
print(f"    real-space grid:    256 x 256")
print(f"    peak intensity:     {PEAK_INTENSITY}")
