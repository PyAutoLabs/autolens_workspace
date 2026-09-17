"""
Prepares the SDSS J1011+0143 dataset from the archival HST ACS/WFC frame on
MAST. See README.md for the provenance, units and frame convention; run from
the workspace root:

    python dataset/multi_galaxy/sdssj1011+0143/prep.py

The 215 MB drizzled frame is downloaded to a cache OUTSIDE the repository
(`--cache-dir`, default `~/.cache/pyautolens/mast`) and re-used on every
subsequent run, so this script is cheap to re-run and idempotent: the same
frame always produces byte-identical `.fits` outputs.

`--filter F555W` prepares the bluer frame of the same programme
(`j9qj51010_drc.fits`) instead; nothing else in the script changes.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, maximum_filter

import autolens as al
import autolens.plot as aplt

OUT = Path(__file__).resolve().parent

# HST programme 10831 (PI: Bolton), the ACS/WFC imaging Shu et al. 2016 (ApJ
# 820, 43; arXiv:1602.02927) modelled. One MAST observation per filter, each a
# 4 x 522 s drizzled stack at the ACS/WFC native 0.05"/pixel.
FILTERS = {
    "F814W": {"obs_id": "j9qj02010", "product": "j9qj02010_drc.fits"},
    "F555W": {"obs_id": "j9qj51010", "product": "j9qj51010_drc.fits"},
}

# The MAST target position of SDSSJ1011+0143 — the starting point for the
# cutout centre, which is then refined onto the two lens galaxies themselves.
TARGET_RA, TARGET_DEC = 152.87284, 1.72313

PIXEL_SCALE = 0.05  # arcsec / pixel, the ACS/WFC drizzled scale (D001SCAL)
CUTOUT_SHAPE = 201  # pixels, 10.05" on a side
PSF_SHAPE = 21  # pixels, the size recommended by imaging/data_preparation

PEAK_SEARCH_RADIUS = 2.0  # arcsec — the two lens galaxies lie within this of the target
SKY_ANNULUS_RADIUS = 4.0  # arcsec — sky is measured outside this radius of the centre
EXTRA_GALAXY_RADIUS = 3.0  # arcsec — the mask radius start_here.py fits inside

# A usable PSF star is bright enough to have structure in its wings but far
# below the ACS/WFC full well (~80,000 e- in a 522 s exposure), isolated, and
# as compact as the drizzled PSF (FWHM ~ 2.1 pixels).
STAR_PEAK_MIN, STAR_PEAK_MAX = 5.0, 60.0  # electrons / s
STAR_NEIGHBOUR_MAX = 0.02  # neighbour peak as a fraction of the star's peak
STAR_FWHM_MAX = 2.6  # pixels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "pyautolens" / "mast",
        help="where the 215 MB MAST frame is cached (never inside the repository)",
    )
    parser.add_argument(
        "--filter",
        choices=sorted(FILTERS),
        default="F814W",
        help="which ACS/WFC frame of programme 10831 to prepare",
    )
    parser.add_argument(
        "--diagnostic-png",
        action="store_true",
        help="write cutout / PSF diagnostic figures next to the cached frame",
    )
    return parser.parse_args()


def download(cache_dir, filter_name):
    """Fetch the drizzled frame from MAST, skipping the download if it is cached."""
    product = FILTERS[filter_name]["product"]
    for path in cache_dir.rglob(product):
        print(f"[download] cached: {path}")
        return path

    from astroquery.mast import Observations

    cache_dir.mkdir(parents=True, exist_ok=True)
    observations = Observations.query_criteria(obs_id=FILTERS[filter_name]["obs_id"])
    products = Observations.get_product_list(observations)
    products = products[products["productFilename"] == product]
    manifest = Observations.download_products(
        products, download_dir=str(cache_dir), cache=True
    )
    path = Path(manifest["Local Path"][0])
    print(f"[download] wrote: {path}")
    return path


def inspect_header(path):
    """Report the header facts the preparation depends on, and assert the units."""
    with fits.open(path) as hdu_list:
        primary, science, weight = (
            hdu_list[0].header,
            hdu_list["SCI"].header,
            hdu_list["WHT"].header,
        )
        wcs = WCS(science, hdu_list)

    pixel_scale = float(np.sqrt(np.abs(np.linalg.det(wcs.celestial.pixel_scale_matrix)))) * 3600.0

    print("[header] TELESCOP / INSTRUME / DETECTOR:", primary["TELESCOP"], primary["INSTRUME"], primary["DETECTOR"])
    print("[header] PROPOSID / TARGNAME / DATE-OBS:", primary["PROPOSID"], primary["TARGNAME"], primary["DATE-OBS"])
    print("[header] FILTER1 / FILTER2:", primary["FILTER1"], primary["FILTER2"])
    print("[header] EXPTIME:", primary["EXPTIME"], "s")
    print("[header] SCI BUNIT:", science["BUNIT"])
    print("[header] SCI PHOTFLAM / PHOTPLAM:", science["PHOTFLAM"], science["PHOTPLAM"])
    print("[header] D001SCAL:", primary["D001SCAL"], "-> WCS pixel scale:", f"{pixel_scale:.5f}\"/px")
    print("[header] D001OUUN (drizzle output units):", primary["D001OUUN"])
    print("[header] SCI ORIENTAT (PA of the +row axis, deg E of N):", science["ORIENTAT"])
    print("[header] WHT extension:", weight["EXTNAME"], "- AstroDrizzle exposure-time weight map (seconds)")

    assert science["BUNIT"].strip().upper() in ("ELECTRONS/S", "ELECTRONS"), science["BUNIT"]
    assert abs(pixel_scale - PIXEL_SCALE) < 1e-3, pixel_scale
    return primary, science, pixel_scale


def load_frame(path):
    """Load the SCI and WHT arrays, converting the SCI array to electrons / second.

    An AstroDrizzle DRC product with `D001OUUN = 'cps'` is already in electrons
    per second (`BUNIT = ELECTRONS/S`) and the WHT array is the exposure time
    contributing to every output pixel, in seconds. A `BUNIT = ELECTRONS` frame
    is divided by the total exposure time.
    """
    with fits.open(path) as hdu_list:
        science = np.asarray(hdu_list["SCI"].data, dtype=np.float64)
        weight = np.asarray(hdu_list["WHT"].data, dtype=np.float64)
        header = hdu_list["SCI"].header
        wcs = WCS(header, hdu_list)
        exposure_time = float(hdu_list[0].header["EXPTIME"])

    if header["BUNIT"].strip().upper() == "ELECTRONS":
        print("[units] BUNIT is ELECTRONS — dividing by EXPTIME to reach electrons / s.")
        science = science / exposure_time

    return science, weight, wcs, exposure_time


def centroid(image, row, column, half_width=4):
    """Flux-weighted centroid of a source in a small box around an integer peak."""
    stamp = np.clip(
        image[row - half_width : row + half_width + 1, column - half_width : column + half_width + 1],
        0.0,
        None,
    )
    offsets = np.arange(-half_width, half_width + 1)
    rows, columns = np.meshgrid(offsets, offsets, indexing="ij")
    total = stamp.sum()
    return row + (stamp * rows).sum() / total, column + (stamp * columns).sum() / total


def lens_galaxy_peaks(science, wcs):
    """Locate the two lens galaxies: the brightest peaks near the MAST target position."""
    column, row = wcs.all_world2pix(TARGET_RA, TARGET_DEC, 0)
    row, column = int(round(float(row))), int(round(float(column)))

    radius = int(round(PEAK_SEARCH_RADIUS / PIXEL_SCALE))
    stamp = science[row - radius : row + radius + 1, column - radius : column + radius + 1]

    smoothed = gaussian_filter(stamp, 1.5)
    peaks = np.nonzero(smoothed == maximum_filter(smoothed, 7))
    order = np.argsort(-smoothed[peaks])
    brightest = [(peaks[0][i], peaks[1][i]) for i in order[:2]]

    centres = [
        centroid(science, row - radius + peak_row, column - radius + peak_column)
        for peak_row, peak_column in brightest
    ]
    print(f"[centres] MAST target pixel (row, column) = ({row}, {column})")
    for i, (centre_row, centre_column) in enumerate(centres):
        print(f"[centres] lens {i} frame centroid (row, column) = ({centre_row:.2f}, {centre_column:.2f})")
    return centres


def measure_sky(values):
    """Sky level and RMS from the source-free outer annulus of the cutout."""
    rows, columns = np.mgrid[0 : values.shape[0], 0 : values.shape[1]]
    centre = (values.shape[0] - 1) / 2.0
    radius = np.hypot(rows - centre, columns - centre) * PIXEL_SCALE

    annulus = values[radius > SKY_ANNULUS_RADIUS]
    _, median, std = sigma_clipped_stats(annulus, sigma=3.0, maxiters=10)
    print(
        f"[sky] {annulus.size} pixels outside r = {SKY_ANNULUS_RADIUS}\": "
        f"level = {median:.6f} e-/s, RMS = {std:.6f} e-/s"
    )
    return float(median), float(std)


def report_blank_sky(science, row0, column0):
    """Report the sky far from the lens, for comparison with the cutout annulus.

    The drizzled frame is already sky-subtracted by AstroDrizzle, so a genuinely
    blank patch sits near zero. The cutout annulus sits above it by the light of
    the pair's own extended halo, which reaches beyond the cutout: subtracting
    the annulus value therefore removes that halo along with the sky. README.md
    records both numbers.
    """
    offset = 900
    for row, column in ((row0 + offset, column0 + offset), (row0 - offset, column0 - offset)):
        patch = science[row - 150 : row + 150, column - 150 : column + 150]
        _, median, std = sigma_clipped_stats(patch, sigma=3.0, maxiters=5)
        print(
            f"[sky] blank field at (row, column) = ({row}, {column}): "
            f"level = {median:.6f} e-/s, RMS = {std:.6f} e-/s"
        )


def find_psf_star(science, weight, centre_row, centre_column):
    """Find the isolated, unsaturated, compact star nearest the lens."""
    smoothed = gaussian_filter(science, 0.8)
    candidates = np.nonzero(
        (smoothed == maximum_filter(smoothed, 9))
        & (science > STAR_PEAK_MIN)
        & (science < STAR_PEAK_MAX)
        & (weight > 0.5 * np.nanmedian(weight))
    )

    half = PSF_SHAPE // 2
    rows, columns = np.mgrid[0 : 2 * half + 11, 0 : 2 * half + 11]
    isolation_radius = np.hypot(rows - (half + 5), columns - (half + 5))

    stars = []
    for row, column in zip(*candidates):
        if not (half + 5 <= row < science.shape[0] - half - 5):
            continue
        if not (half + 5 <= column < science.shape[1] - half - 5):
            continue
        # Reject anything with a comparably bright neighbour just outside the core.
        wide = science[row - half - 5 : row + half + 6, column - half - 5 : column + half + 6]
        neighbour = np.where(isolation_radius < 7, -np.inf, wide).max()
        if neighbour > STAR_NEIGHBOUR_MAX * science[row, column]:
            continue
        stamp = science[row - half : row + half + 1, column - half : column + half + 1]
        fwhm = fwhm_of(stamp)
        if not np.isfinite(fwhm) or fwhm > STAR_FWHM_MAX:
            continue
        separation = float(np.hypot(row - centre_row, column - centre_column)) * PIXEL_SCALE
        stars.append((separation, int(row), int(column), float(fwhm)))

    assert stars, (
        "no star passed the isolation / peak / compactness cuts; relax "
        "STAR_PEAK_MIN, STAR_NEIGHBOUR_MAX or STAR_FWHM_MAX and re-run."
    )
    separation, row, column, fwhm = min(stars)
    print(
        f"[psf] {len(stars)} usable stars; nearest is at (row, column) = ({row}, {column}), "
        f"{separation:.1f}\" from the lens, peak = {science[row, column]:.2f} e-/s, "
        f"FWHM = {fwhm:.2f} px ({fwhm * PIXEL_SCALE:.3f}\")"
    )
    return row, column, fwhm


def fwhm_of(stamp):
    """FWHM in pixels from the azimuthally-averaged radial profile of a star."""
    half = stamp.shape[0] // 2
    values = np.clip(stamp, 0.0, None)
    core = values[half - 2 : half + 3, half - 2 : half + 3]
    offsets = np.arange(-2, 3)
    rows, columns = np.meshgrid(offsets, offsets, indexing="ij")
    centre_row = half + (core * rows).sum() / core.sum()
    centre_column = half + (core * columns).sum() / core.sum()

    grid_rows, grid_columns = np.mgrid[0 : stamp.shape[0], 0 : stamp.shape[1]]
    radius = np.hypot(grid_rows - centre_row, grid_columns - centre_column)

    edges = np.arange(0.0, half + 0.5, 0.5)
    profile = np.array(
        [
            values[(radius >= low) & (radius < high)].mean()
            if ((radius >= low) & (radius < high)).any()
            else np.nan
            for low, high in zip(edges[:-1], edges[1:])
        ]
    )
    centres = 0.5 * (edges[:-1] + edges[1:])

    peak = values.max()
    below = np.nonzero(profile < 0.5 * peak)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    fraction = (profile[i - 1] - 0.5 * peak) / (profile[i - 1] - profile[i])
    return 2.0 * (centres[i - 1] + fraction * (centres[i] - centres[i - 1]))


def to_pyautolens(cutout):
    """Orient a FITS cutout the way PyAutoLens displays a native array.

    A FITS image array's first row is the bottom of the displayed image, while a
    PyAutoLens native array's first row is the TOP (`y` decreases with the row
    index). Reversing the rows therefore reproduces the sky orientation of the
    original frame — the same handedness a FITS viewer shows — rather than
    mirroring it.
    """
    return np.ascontiguousarray(cutout[::-1, :])


def pyautolens_coordinates(row, column, shape):
    """(y, x) in arcsec of a native array index, relative to the array centre."""
    centre = (shape - 1) / 2.0
    return (centre - row) * PIXEL_SCALE, (column - centre) * PIXEL_SCALE


def main():
    arguments = parse_args()
    filter_name = arguments.filter

    path = download(arguments.cache_dir, filter_name)
    primary, science_header, pixel_scale = inspect_header(path)
    science, weight, wcs, exposure_time = load_frame(path)

    # --- Cutout centre: the midpoint of the two lens galaxies ---

    galaxy_centres = lens_galaxy_peaks(science, wcs)
    separation = float(
        np.hypot(
            galaxy_centres[0][0] - galaxy_centres[1][0],
            galaxy_centres[0][1] - galaxy_centres[1][1],
        )
    ) * PIXEL_SCALE
    midpoint_row = 0.5 * (galaxy_centres[0][0] + galaxy_centres[1][0])
    midpoint_column = 0.5 * (galaxy_centres[0][1] + galaxy_centres[1][1])

    row0, column0 = int(round(midpoint_row)), int(round(midpoint_column))
    centre_ra, centre_dec = [float(v) for v in wcs.all_pix2world(column0, row0, 0)]
    print(
        f"[cutout] centre pixel (row, column) = ({row0}, {column0}), "
        f"(RA, Dec) = ({centre_ra:.6f}, {centre_dec:.6f}), "
        f"galaxy separation = {separation:.3f}\""
    )

    half = CUTOUT_SHAPE // 2
    science_cutout = to_pyautolens(
        science[row0 - half : row0 + half + 1, column0 - half : column0 + half + 1]
    )
    weight_cutout = to_pyautolens(
        weight[row0 - half : row0 + half + 1, column0 - half : column0 + half + 1]
    )
    assert science_cutout.shape == (CUTOUT_SHAPE, CUTOUT_SHAPE), science_cutout.shape
    assert np.isfinite(science_cutout).all(), "cutout contains NaN / inf"

    # The position angle of the array's +y axis, measured East of North: the
    # frame is drizzled in the detector frame, not rotated to North-up.
    step_ra, step_dec = [float(v) for v in wcs.all_pix2world(column0, row0 + 1, 0)]
    position_angle = float(
        np.degrees(
            np.arctan2(
                (step_ra - centre_ra) * np.cos(np.radians(centre_dec)),
                step_dec - centre_dec,
            )
        )
    )
    print(f"[cutout] position angle of the +y array axis: {position_angle:.2f} deg E of N")

    # --- data.fits: sky-subtracted science cutout, electrons / second ---

    sky_level, sky_rms = measure_sky(science_cutout)
    report_blank_sky(science, row0, column0)
    data = science_cutout - sky_level

    aplt.fits_array(
        array=al.Array2D.no_mask(values=data, pixel_scales=PIXEL_SCALE),
        file_path=OUT / "data.fits",
        overwrite=True,
    )

    # --- noise_map.fits: RMS electrons / second, Poisson + background ---

    exposure_time_map = weight_cutout
    assert exposure_time_map.min() > 0.0, "the WHT cutout contains zero-exposure pixels"
    print(
        f"[noise] exposure-time map (WHT) min / median / max = "
        f"{exposure_time_map.min():.1f} / {np.median(exposure_time_map):.1f} / "
        f"{exposure_time_map.max():.1f} s (EXPTIME = {exposure_time:.0f} s)"
    )

    # The Poisson term uses the counts before sky subtraction, clipped at zero so
    # that noise-dominated negative pixels do not contribute an imaginary
    # variance. The sky's own Poisson noise, the read noise and the drizzle
    # correlation are all carried empirically by the measured background RMS.
    noise_map = al.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from(
        data_eps=al.Array2D.no_mask(
            values=np.clip(data + sky_level, 0.0, None), pixel_scales=PIXEL_SCALE
        ),
        exposure_time_map=al.Array2D.no_mask(
            values=exposure_time_map, pixel_scales=PIXEL_SCALE
        ),
        background_noise_map=al.Array2D.no_mask(
            values=np.full_like(data, sky_rms), pixel_scales=PIXEL_SCALE
        ),
    )
    noise_values = np.asarray(noise_map.native, dtype=np.float64)
    assert np.isfinite(noise_values).all(), "noise-map contains NaN / inf"
    assert (noise_values > 0.0).all(), "noise-map contains non-positive values"

    aplt.fits_array(
        array=al.Array2D.no_mask(values=noise_values, pixel_scales=PIXEL_SCALE),
        file_path=OUT / "noise_map.fits",
        overwrite=True,
    )

    rows, columns = np.mgrid[0:CUTOUT_SHAPE, 0:CUTOUT_SHAPE]
    radius = np.hypot(rows - half, columns - half) * PIXEL_SCALE
    sky_signal_to_noise = (data / noise_values)[radius > SKY_ANNULUS_RADIUS]
    print(
        f"[noise] sky pixels' data / noise: mean = {sky_signal_to_noise.mean():.3f}, "
        f"std = {sky_signal_to_noise.std():.3f} (should be ~0 and ~1)"
    )

    # --- psf.fits: an isolated star from the same frame ---

    star_row, star_column, star_fwhm = find_psf_star(science, weight, row0, column0)
    star_centre_row, star_centre_column = centroid(science, star_row, star_column, half_width=3)
    star_row, star_column = int(round(star_centre_row)), int(round(star_centre_column))

    psf_half = PSF_SHAPE // 2
    psf = to_pyautolens(
        science[
            star_row - psf_half : star_row + psf_half + 1,
            star_column - psf_half : star_column + psf_half + 1,
        ]
    )
    corner = np.concatenate([psf[0, :], psf[-1, :], psf[:, 0], psf[:, -1]])
    _, star_sky, _ = sigma_clipped_stats(corner, sigma=3.0, maxiters=5)
    psf = np.clip(psf - star_sky, 0.0, None)
    psf = psf / psf.sum()

    assert PSF_SHAPE % 2 == 1, "the PSF must have odd dimensions"
    assert abs(psf.sum() - 1.0) < 1e-12, psf.sum()
    assert np.unravel_index(np.argmax(psf), psf.shape) == (psf_half, psf_half), (
        "the PSF peak is not the central pixel"
    )

    star_ra, star_dec = [float(v) for v in wcs.all_pix2world(star_column, star_row, 0)]
    print(
        f"[psf] star (RA, Dec) = ({star_ra:.6f}, {star_dec:.6f}), local sky = "
        f"{star_sky:.5f} e-/s, kernel sum = {psf.sum():.6f}"
    )

    aplt.fits_array(
        array=al.Array2D.no_mask(values=psf, pixel_scales=PIXEL_SCALE),
        file_path=OUT / "psf.fits",
        overwrite=True,
    )

    # --- main_lens_centres.json: the two light centroids, in PyAutoLens (y, x) ---
    #
    # Measured on the array as PyAutoLens loads it back, so the orientation of
    # the written file is what defines the centres — no convention is assumed.

    loaded = np.asarray(
        al.Array2D.from_fits(file_path=OUT / "data.fits", pixel_scales=PIXEL_SCALE).native
    )
    smoothed = gaussian_filter(loaded, 1.5)
    peaks = np.nonzero(smoothed == maximum_filter(smoothed, 7))
    order = np.argsort(-smoothed[peaks])
    centres = []
    for i in order[:2]:
        centre_row, centre_column = centroid(loaded, int(peaks[0][i]), int(peaks[1][i]))
        centres.append(pyautolens_coordinates(centre_row, centre_column, CUTOUT_SHAPE))
    centres.sort(reverse=True)  # deterministic order: the northern galaxy first

    centre_separation = float(
        np.hypot(centres[0][0] - centres[1][0], centres[0][1] - centres[1][1])
    )
    print(
        f"[centres] PyAutoLens (y, x) = "
        f"({centres[0][0]:.4f}, {centres[0][1]:.4f}) and "
        f"({centres[1][0]:.4f}, {centres[1][1]:.4f}); separation = {centre_separation:.4f}\""
    )

    al.output_to_json(
        obj=al.Grid2DIrregular([tuple(centres[0]), tuple(centres[1])]),
        file_path=OUT / "main_lens_centres.json",
    )

    # --- mask_extra_galaxies.fits ---
    #
    # The field inside the 3.0" mask start_here.py fits holds the two lens
    # galaxies and the lensed images of the source, and nothing else: there is
    # no contaminant to scale the noise-map over, so no mask is written. See
    # README.md; if a re-run on another field needs one, add its circles here.
    extra_galaxy_circles = []  # (y, x, radius) in arcsec
    if extra_galaxy_circles:
        grid = al.Grid2D.uniform(
            shape_native=(CUTOUT_SHAPE, CUTOUT_SHAPE), pixel_scales=PIXEL_SCALE
        )
        distances = np.asarray(grid.native)
        masked = np.zeros((CUTOUT_SHAPE, CUTOUT_SHAPE), dtype=bool)
        for centre_y, centre_x, circle_radius in extra_galaxy_circles:
            masked |= (
                np.hypot(distances[:, :, 0] - centre_y, distances[:, :, 1] - centre_x)
                < circle_radius
            )
        aplt.fits_array(
            array=al.Mask2D(mask=masked, pixel_scales=PIXEL_SCALE),
            file_path=OUT / "mask_extra_galaxies.fits",
            overwrite=True,
        )
        print(f"[mask] wrote mask_extra_galaxies.fits ({masked.sum()} pixels)")
    else:
        print("[mask] field is clean inside 3.0\" — no mask_extra_galaxies.fits written")

    # --- info.json: the provenance of everything above ---

    info = {
        "reference": "Shu et al. 2016, ApJ 820, 43 (arXiv:1602.02927)",
        "program": int(primary["PROPOSID"]),
        "obs_id": FILTERS[filter_name]["obs_id"],
        "product": FILTERS[filter_name]["product"],
        "instrument": f"{primary['INSTRUME']}/{primary['DETECTOR']}",
        "filter": filter_name,
        "date_obs": primary["DATE-OBS"],
        "exptime_s": float(primary["EXPTIME"]),
        "bunit": science_header["BUNIT"].strip(),
        "photflam": float(science_header["PHOTFLAM"]),
        "pixel_scale": pixel_scale,
        "shape_native": [CUTOUT_SHAPE, CUTOUT_SHAPE],
        "cutout_centre_ra_dec": [centre_ra, centre_dec],
        "position_angle_y_axis_deg_e_of_n": position_angle,
        "sky_level_eps": sky_level,
        "sky_rms_eps": sky_rms,
        "main_lens_centres_arcsec": [list(centres[0]), list(centres[1])],
        "main_lens_separation_arcsec": centre_separation,
        "psf_star_ra_dec": [star_ra, star_dec],
        "psf_star_pixel": [star_row, star_column],
        "psf_fwhm_pixels": star_fwhm,
        "prep_date": date.today().isoformat(),
    }
    with open(OUT / "info.json", "w") as file:
        json.dump(info, file, indent=4)
        file.write("\n")

    if arguments.diagnostic_png:
        write_diagnostics(arguments.cache_dir, data, noise_values, psf, centres)

    print(f"[done] wrote {sorted(p.name for p in OUT.glob('*'))}")


def write_diagnostics(cache_dir, data, noise_map, psf, centres):
    """Optional figures, written beside the cached frame (never into the repo)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (values, title) in zip(
        axes,
        [
            (np.arcsinh(data / 0.02), "data (asinh)"),
            (data / noise_map, "signal-to-noise"),
            (np.arcsinh(psf / psf.max() / 0.01), "psf (asinh)"),
        ],
    ):
        image = axis.imshow(values, cmap="magma")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    path = cache_dir / "sdssj1011+0143_prep.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    print(f"[diagnostic] wrote {path}")


if __name__ == "__main__":
    main()
