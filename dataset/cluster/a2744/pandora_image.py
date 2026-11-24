"""Read the Pandora HST cutout in the east/north frame of point_datasets.csv."""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import autolens as al


def image_from_fits(path):
    """Place each HST pixel at its catalogue offset from the A2744 core."""
    pixels, header = fits.getdata(path, header=True)
    wcs = WCS(header).celestial
    matrix = wcs.pixel_scale_matrix * 3600.0
    if (
        pixels.ndim != 2
        or list(wcs.wcs.ctype) != ["RA---TAN", "DEC--TAN"]
        or wcs.has_distortion
        or not np.allclose(wcs.wcs.crval, (3.5875, -30.3972), rtol=0, atol=1e-8)
        or not np.allclose(matrix, np.diag(np.diag(matrix)), rtol=0, atol=1e-10)
    ):
        raise ValueError(f"Expected an axis-aligned TAN cutout centred on the Pandora catalogue: {path}")

    dx, dy = np.diag(matrix)
    # Array2D columns increase east and its first row is the northern edge.
    # FITS CRPIX is one-based and need not coincide with the array midpoint.
    if dx < 0:
        pixels = pixels[:, ::-1]
    if dy > 0:
        pixels = pixels[::-1, :]
    centre_y, centre_x = (np.asarray(pixels.shape) - 1) / 2
    ref_x, ref_y = wcs.wcs.crpix - 1
    return al.Array2D.no_mask(
        values=pixels.copy(),
        pixel_scales=(abs(dy), abs(dx)),
        origin=((centre_y - ref_y) * dy, (centre_x - ref_x) * dx),
    )
