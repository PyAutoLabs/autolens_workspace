# SDSS J1011+0143 — HST ACS/WFC imaging of a co-dominant lens pair

Real Hubble imaging of **SDSS J1011+0143**, the merging pair of early-type
galaxies at z = 0.331 that lenses a z = 2.701 Lyman-alpha emitter, published by
**Shu et al. 2016 (ApJ 820, 43, arXiv:1602.02927)**. It is the dataset
`scripts/multi_galaxy/start_here.py` fits.

## Source and selection

| | |
|---|---|
| Programme | HST GO 10831 (PI Bolton) |
| Observation | `j9qj02010`, product `j9qj02010_drc.fits` (215 MB) |
| Instrument | ACS/WFC |
| Filter | F814W (`FILTER1 = CLEAR1L`, `FILTER2 = F814W`) |
| Date | 2006-11-10 |
| Exposure | `EXPTIME = 2088 s` (4 x 522 s, AstroDrizzle-combined, CTE corrected) |
| Units | `BUNIT = ELECTRONS/S`, `D001OUUN = cps` |
| Pixel scale | 0.05"/pixel (`D001SCAL`, confirmed from the WCS) |
| Zeropoint inputs | `PHOTFLAM = 7.131553175e-20`, `PHOTPLAM = 8045.2 A` |

The same programme observed the field in F555W (`j9qj51010`). Shu et al. modelled
both; F814W is used here because the lens galaxies and the lensed images are
both better detected in it.

**The full 215 MB frame is not committed.** `prep.py` downloads it from MAST into
a cache *outside* the repository and cuts the small stamp that is committed.

## What is committed

| File | Size | Contents |
|---|---|---|
| `data.fits` | 321 KB | 201 x 201 pixel (10.05") sky-subtracted science cutout, electrons / second |
| `noise_map.fits` | 321 KB | Per-pixel RMS in electrons / second (Poisson + background) |
| `psf.fits` | 8.4 KB | 21 x 21 pixel PSF from a star in the same frame, normalised to sum 1 |
| `main_lens_centres.json` | 495 B | The two lens galaxies' light centroids, as a `Grid2DIrregular` |
| `info.json` | 1.1 KB | Machine-readable provenance for everything below |
| `prep.py` | 24 KB | The script that produced all of the above |

Total: **696 KB**. There is no `mask_extra_galaxies.fits` — see *Extra galaxies*.

## Units and frame

- The data and noise-map are in **electrons per second**, the units PyAutoLens
  assumes. The drizzled product is already in these units; `prep.py` asserts it
  and divides by `EXPTIME` if it ever finds a frame in electrons.
- The cutout is centred at **(RA, Dec) = (152.872965, 1.723122)** — the midpoint
  of the two galaxies' light centroids, refined from the MAST target position
  (152.87284, 1.72313).
- The frame is drizzled in the **detector frame, not rotated North-up**: the
  `+y` axis of the PyAutoLens array lies at a position angle of **-71.17 degrees
  East of North** (the frame's `ORIENTAT`). `prep.py` reverses the row order of
  the FITS cutout when it writes `data.fits`, because a FITS array's first row is
  the *bottom* of the displayed image while a PyAutoLens native array's first row
  is the *top*. That reversal preserves the sky handedness — the committed array
  is what a FITS viewer shows, not its mirror image.
- Lens modelling is indifferent to the orientation, so the frame is left
  unrotated rather than resampled; resampling would correlate the noise further.

## Measured quantities

- **Sky.** Sigma-clipped median of the cutout's outer annulus (r > 4"):
  `0.014638 e-/s`, RMS `0.009786 e-/s`. This is the value subtracted from
  `data.fits`. For comparison, blank patches elsewhere in the (already
  sky-subtracted) frame give `0.0007-0.0022 e-/s` with an RMS of
  `0.0071-0.0073 e-/s`: the annulus sits above them because the pair's own
  extended halo reaches past 4", so the subtraction removes that halo along with
  the sky. `prep.py` prints both numbers on every run.
- **Noise-map.** Built with
  `al.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from`,
  i.e. `sqrt(|counts| + (sky_rms * t)^2) / t`. The exposure-time map `t` is the
  frame's `WHT` extension, which for an AstroDrizzle DRC product is the exposure
  time contributing to each output pixel in seconds (770.7 / 2083.1 / 2093.2 s
  min / median / max across the cutout, against `EXPTIME = 2088 s`). The Poisson
  term uses the counts before sky subtraction, clipped at zero; the sky's own
  Poisson noise, the read noise and the drizzle pixel-to-pixel correlation are
  all carried empirically by the measured background RMS. Every pixel is finite
  and positive. Sky pixels' `data / noise` has mean `0.042` and standard
  deviation `1.018`, so the noise-map is correctly normalised.
  Note that drizzling correlates neighbouring pixels, so a pixel-to-pixel RMS
  slightly *under*-states the uncertainty on a resolution element; this is true
  of every drizzled dataset and is not corrected here.
- **PSF.** The isolated, unsaturated, compact star nearest the lens: frame pixel
  (row 1092, column 1545), **(RA, Dec) = (152.875283, 1.730432)**, 27.6" from the
  lens, peak `14.63 e-/s` — far below the ACS/WFC full well — with no neighbour
  above 2% of its peak within 15 pixels. Measured FWHM **2.20 pixels (0.110")**,
  as expected for drizzled ACS/WFC F814W. The 21 x 21 stamp is local-sky
  subtracted (`0.00906 e-/s`), clipped at zero and normalised to sum 1, with its
  peak on the central pixel.
- **Lens centres.** The two light centroids, in PyAutoLens (y, x) arcsec relative
  to the cutout centre: **(0.3522, -0.2403)** and **(-0.3559, 0.2483)**,
  **separated by 0.8604"** — the ~4.2 kpc projected separation at z = 0.331 that
  makes this a co-dominant pair rather than a lens with a satellite.

## Extra galaxies

Inside the 3.0" radius `start_here.py` fits, the only sources are the two lens
galaxies and the compact lensed images of the source at 1.3-2.0" — no foreground
star, no unrelated galaxy. There is therefore nothing to scale the noise-map over
and **no `mask_extra_galaxies.fits` is written**; `start_here.py` has no
`apply_noise_scaling` step. If you re-run `prep.py` on a field that does need
one, add the contaminants' circles to `extra_galaxy_circles` near the end of the
script and it will write the mask.

## Re-running the preparation

From the workspace root:

```
python dataset/multi_galaxy/sdssj1011+0143/prep.py
```

It downloads the frame to `~/.cache/pyautolens/mast` (change with
`--cache-dir`), skips the download when the frame is already cached, and
overwrites the committed files. It is idempotent: the same frame always produces
byte-identical `data.fits`, `noise_map.fits`, `psf.fits` and
`main_lens_centres.json`; only `info.json` changes, because it records the date
the run was made. `--diagnostic-png` writes data / signal-to-noise / PSF figures
beside the cached frame.

To prepare the **F555W** frame of the same programme instead, add one flag:

```
python dataset/multi_galaxy/sdssj1011+0143/prep.py --filter F555W
```

which fetches `j9qj51010_drc.fits` and runs the identical preparation. Nothing
else in the script changes.

`scripts/imaging/data_preparation/` documents each of these steps in isolation,
for your own data. `prep.py` cuts the archive's own AstroDrizzle DRC product, so
the dataset is reproducible with `astropy` and `astroquery` alone; PyAutoReduce
(`TargetSpec` / `reduce_target`) is the full re-drizzle-from-raw-exposures route,
for users who want to re-reduce the frame rather than re-cut it.

## Citation

Cite **Shu et al. 2016 (ApJ 820, 43)** and the HST programme (GO 10831) in any
publication that uses these data. The raw frames are public STScI archive
products.
