The `imaging/data_preparation/gui` package holds interactive tools for marking up an imaging dataset by hand:
painting masks with the mouse (`Scribbler`) and clicking positions (`Clicker`). None of them are required; each
writes one product into the dataset's folder that the modeling scripts can then load.

# Recommended Order

The mask tools build on one another, so run them in this order:

1. `mask_extra_galaxies`: paint the CONTAMINANTS to remove (neighbouring galaxies, stars, artefacts). Do this
   first, because every later step shows this mask so that a contaminant is never mistaken for a lensed image.
2. `positions`: click the multiple images of the lensed source. The positions are shown as crosses in the later
   mask tools, and a modeling script can use them to reject unphysical mass models.
3. `mask_arcs` (optional): paint the lensed ARCS to keep, guided by the contaminant mask outline and the position
   crosses, with the lens light subtracted from the display so the arcs are visible. Only needed for source-only
   fits, arc signal-to-noise measurements or a source-plane analysis.

`mask` (the region to fit, usually just a circle), `lens_light_centre` and `extra_galaxies_centres` are
independent of the sequence above and can be run at any point.

Every mask tool can be reopened on its own product to refine it rather than redraw it (`refine_existing = True`
in each script), and for a multi-wavelength dataset a mask drawn on one waveband can be regridded onto the next and
reviewed there (`mask_extra_galaxies_multi`).

# Where Files Belong

Every GUI reads the dataset from, and writes its product back into, that dataset's own folder under `dataset/`, next
to the `data.fits` / `noise_map.fits` / `psf.fits` it describes. Nothing goes in `output/`, which is reserved for
model-fit results.

Single waveband:

```
dataset/imaging/<dataset_name>/
    data.fits                      image                              (data_preparation/examples)
    noise_map.fits                 noise-map                          (data_preparation/examples)
    psf.fits                       PSF                                (data_preparation/examples)
    mask_extra_galaxies.fits       contaminants, True = masked        (gui/mask_extra_galaxies)      step 1
    positions.json                 (y, x) arcsec of the lensed images (gui/positions)                step 2
    mask_arcs.fits                 arcs, True = masked (everything    (gui/mask_arcs)                step 3
                                   except the painted arcs)
    mask_gui.fits                  region to fit, True = masked       (gui/mask)
    light_centre.json              lens light centre                  (gui/lens_light_centre)
    extra_galaxies_centres.json    extra galaxy centres               (gui/extra_galaxies_centres)
    info.json                      redshifts etc.                     (by hand)
```

Multi-wavelength, one sub-folder per waveband, each holding its own copy of the products above:

```
dataset/imaging/<dataset_name>/wavebands/<waveband>/
    data.fits  noise_map.fits  psf.fits
    mask_extra_galaxies.fits  positions.json  mask_arcs.fits  ...
```

Every `.fits` mask is stored in `Mask2D`'s own convention (`True` = excluded from the fit) and loads with
`al.Mask2D.from_fits(file_path=..., pixel_scales=...)`; the scripts note where a painted region is inverted before
saving. Positions and centres load with `al.from_json`.

# Files

- `mask_extra_galaxies`: paint the extra galaxies / contaminants to remove (step 1).
- `positions`: click the multiple images of the lensed source (step 2).
- `mask_arcs`: paint the lensed arcs to keep, over the products of steps 1 and 2 (optional step 3).
- `mask_extra_galaxies_multi`: draw one waveband's contaminant mask and regrid it onto the next for review.
- `mask`: paint the region of the image to fit.
- `lens_light_centre`: click the lens galaxy's light centre.
- `extra_galaxies_centres`: click the centres of the extra galaxies.

# Keys

All painting GUIs share the same keys: `1` green brush adds, `2` red brush erases, `=` / `-` grow / shrink the
brush, `z` undoes the last stroke, `Esc` finishes.
