The `group/data_preparation` package has no scripts of its own.

Every tool needed to prepare group-scale CCD imaging for **PyAutoLens** lives in
`imaging/data_preparation` — the image, noise-map and PSF standards, and the GUI
tools for marking centres, masks and positions.

# Group-Specific Inputs

A group-scale model additionally needs the galaxy tiers specified up front:

- `main_lens_centres.json`: centres of the free main lens galaxies.
  Mark them with `imaging/data_preparation/gui/lens_light_centre.py`.
- `extra_galaxies_centres.json`: centres of bounded companions.
  Mark them with `imaging/data_preparation/gui/extra_galaxies_centres.py`.
- `scaling_galaxies.csv`: `y, x, luminosity` for the scaling tier.

See `group/start_here` and `group/features/scaling_relation` for how each is
loaded into a model.
