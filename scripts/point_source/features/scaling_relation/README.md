The `scaling_relation` folder contains example scripts showing how to include a population of foreground galaxies in a
point-source lens model by tying their masses to the main lens's, rather than freeing each one.

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

`einstein_radius_anchor` is the main lens's own `einstein_radius`, which the model already fits, so the tier adds
**zero free parameters** however many galaxies it holds.

# This is the regime where the relation matters most

A quadruply imaged point source gives 8 positional data points; adding fluxes brings it to 12. That is the whole
budget. In the example here, tying five companions gives a model with **8** free parameters; freeing them gives
**13** — more parameters than the data has points.

And the companions cannot simply be dropped. Removing the tier moves the four multiple images by **182, 398, 1596 and
1633 mas**, against 5 mas astrometric precision. A multiple image position is the *solution* of the lens equation, not
a linear readout of the deflection field, so a 0.2" deflection near the ring slides an image much further than 0.2".
The per-member deflections here are 150-300 mas but the resulting image shifts reach 1633 mas — the lens equation
amplifies.

So the tier must be in the model, and the relation is what makes having it there affordable.

# Mass only, and no slam.py

A `PointDataset` is positions and fluxes, not an image. There is no companion light in it to blend with anything,
nothing to mask, nothing to noise-scale. These examples are mass-only.

It follows that neither the centres nor the luminosities can come from the data being fitted — both are measured from
the **accompanying imaging** the positions were extracted from, which the simulator writes to `data.fits` so you can
see the galaxies the numbers refer to. There is no `slam.py` in this folder for the same reason: with no light in the
data there is no light stage in which to measure anything.

Mass profiles are **untruncated**: truncation encodes tidal stripping by a host halo's potential, which a galaxy-scale
lens does not have. Truncated `dPIEMass` members belong to the group- and cluster-scale workflows.

# Files

- `simulator`: Simulating the quad plus five tied companions, and the accompanying imaging the luminosities come from.
- `modeling`: Lens modeling with the tier tied to the main lens, and the parameter count against the data budget.
- `fit`: The same composition without a search, measuring how far the tier moves each multiple image.
- `likelihood_function`: The four-step chain from the relation to the chi-squared, and where the amplification happens.

# Related

- `point_source/features/extra_galaxies`: companions modelled with individual freedom, and when that is affordable.
- `imaging/features/scaling_relation`: the CCD-imaging version, which models companion light and measures its own
  luminosities in `slam.py`.
- `group/features/scaling_relation` and `cluster/modeling`: the reference-magnitude (Lenstool `mag0`) normalisation.
  Cluster-scale analyses fit many point-source families at once and lean heavily on relations like this one.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
