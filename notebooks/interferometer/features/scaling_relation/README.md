The `scaling_relation` folder contains example scripts showing how to include a population of foreground galaxies in an
interferometer lens model by tying their masses to the main lens's, rather than freeing each one.

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

`einstein_radius_anchor` is the main lens's own `einstein_radius`, which the model already fits, so the tier adds
**zero free parameters** however many galaxies it holds.

# Mass only, and where the luminosities come from

Foreground galaxy light is rarely detected at the long wavelengths an interferometer probes, so these examples model
companion **mass only** — as do all the interferometer `extra_galaxies` examples.

That settles the luminosity question differently from the CCD-imaging version of this feature: the luminosities
**cannot** be measured from this dataset, because the foreground light is not in it. They come from ancillary optical or
near-infrared imaging of the same field — the same imaging that supplies the companions' centres, since visibilities
supply neither.

`slam.py` here is explicit about the consequence: an interferometer SLaM pipeline has no `light_lp` stage at all, which
is exactly the stage the imaging pipeline measures its luminosities in. This pipeline threads externally measured
luminosities through instead. That is not purely a limitation — luminosities that never depend on the fit cannot be
biased by it.

Mass profiles are **untruncated**: truncation encodes tidal stripping by a host halo's potential, which a galaxy-scale
lens does not have. Truncated `dPIEMass` members belong to the group- and cluster-scale workflows.

# Files

- `simulator`: Simulating a lens plus five tied companions, with truth masses derived from the relation.
- `modeling`: Lens modeling with the tier tied to the main lens.
- `fit`: The same composition without a search, showing the real-space deflection sum before the Fourier transform.
- `likelihood_function`: The one step of the uv-plane likelihood a scaling relation changes.
- `slam`: The Source, Light and Mass pipeline, and why it cannot measure the luminosities.

# Related

- `interferometer/features/extra_galaxies`: companions modelled with individual freedom, plus the
  `TransformerNUFFT` + sparse-operator machinery for large visibility sets.
- `imaging/features/scaling_relation`: the CCD-imaging version, which models companion light and whose `slam.py`
  measures its own luminosities.
- `group/features/scaling_relation` and `cluster/modeling`: the reference-magnitude (Lenstool `mag0`) normalisation,
  which costs one free parameter and does not assume a single anchoring galaxy.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
