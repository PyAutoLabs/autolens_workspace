The `operated_light_profile` folder contains example scripts showing how to perform analysis of
`Interferometer` data using operated light profiles, which represent compact point-source emission (e.g. an
AGN) whose image-plane shape is specified directly.

For interferometer data there is no PSF, so operated light profiles are Fourier transformed to the visibility
plane like every other light profile — the PSF-bypass behaviour of the imaging examples applies only where a
PSF exists. Using them keeps a lens model consistent across imaging and interferometer datasets.

# Files

The following example scripts illustrating lens modeling where:

- `modeling`: Lens modeling of an `Interferometer` dataset using operated light profiles.
- `simulator`: Simulating interferometer data of a strong lens using operated light profiles.

# Results

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
