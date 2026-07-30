The `extra_galaxies` folder contains example scripts showing how to perform analysis of point-source data using
the extra galaxies API, which includes galaxies surrounding a strong lens in the lens model, accounting for their
mass in the ray-tracing.

For point sources the extra galaxies API is mass-only. A `PointDataset` contains image positions and fluxes, not
an image, so there is no extra galaxy light to mask, noise-scale or fit — the only question is whether their mass
is included in the model. It usually matters more here than for extended sources, because multiple image
positions are extremely sensitive to perturbing mass.

# Files

The following example scripts illustrate point-source lens modeling where:

- `modeling`: Point-source lens modeling using a model which includes extra galaxies.
- `simulator`: Simulating point-source data of a lens with extra galaxies surrounding it.

There is no `slam` example in this folder. The Source, Light and Mass pipelines are built around extended-source
imaging and interferometer data; see `imaging/features/extra_galaxies/slam` for that workflow.

# Results

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
