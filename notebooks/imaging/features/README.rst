The ``modeling/features`` folder contains example scripts showing how to fit a lens model to imaging data using
different **PyAutoLens** features.

The scripts in this folder are all recommend, as they provide tools which make lens modeling more reliable and efficient.
Most users will benefit from these features irrespective of the quality of their data, complexity of their lens model
and scientific topic of study.

Folders
-------

The following example scripts illustrating lens modeling where:

- ``no_lens_light``: The foreground lens's light is not present in the data and thus omitted from the model.
- ``extra_galaxies``: Modeling which account for the light and mass of extra nearby galaxies.
- ``linear_light_profiles``: The model includes light profiles which use linear algebra to solve for their intensity, reducing model complexity.
- ``multi_gaussian_expansion``: The lens (or source) light is modeled as ~25-100 Gaussian basis functions
- ``pixelization``: The source is reconstructed using an adaptive rectangular or Delaunay mesh
- ``scaling_relation``: Use scaling relations, for example relating light and mass, to compose lens models with few parameters for many galaxies.
- ``advanced``: Advanced features for expert users, for example modeling with multiple datasets simultaneously.

Files
-----

- ``simulator_manual_signal_to_noise``: Simulate imaging data where the input is not galaxy intensities but their output S/N.

Notes
-----

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/guides/results``.