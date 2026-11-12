"""
Features: Subhalo
=================

Use the distortions of strongly lensed arcs to investigate small dark matter halos that may be too faint to
see through their own starlight. This feature connects two scientific tasks: searching for a halo in an
observed lens, and measuring which halos that observation and analysis could have detected.

__Scientific Use Cases__

- **Search for dark substructure:** compare models with and without an additional halo to investigate a local
  arc perturbation and infer candidate halo properties under an adopted mass model.
- **Interpret a non-detection:** measure which halo masses and positions the analysis could have recovered,
  rather than interpreting every absence of a candidate as evidence that no halo is present.
- **Study dark matter populations:** combine detections and calibrated non-detections across lenses with a
  population model to investigate the abundance of small halos and test dark matter scenarios.

This folder provides the simulation and analysis routes. A population-level dark matter inference requires
additional modeling beyond finding a candidate in one lens or producing one sensitivity map.

![An inferred subhalo perturbing a lensed arc](../../../../../images/features/subhalo_detection_illustration.png)

Reused conceptual illustration from the detection introduction. The distortion is exaggerated for clarity;
the dashed marker denotes an inferred halo position, not light emitted by dark matter. The smooth and
perturbed views illustrate the comparison, not a measured detection in this overview.

__Why Lensed Arcs Reveal Small Mass Structures__

A small halo changes the deflection field near an arc, shifting how source structure maps into the observed
image. Its effect depends on its mass profile, position relative to the arcs and the larger lens model, as
well as the source morphology, angular resolution and noise.

A coherent residual can motivate testing an additional halo, but does not identify its cause on its own.
Foreground-light errors, an inadequate source reconstruction or an insufficiently flexible main-lens model
can also leave structure in the residuals. The scientific test is whether a halo interpretation remains
supported when these alternatives and observational assumptions are examined.

__Detection: Is an Additional Halo Favored?__

The [detection introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/detect/start_here.ipynb)
explains how the SLaM analysis fits the main lens and source, searches possible halo positions, and refines a
candidate. It compares models using Bayesian evidence, which averages the likelihood over the parameter priors.

The detection search uses the observed dataset to test alternative models. An evidence improvement is
conditional on those alternatives, their priors and the treatment of the data. Inspect the fitted images,
residuals and stability of the result alongside the evidence comparison. The detection introduction also
links to a scientific lens-sample application and discusses main-lens modeling flexibility.

__Sensitivity: What Could the Analysis Have Detected?__

The [sensitivity introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/start_here.ipynb)
explains injection and recovery: simulate a known halo at a trial position, then fit that simulated dataset
with and without a halo. Repeating the experiment measures how recoverability varies over the chosen trials.

These trials create new simulated observations; they are not additional detections in the original image.
The introductory demonstration samples four positions at one fixed halo mass. Exploring a mass-dependent
selection function requires additional trials, and recovery probabilities can require repeated noise realizations.

Use a recovery criterion consistent with the detection analysis. A non-detection is informative about a halo
population only in combination with the analysis's sensitivity and the assumptions used to calculate it.

__Subhalos and Line-of-Sight Halos__

The simulator in this folder places the perturber at the foreground lens redshift. Halos elsewhere along the
line of sight can also perturb arcs, but their lensing geometry and inferred masses differ. Continue to the
[line-of-sight halos feature](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/los_halos/start_here.ipynb)
when that population is part of the scientific question. An arc anomaly alone does not determine halo redshift.

__Choose An Example__

- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/simulator.ipynb) — Generate imaging with a known halo at the lens redshift.
- [Detect](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/detect/start_here.ipynb) — Search for a candidate in the data and compare models.
- [Sensitivity](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/start_here.ipynb) — Inject halos and measure their recoverability.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
