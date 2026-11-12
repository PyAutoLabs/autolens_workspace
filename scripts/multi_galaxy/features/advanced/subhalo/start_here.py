"""
Features: Subhalo
=================

Investigate small dark matter halos through their perturbations of lensed emission. Separate the search for a candidate in the data from the calibration of which halos the same analysis could recover.

__Scientific Use Cases__

- Test whether an arc anomaly favors an additional mass component.
- Interpret non-detections using appropriately matched injection-and-recovery experiments.
- Prepare detection and selection information for a separate halo-population analysis.

__This Dataset and Lens Regime__

This folder treats imaging with multiple important deflectors. Include all of them in the ray tracing and
keep their light separate from their mass contributions. The combined deflection can be better constrained
than the allocation of mass between galaxies; examine that allocation when interpreting source properties
or small residual structures.

![subhalo](../../../../../images/features/subhalo_detection_illustration.png)

Conceptual illustration. The local perturbation is exaggerated, and the marker indicates an inferred halo position rather than light emitted by dark matter. The schematic illustrates the feature; retain all deflectors in this multi-galaxy example.

__Detection and Sensitivity Ask Different Questions__

A detection analysis asks whether the observed data favour a model with a perturber over the chosen
smooth-model alternative. Sensitivity mapping asks which injected perturbers the analysis could recover under
a specified set of assumptions.

A sensitivity map is therefore not a map of detected halos. Likewise, a residual near an arc is not by itself
a subhalo detection: the foreground light, source model and smooth mass model also affect the residuals.

__Choose a Route__

The simulator shows the signature of an injected halo. The detection workflow fits candidate models, while the
sensitivity workflow, where provided, repeats an injection-and-recovery experiment. Each keeps its pipeline
stages and comparison settings in its own start-here example.

Read the appropriate workflow before interpreting a search map or evidence difference. Its result is
conditional on the model alternatives and data treatment used in that analysis.


__Choose An Example__

- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/subhalo/simulator.ipynb) — Generate the example dataset.
- [Detect](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/subhalo/detect/start_here.ipynb) — follow this feature’s introduction and worked examples.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
