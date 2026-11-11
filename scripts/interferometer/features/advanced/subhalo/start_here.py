"""
Features: Subhalo
=================

A dark matter subhalo can perturb lensed emission without supplying visible light of its own. The effect
depends on the halo properties, its location relative to the arcs and the larger lens model.

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

__In This Folder__

These examples fit visibilities. The image-plane light prediction is transformed into the measured
Fourier-domain quantities; a dirty image is an inspection product, not a substitute for that likelihood. Keep
the visibility sampling, noise and model-specific response choices in view when adapting an imaging example.

__Choose An Example__

- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/subhalo/simulator.ipynb) — Generate the example dataset.
- [Detect](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/subhalo/detect/start_here.ipynb) — follow this feature’s introduction and worked examples.
- [Sensitivity](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/subhalo/sensitivity/start_here.ipynb) — follow this feature’s introduction and worked examples.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
