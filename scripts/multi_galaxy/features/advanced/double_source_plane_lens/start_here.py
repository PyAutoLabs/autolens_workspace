"""
Features: Double Source Plane Lens
==================================

Use sources at different redshifts to constrain one foreground lens system through more than one lensing geometry. Modeling the planes together lets the nearer source also act as a deflector when it carries mass.

__Scientific Use Cases__

- Reconstruct two background sources without assigning their arcs to the same redshift plane.
- Test a mass model against multiple source-plane constraints.
- Build the multi-plane model needed for applications that study geometry or cosmology; a cosmological inference is an additional analysis, not an automatic output of this example.

__This Dataset and Lens Regime__

This folder treats imaging with multiple important deflectors. Include all of them in the ray tracing and
keep their light separate from their mass contributions. The combined deflection can be better constrained
than the allocation of mass between galaxies; examine that allocation when interpreting source properties
or small residual structures.

![double source plane lens](../../../../../images/features/double_source_plane_illustration.png)

Conceptual illustration. The geometry is schematic, not to scale; the example's actual redshifts and mass assignments determine the ray tracing. The schematic illustrates the feature; retain all deflectors in this multi-galaxy example.

__Keep the Planes Distinct__

The two sources do not share the same lensing geometry merely because their arcs appear in the same image.
Their redshifts enter the calculation, and a galaxy that is a source on one plane may also deflect light from
a more distant source when its mass is included.

The model must therefore identify each galaxy's plane and distinguish its light from its mass contribution.
The examples retain their particular redshifts and component choices beside the construction of the model.

__Follow the Geometry Before Fitting__

Start with the simulator or direct fit to see which arcs belong to which source. Then read the modeling
example. Chaining and SLaM examples, where provided, show how the information is carried through a staged fit;
they do not replace the need to check the plane assignments.

Multiple deflectors in group or multi-galaxy examples add components within this geometry. Preserve those
distinctions when adapting a single-deflector example.


__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/simulator.ipynb) — Generate the example dataset.
- [chaining.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/chaining.ipynb) — Fit the model through linked searches.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
