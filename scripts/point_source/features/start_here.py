"""
Point Source Features
=====================

These features build on the topic’s main example. Each introduction explains the method and its trade-offs; the worked scripts retain their model choices and detailed calculations.

__Choose a Feature__

- [Deblending](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/deblending/start_here.ipynb): Compact lensed images can overlap the foreground galaxy or other extended light. Deblending fits their contributions so the point-image positions and other relevant quantities can be measured.
- [Extra Galaxies](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/extra_galaxies/start_here.ipynb): A target rarely occupies an otherwise empty field. Nearby galaxies may contribute light inside the region being fitted, and a lensing model may also need to account for their deflection of the light rays.
- [Multiple Sources](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/multiple_sources/start_here.ipynb): Several background point sources can be lensed by the same foreground system. Their multiple images must be associated with the correct source, and sources at different redshifts require the corresponding lensing geometry.
- [Scaling Relation](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/scaling_relation/start_here.ipynb): A scaling relation connects the properties of a galaxy population through a shared rule, rather than giving every member a completely independent model. In these lensing examples, member luminosities help determine their mass normalizations.

__Additional Examples__

- [fluxes.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/fluxes.ipynb): Modeling: Fluxes.
- [time_delays.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/time_delays.ipynb): Modeling: Time Delays.

For the common setup, begin with [point_source/start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/start_here.ipynb). Run Python scripts from the workspace root.
"""
