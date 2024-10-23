The ``simulators/examples/advanced`` folder contains advanced example scripts for simulating imaging
datasets (E.g. Hubble Space Telescope).

Files (Advanced)
----------------

- ``lens_light_asymmetric.py``: The lens galaxy light is an asymmetric combination of Gaussian light profiles.
- ``mass_power_law.py``:The lens galaxy mass is a power-law profile.
- ``mass_stellar_dark.py``: The lens galaxy mass is a combination of a stellar mass and dark matter halo.
- ``light_operated.py``: The lens galaxy light includes point source emission which is a Sersic profile already operated on by the PSF.
- ``dark_matter_subhalo.py``: The lens galaxy mass includes a dark matter subhalo which overlaps the lensed source emission.
- ``x2_lens_galaxies.py``: The lens galaxy is two galaxies with Sersic light profiles and Isothermal mass profiles.
- ``extra_galaxies.py``: The lens galaxy has the emission of extra galaxies in the image and needs removing or modelling.
- ``double_einstein_ring.py``: The lens is a double Einstein ring system with two lensed sources at different redshifts.
- ``sky_background.py``: Simulate a strong lens with a sky background which is not subtracted from the image.