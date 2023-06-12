The ``sample`` folder contains example scripts for simulating imaging datasets (E.g. Hubble Space Telescope)
consisting of large samples of strong lenses.

This is done by creating a model of the sample's overall population parameters and drawing lens model parameters
from these prior distributions.

Files (Advanced)
----------------

These scripts simulate samples of strong lenses consisting of:

- ``mass_power_law.py``: Lenses where the mass distribution is a power-law.
- ``mass_bpl.py``: Lenses where the mass distribution is a broken power-law.
- ``double_einstein_ring``: The lens is a double Einstein ring system with two lensed sources at different redshifts.