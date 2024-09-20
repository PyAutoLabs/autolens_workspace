The ``simulators/examples`` folder contains example scripts for simulating imaging datasets (E.g.
Hubble Space Telescope) using functionality not described in ``start_here.ipynb``.

Files (Beginner)
----------------

These scripts simulating CCD imaging of a strong lens where:

- ``no_lens_light.py``: The lens galaxy light is not included.
- ``no_lens_light.py``: The lens galaxy light is not included and the lens galaxy mass is a Isothermal sphere profile.
- ``no_lens_light.py__Sersic_no_Core``: The source is a regular Sersic profile, to illustrated over sampling.
- ``lens_sersic.py``: The lens galaxy light is a Sersic profile.
- ``source_x2.py``: The source galaxy is a double Sersic profile.
- ``source_complex.py``: The source galaxy is complex (a combination of many Sersic profiles).
- ``manual_signal_to_noise_ratio``: Use light profiles where the signal-to-noise of the lens and lensed images are input.