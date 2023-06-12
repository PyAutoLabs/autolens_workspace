The calculation of many quantities from light profiles and mass profiles, for example their image, convergence
or deflection angles are ill-defined at (y,x) coordinates (0.0, 0.0). This can lead **PyAutoLens** to crash if not
handled carefully.

The *radial_minimum.yaml* config file defines, for every profile, the values coordinates at (0.0, 0.0) are rounded to
to prevent these numerical issues. For example, if the value of a profile is 1e-8, than input coordinates of (0.0, 0.0)
will be rounded to values (1e-8, 0.0).