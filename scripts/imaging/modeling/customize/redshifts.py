"""
Customize: Redshifts
====================

The redshift of the lens and source galaxies are also input.

All `LightProfile`'s and `MassProfiles`'s in  **PyAutoLens** are defined in units that mean the redshifts do not
change the modeling results. For example, because the *einstein_radius_ parameter of the `Isothermal` is
in units arc-seconds, changing the lens and source redshifts does not change the model behaviour.

Changing the redshifts will only change how quantities in arc-seconds are converted to units such as kilo-parsecs, or
how an Einstein mass is converted to Solar Masses, etc.
"""
