import numpy as np
import autolens as al
import autogalaxy as ag
import astropy.cosmology as cosmology

"""
Author: Kaihao Wang

This guide is in development and not written with extensive text and descrptions (yet!).

In a nutshell, it shows how to conver the units of a Sersic mass profile from PyAutoLens's internal units ot physical
units like solar masses. The code therefore also includes conversion of a Sersic light profile from internal units
(e.g. counts/s/arcsec^2) to physical units (e.g. solar luminosity).

If you read this and anything is unclear please join the SLACK channel and ask a quesiton in the #generla channel,
we can then work on making this script more clear :).
"""


class mass_profile(al.mp.Sersic):
    """
    This is an example light-trace-mass mass profile showing how to calculate signals on each pixel
    from the intensity of the light profile and mass-to-light ratio in M_sun/L_sun.

    I'll let the light profile to be a Sersic sphere.
    What I aim to do is to make the units of mass intensity = intensity * mass-to-light ratio
    to be critical_surface_density / (counts/s) correctly.

    Note that the units of intensity must be counts / s / arcsec^2, since astropy's output of critical
    surface density is in units of solar mass / arcsec^2
    """

    def __init__(
        self,
        redshift_lens: float,
        redshift_source: float,
        solar_magnitude: float,  # absolute magnitude for a given band
        effective_radius: float,
        sersic_index: float,
        centre=(0.0, 0.0),
        zero_point: float = 25.23,
        intensity: float = 1.0,  # units: counts/s/arcsec^2
        mass_to_light_ratio: float = 1.0,  # units: M_sun/L_sun
        cosmo=ag.cosmo.FlatLambdaCDMWrap(),
    ) -> None:
        # critical surface density in solar mass / arcsec^2
        self.critical_surface_density = (
            cosmo.critical_surface_density_between_redshifts_from(
                redshift_0=redshift_lens, redshift_1=redshift_source
            )
        )
        self.redshift = redshift_lens
        self.cosmo = cosmo

        super().__init__(
            centre=centre,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio
            * self.unit_mass2light_instrument(solar_magnitude, zero_point),
        )

    def magnitude_absolute2apparent(self, mag):
        """
        Calculate the apparent magnitude of an object with a given absolute magnitude and on a certain redshift
        """
        distance = self.cosmo.luminosity_distance(self.redshift) * 1e6
        return mag + 5 * np.log10(distance.value / 10)

    def mag2counts(self, mag, zero_mag=25.23):
        """
        Convert apparent magnitude to counts in e-/s
        """
        return 10 ** ((zero_mag - mag) / 2.5)

    def unit_mass2light_instrument(self, solar_magnitude, zero_point):
        """
        Calculate a factor so that (mass-to-light ratio * factor * intensity) could be equal to kappa
        This is the key of mass-to-light ratio units conversion.
        """

        solar_magapp = self.magnitude_absolute2apparent(solar_magnitude)
        counts_per_sec_per_solar_luminosity = self.mag2counts(solar_magapp, zero_point)

        return 1 / self.critical_surface_density / counts_per_sec_per_solar_luminosity


if __name__ == "__main__":
    # This is a Sersic light profile at redshift = 0.2, whose absolute magnitude is -15.4, corresponding to an apparent magnitude ~ 24.6
    light = al.lp.Sersic(
        centre=(0, 0),
        intensity=0.892476,  # unit: counts/s/arcsec^2
        effective_radius=0.381768,
        sersic_index=1.3,
    )

    # Assuming it's mass-to-light ratio is 2 and the absolute magnitude of the sun is 4.83,
    # this turns out a total mass of 2.47 * 10^8 times solar mass
    # Let's check if my light-trace-mass profile can give the right answer.
    mass = mass_profile(
        redshift_lens=0.2,
        redshift_source=1.0,
        solar_magnitude=4.83,
        intensity=0.892476,
        effective_radius=0.381768,
        sersic_index=1.3,
        mass_to_light_ratio=2,
        zero_point=25.23,
    )

    from scipy.integrate import quad

    def delta_mass(r):
        # the length scale of MassProfile is defined on the unit of arcsec, so that r is in unit of arcsec
        # thus, critical_surface_density should also be defined in per arcsec^2 unit
        return mass.convergence_func(r) * mass.critical_surface_density * 2 * np.pi * r

    total_mass, _ = quad(delta_mass, 0, np.inf)

    print(mass.mass_to_light_ratio)
    print("The total mass of this galaxy should be ~ 2.47 * 10^8")
    print(f"While the result of the light-trace-mass profile is {total_mass:.4e}")
