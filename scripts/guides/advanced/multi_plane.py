"""
Misc: Multi-Plane
=================

Multi-plane ray-tracing is used when there are more planes than just an image-plane and source-plane. When tracing
from one plane to another, the redshifts of the different planes must be used to determine scaling factors that are
applied to the deflection angles.

There are different formalisms for multi-plane ray-tracing, PyAutoLens follows the formalism described in
this paper: ?.

Examples of multi-plane lensing systems include:

 - A standard lens galaxy and source galaxy system, but where there is also a dark matter subhalo whose redshift is
 not at the redshift of the lens galaxy.

 - A strong lens system where the deflection due to many dark matter halos down the line-of-sight are included, which
 may be at a large range of different redshifts.

 - A galaxy cluster, where the observed different background source galaxies are at a range of different redshifts
 and their deflections due to one another must be included.

__Example__

To illustrate multi-plane ray-tracing, we first set up a simple lens system, using a `Tracer` object.

We'll make things simple and assume 3 galaxies at redshifts 0.5, 1.0 and 2.0. We'll use a singular isothermal sphere
for each galaxy's mass profile.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from typing import List, Optional, Union

import autoarray as aa
import autolens as al

lens_0 = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))
lens_1 = al.Galaxy(redshift=1.0, mass=al.mp.IsothermalSph(einstein_radius=1.0))
lens_2 = al.Galaxy(redshift=2.0, mass=al.mp.IsothermalSph(einstein_radius=1.0))

"""
Multi-plane ray tracing is based on the redshifts of the planes that make up the lens system, as opposed to the
redshifts of the galaxies. 

These two things are equivalent, but it means we need to set up the above galaxies as planes in order to perform
multi-plane ray-tracing.
"""
galaxies_0 = al.Galaxies(galaxies=[lens_0])
galaxies_1 = al.Galaxies(galaxies=[lens_1])
galaxies_2 = al.Galaxies(galaxies=[lens_2])

"""
__Ray Tracing__

Multi-plane ray tracing is implemented in the `tracer_util.py` module of the following package:

https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/lens/tracer_util.py

It uses the function `traced_grid_2d_list_from`.

Multi-plane ray-tracing also heavily relies on the `scaling_factor_between_redshifts_from` function, which is
implemented in the `cosmology` package of autolens.

I have copy and pasted both functions below, and put print statements in to show how they works.
"""


def scaling_factor_between_redshifts_from(
    cosmology, redshift_0: float, redshift_1: float, redshift_final: float
) -> float:
    """
    For strong lens systems with more than 2 planes, the deflection angles between different planes must be scaled
    by the angular diameter distances between the planes in order to properly perform multi-plane ray-tracing.

    For a system with a first lens galaxy l0 at `redshift_0`, second lens galaxy l1 at `redshift_1` and final
    source galaxy at `redshift_final` this scaling factor is given by:

    (D_l0l1 * D_s) / (D_l1* D_l1s)

    The critical surface density for lensing, often written as $\sigma_{cr}$, is given by:

    critical_surface_density = (c^2 * D_s) / (4 * pi * G * D_ls * D_l)

    D_l0l1 = Angular diameter distance of first lens redshift to second lens redshift.
    D_s = Angular diameter distance of source redshift to earth
    D_l1 = Angular diameter distance of second lens redshift to Earth.
    D_l1s = Angular diameter distance of second lens redshift to source redshift

    For systems with more planes this scaling factor is computed multiple times for the different redshift
    combinations and applied recursively when scaling the deflection angles.

    Parameters
    ----------
    redshift_0
        The redshift of the first strong lens galaxy.
    redshift_1
        The redshift of the second strong lens galaxy.
    redshift_final
        The redshift of the source galaxy.
    """
    angular_diameter_distance_between_redshifts_0_and_1 = (
        cosmology.angular_diameter_distance_z1z2(z1=redshift_0, z2=redshift_1)
        .to("kpc")
        .value
    )

    angular_diameter_distance_to_redshift_final = (
        cosmology.angular_diameter_distance(z=redshift_final).to("kpc").value
    )

    angular_diameter_distance_of_redshift_1_to_earth = (
        cosmology.angular_diameter_distance(z=redshift_1).to("kpc").value
    )

    angular_diameter_distance_between_redshift_1_and_final = (
        cosmology.angular_diameter_distance_z1z2(z1=redshift_0, z2=redshift_final)
        .to("kpc")
        .value
    )

    return (
        angular_diameter_distance_between_redshifts_0_and_1
        * angular_diameter_distance_to_redshift_final
    ) / (
        angular_diameter_distance_of_redshift_1_to_earth
        * angular_diameter_distance_between_redshift_1_and_final
    )


def traced_grid_2d_list_from(
    planes: Union[List[List[al.Galaxy]], List[al.Galaxies]],
    grid: aa.type.Grid2DLike,
    cosmology: al.cosmo.LensingCosmology = al.cosmo.Planck15(),
    plane_index_limit: int = Optional[None],
):
    """
    Returns a ray-traced grid of 2D Cartesian (y,x) coordinates which accounts for multi-plane ray-tracing.

    This uses the redshifts and mass profiles of the galaxies contained within the tracer to perform the multi-plane
    ray-tracing calculation.

    This function returns a list of 2D (y,x) grids, corresponding to each redshift in the input list of planes. The
    plane redshifts are determined from the redshifts of the galaxies in each plane, whereby there is a unique plane
    at each redshift containing all galaxies at the same redshift.

    For example, if the `planes` list contains three lists of galaxies with `redshift`'s z0.5, z=1.0 and z=2.0, the
    returned list of traced grids will contain three entries corresponding to the input grid after ray-tracing to
    redshifts 0.5, 1.0 and 2.0.

    An input `AstroPy` cosmology object can change the cosmological model, which is used to compute the scaling
    factors between planes (which are derived from their redshifts and angular diameter distances). It is these
    scaling factors that account for multi-plane ray tracing effects.

    The calculation can be terminated early by inputting a `plane_index_limit`. All planes whose integer indexes are
    above this value are omitted from the calculation and not included in the returned list of grids (the size of
    this list is reduced accordingly).

    For example, if `planes` has 3 lists of galaxies, but `plane_index_limit=1`, the third plane (corresponding to
    index 2) will not be calculated. The `plane_index_limit` is used to avoid uncessary ray tracing calculations
    of higher redshift planes whose galaxies do not have mass profile (and only have light profiles).

    Parameters
    ----------
    galaxies
        The galaxies whose mass profiles are used to perform multi-plane ray-tracing, where the list of galaxies
        has an index for each plane, correspond to each unique redshift in the multi-plane system.
    grid
        The 2D (y, x) coordinates on which multi-plane ray-tracing calculations are performed.
    cosmology
        The cosmology used for ray-tracing from which angular diameter distances between planes are computed.
    plane_index_limit
        The integer index of the last plane which is used to perform ray-tracing, all planes with an index above
        this value are omitted.

    Returns
    -------
    traced_grid_list
        A list of 2D (y,x) grids each of which are the input grid ray-traced to a redshift of the input list of planes.
    """

    traced_grid_list = []
    traced_deflection_list = []

    redshift_list = [galaxies[0].redshift for galaxies in planes]

    for plane_index, galaxies in enumerate(planes):
        scaled_grid = grid.copy()

        if plane_index > 0:
            for previous_plane_index in range(plane_index):
                scaling_factor = cosmology.scaling_factor_between_redshifts_from(
                    redshift_0=redshift_list[previous_plane_index],
                    redshift_1=galaxies[0].redshift,
                    redshift_final=redshift_list[-1],
                )

                scaled_deflections = (
                    scaling_factor * traced_deflection_list[previous_plane_index]
                )

                scaled_grid -= scaled_deflections

        traced_grid_list.append(scaled_grid)

        if plane_index_limit is not None:
            if plane_index == plane_index_limit:
                return traced_grid_list

        deflections_yx_2d = sum(
            map(lambda g: g.deflections_yx_2d_from(grid=scaled_grid), galaxies)
        )

        traced_deflection_list.append(deflections_yx_2d)

    return traced_grid_list


"""
__Example__

The code below ray-traces a Cartesian coordinate y=1.0", x=0.0" to redshift 0.5, 1.0 and 2.0 via multi-plane
ray-tracing.

The print statements show how the coordinates are transformed as they are ray-traced through each plane and
therefore how the multi-plane ray-tracing algorithm works.
"""
grid = al.Grid2DIrregular(values=[(1.0, 0.0)])

traced_grid_2d_list_from(
    planes=[[galaxies_0], [galaxies_1], [galaxies_2]],
    grid=grid,
)

"""
__Profiles With Physical Units__

The above ray-tracing used dimensionless angular units (e.g. the grid was in arc-seconds and mass profile quantities 
like the `einstein_radius` were in arc-seconds).

For certain mass profiles, we define them in physical units (e.g. kpc, solar masses). For example, for the dark matter
NFW profile called `NFWMCRLudlow` in **PyAutoLens**, it is defined physically with a `mass_at_200` parameter,
which is the mass in solar masses at which the density profile drops to 200 times the critical density of the Universe.

All internal **PyAutoLens** calculations use dimensionless units, irrespective of whether a mass profile is defined
in angular dimensionless units of physical units. Therefore, when a physical mass profile is set up, an internal 
conversion is performed which converts its parameters to dimensionless units. This typically requires a cosmology,
the mass profile redshift and the redshift of the highest redshift plane in the multi-plane system, which are 
often input parameters of physical mass profiles.

For example, when setting up the ``NFWMCRLudlow``'s  `mass_at_200`, an internal conversion of this value to the 
dimensionless value used for NFWs, `kappa_s`, is performed. This uses the lens's critical surface mass density, 
`sigma_crit`. This is computed using a  cosmology, the NFW redshift and the redshift of the highest redshift 
galaxy (`redshift_source`).

The `scaling_factor` of multi-plane ray-tracing is based on ratios of `sigma_crit` values at different redshifts. For 
NFW profiles in physical units, this can create ambiguity whether the `scaling_factor`'s being applied in multi-plane 
ray-tracing systems are consistent with the `sigma_crit` values used to set up the physical mass profiles values. 

**PyAutoLens** uses a convention such that for every physical mass profile in a multi-plane system, their input
`redshift_source` parameters are the highest redshift plane in the system. When multi-plane ray-tracing algorithm 
computes the `scaling_factor`  between planes it correctly scales the lensing parameter (e.g. the `kappa_s` values 
for NFWs), in order to produce the correct deflection angles.

The factor which converts between the physical lens mass and it's lensing strength is sigma_crit. **PyAutoLens**, 
always interprets this with `redshift_source` = `redshift_max_plane`. Therefore, for any profile, if you want the 
projected mass associated with it at some point, you can multiply kappa at that point by sigma_crit(z_profile, z_max).

__SLACK__

This script was written after discussion on the PyAutoLens Slack channel, where some users modeling cluster-scale
lenses wanted to know how to perform multi-plane ray-tracing in physical units. The following text is the SLACK 
conversion, which if you read in detail should help you fully understand the autolens implementation and details
of the issue.



Jack

Hi all - I am working with an undergrad to model cluster-scale lenses with PyAutoLens.

We need to sample in physical units, rather than dimensionless quantities. For example, we want to sample an 
NFW by (log10(M200), c200), rather than (kappa_s, theta_s).

To do so, we will likely be making our own classes along these lines:

class PhysicalNFW(autogalaxy.NFW):
    def __init__(self, center, ell_comps, logM200m, c200m, cosmology):
        kappa_s = ... computing rhos*rs ... / cosmology.critical_density()
        theta_s = ... computing rs ... / cosmology.angular_diamter_distance()
        super(autogalaxy.NFW, self).__init__(
            center=center, ell_comps=ell_comps,
            kappa_s=kappa_s, scale_radius=theta_s
        )

That seems wonderfully nice and simple!
However, the critical density is ambiguous when we have multiple lens and source planes (our clusters have multiple 
sources at different redshifts). With multiple source planes, which zs should be used to compute the critical density 
to give the right behavior? Is it the redshift of i.e. the next plane after the halo, or the last plane?


Jam

 However, the critical density is ambiguous when we have multiple lens and source planes (our clusters have multiple 
 sources at different redshifts). With multiple source planes, which zs should be used to compute the critical density 
 to give the right behavior? Is it the redshift of i.e. the next plane after the halo, or the last plane?
I have no idea, but I would guess it comes out as a lot of ratios of angular diameter distances. (edited) 



Jack

I think it's an implementation detail of PyAutoLens: when a mass profile defines a bunch of deflection angles, 
which planes does PyAutoLens interpret them as deflections between?


Jack

If the mass is in plane i, is it between plane i and i+1?


Andrew

So, without being an expert on the internals of PyAutoLens...
I'm going to use "critical density" to 
mean 3H^2/8.pi.G (https://en.wikipedia.org/wiki/Friedmann_equations#Density_parameter), i.e. the 3D density for a 
spatially flat Universe, and "Sigma_crit" to mean the critical surface density 
for lensing (https://en.wikipedia.org/wiki/Gravitational_lensing_formalism) (from @Jack's pseudo-code, I'm guessing 
cosmology.critical_density() is Sigma_crit?)
You will need the critical density at the lens redshift to convert from M200 and c to more physical NFW 
parameters (i.e. rho_0 and R_s here https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile, I 
guess @Jack's rhos, rs).

Projecting this, you get the surface density as a function of position in the image plane (i.e. Sigma(theta))
One would normally divide by Sigma_crit to get kappa(theta). But it doesn't really make sense to define some 
convergence normalisation (i.e.) kappa_s for autogalaxy's NFW, because with multiple source planes there is no 
one convergence field (because Sigma_crit depends on z_s). So @Jam, do you have a standard way to deal with multiple 
source planes?

That said, the convergences for the different source planes are just re-scaled versions of one another, as are the 
shear and the deflection angles (though not things like the magnification). So, for example: if you want to know the 
mapping from image plane coordinates to (multiple) source plane coordinates over a grid of image plane positions, 
you could calculate a deflection angle field for one source plane (which might be costly, involve numerical 
integrals, etc.) and then the deflection angle for some other source plane can be found easily be re-scaling by the 
ratio of 'Sigma_crit's between the two different source redshifts
In terms of multiple source planes, @Jack, what is the data you intend to fit to / how do you intend to do your fit? 
By which I mean, people fitting clusters often treat galaxies more like point sources than people fitting to 
galaxy-galaxy strong lensing (the cluster people often just want their lens model to get the different multiple 
images of each multiply imaged background galaxy to map back to common positions behind the cluster, as opposed to 
caring about the structure of each lensed image), but I've typically seen PyAutoLens used to fit an observed image 
pixel-by-pixel (often with a pixelised source reconstruction). If you intend to do the latter (with pixelised sources) 
then you would have multiple pixelised source planes, each with their own regularisation, and (I imagine) the linear 
algebra to find the most likely set of pixel fluxes across all the source planes would be rather difficult.
Not sure that will have helped, but my two cents...

WikipediaWikipedia
Friedmann equations
https://en.wikipedia.org/wiki/Friedmann_equations#Density_parameter


WikipediaWikipedia
Gravitational lensing formalism
https://en.wikipedia.org/wiki/Gravitational_lensing_formalism


WikipediaWikipedia
Navarro–Frenk–White profile

Jack

Hi 
@Andrew, thanks for this! I agree with all of this: my concern is that since the critical surface density depends on zs, 
if there are multiple zses, it is ambiguous which one to use.

For now we are using an observed-position likelihood as you point out is the standard with clusters, which James has 
implemented and is working fine. The system we're looking at has two sources at different redshifts, but one is 
clearly visible and the other someone barely noticed an emission line in MUSE data.

For the more obvious source, we may end up doing a pixel reconstruction. (In which case we would probably ignore the 
other source). Andrew Newman was able to do so with a double sersic model in 
this paper: https://ui.adsabs.harvard.edu/abs/2018ApJ...862..125N/abstract


Jam

The multi-plane implementation 
follows (section 2): https://arxiv.org/abs/1403.5278 [NOTE TO READER, I WAS WRONG ABOUT THIS, DIFFERENT CORRECT PAPER LINKED TO BELOW]

I can provide links to the source code, but basically you compute scaling factors (beta in equation 5) based on 
angulars of diameter distances and then apply them when doing the multi-plane tracing the image-plane to each plane one 
after another. The ray-tracing is recursive in that you go from the image-plane to each source-plane one-by-one I believe.


Then the deflection angle for some other source plane can be found easily be re-scaling by the ratio of 'Sigma_crit's 
between the two different source redshifts

I'm going to hazard a guess that equation (5) can be rewritten as a ratio of sigma_crit values (e.g. via equation 4, 
provided D_l1 = D_l2). We basically then just need the code to use the sigma_crit values computed specifically for 
the NFW's (which are related to kappa_s) , when computing the beta values, instead of how the values are computed currently?


Jack

(You can even use the ratios of sigma_crit as a probe of cosmology! https://arxiv.org/abs/2110.06232)

For now, the question is: which zsource is correct to use? Is it the redshift of the first source, or the last one?


Jam

@Qiuhan He When simulating lenses with many DM subhalos (e.g. for the ABC paper) did we account for how to treat the 
source redshift when computing their sigma_crit but also how to set their mass parameters via the critical density 
of the Universe?


Qiuhan He

If we want to compute how the first source is lensed by an NFW halo, then the source redshift is the first source's 
redshift. If we want to compute the lensing quantities about the second source, then the source redshift should be 
the second source's.

The critical density needed for an NFW halo is the critical density of the Universe at the redshift of the halo

The PhysicalNFW defined should involve one more parameter called source_redshift, because the lensing quantity of an NFW halo would change with different background sources at different redshifts

Jam

When simulating lenses for the ABC paper, we had 100s of NFWs at different redshifts making up the line of sight.

I am assuming we always used 'redshift_source' as the source redshift for every NFW, which was what went into 
computing its sigma_crit value.

When performing multi plane ray tracing, we used the redshift of the NFW we were computing deflections of, 
as well as all other NFWs.

Independently, I know these two calculations are valid. What I'm unclear on is whether combining them in the way 
we did is valid (e.g. is the sigma_crit we compute when using collosus defined the same as the ones used for scaling 
between different planes for multi plane ray tracing?l


Qiuhan He

yes

I need to find the multiplane ray tracing equation in Schneider+1992

PDF

Here, Qiuhan linked to section "9.1 The multiple lens-plane theory" of Schneider 1992: 
https://ui.adsabs.harvard.edu/abs/1992grle.book.....S/abstract

Eq. (9.7b) is the multiplane ray-tracing equation. The dimensionless deflection angle of each plane is defined as 
Eq. (9.6), which is the way AutoLens implements.


Jack

It sounds like what you are saying is that for a mass in plane i, AutoLens interprets it's deflection angles as 
between plane i+1  and plane i. So the Sigma_crit is a function of z_i and z_{i+1}. Is that correct? Elsewhere 
James said something implying that it is betwen plane i and i_{max}

This whole approach of sampling unitless quantities is cute and simple with one lens/source plane, but with multiple 
mass/source planes like this, or if you actually care about the physical quantities of the halos, I have always 
thought it would make much more sense to only sample physical parameters of each halo and have your ray-tracing code 
do everything correctly under the hood


Qiuhan He

The sigma_crit is what James described as shown by Eq.(9.4).


Jack

I understand what sigma_crit is.

My point of confusion is the following: sigma_crit is a function of cosmology, lens redshift (z_l), and source 
redshift (z_s). When you have one lens plane and one source plane, parametrizing everything with kappa  is clear and 
unambiguous, but when you have multiple lens and source redshifts, it becomes ambiguous.

Choosing which z_l and z_s to use to define kappa then becomes an implementation choice: you could uses the lowest 
redshift z_s, or the highest redshift z_s . I suppose you could do anything in between but that would be a pretty 
pathological. My question is really what choice does PyAutoLens make here?

AutoLens parameterizes an NFW as kappa_s and theta_scale, the angle of the scale radius. The convergence at the 
scale radius kappa_s is only a meaningful quantity between a specific lens redshift, source redshift, and cosmology. 

So what sigma_crit does kappa_s correspond to? What physical mass surface density \Sigma = \Sigma_{crit} \kappa is causing the lensing?


Qiuhan He

The NFW  profile parameterized by kappa_s  and theta_scale  has no physical meaning. It can be a halo of mass A 
in one strong lensing system or a halo of mass 2A in another strong lensing system whose sigma_crit is twice.

It is only a model for unitless lensing computation.

We have a profile called NFWMCRLudlow . To use that, one needs to specify the redshift of the halo and the source by 
redshift_object  and redshift_source.

In multiplane lensing, we set redshift_object to be the redshift of the i-th lensing plane and redshift_source 
to be the source galaxy redshift


Jack

So I have the opposite problem: We have the physical parameters (Mass, concentration) and need to get the correct 
lensing from that mass.

Ahh so the ambiguity I am struggling with is that we have multiple source galaxies at different redshifts

James sent me a code snippet where it appears that the multi-plane lensing always interprets the deflection angles as 
between the i'th plane and the highest-redshift plane. That would answer my question


Jam

I don't think it is necessary to think about whether a plane has a "source galaxy" (in the sense that you have 
observed imaging data of it and want to model or analyse it).

The multi-plane ray tracing calculations do not care if a plane has a "source galaxy" or if its just another plane 
with a galaxy with mass in. The deflection angles would not change if you added a "source galaxy" to a plane in a 
multi-plane system which previously only had mass components.

Andrew
@Jam, @Qiuhan He have we ever dealt with multiple source planes, or just multiple lens planes? 
@Jack's sentiment that defining a lens by its convergence (and not it's physical mass distribution) is ambiguous when 
there are multiple source planes is something I very much agree with (well, one cannot disagree, it is clearly true!)


Jam

I have used autolens to model multiple source plane systems, but never with physical units throw into the mixer as well


Andrew

But "without physical units" sounds ill defined. If we take a case of a single lens plane (with an NFW) and two 
source planes, what should kappa_s and r_s be? r_s (being an angle to the lens plane) is well defined, but kappa_s 
is different depending on which of the two source planes is being considered


Qiuhan He

Ah. I get it. 

@Jack, for your purpose, ray-tracing for multiple source galaxies, please compute the kappa_s using the last source 
galaxy's redshift


Jack

great, thank you 

Qiuhan He

Autolens will rescale the kappa_s  for source galaxies infront of it. (I think autolens is actually rescaling the 
deflection angles instead) (edited) 


Jam

Ok, I'm gonna write this somewhere permenant.. [Oh look I did lol]



Qiuhan He

Thats how I understand Autolens is doing for multiplane raytracing


Andrew

Sounds like we have an answer :grinning:


Jack

but yes @Andrew that is exactly my concern! in my case we are always concerned with the physical mass of the lenses. 
so for us it never makes sense to sample "lensing units" that can then be rescaled for different redshifts: we always 
want a fixed well-defined cosmology with known redshifts for each source

(I suppose we could vary the cosmology but that doesn't change the principle here)


Andrew

Perhaps best to check what 
@Qiuhan He described with a few simple cases (like do some analysis with a single source plane, then re-do it adding
a second source plane at lower-z with no light and see that nothing changes; and various other variants of this)


Jam

Are you assuming your source redshifts are always known exactly?


Jack

In our case the source redshifts are known to high precision, we don't anticipate working with systems of unknown redshift


Jack

@Andrew James sent me some example code doing something similar to what you described, I'll play with it to 
double-check everything!

and 
@Jam re: source redshift precision: there may be some cluster systems where several sources have known redshift, and 
some do not (or have e.g. photo-zs not spec-zs).

in that case we would like to sample the redshifts of the sources with poorly-constrained redshift. the sources with 
known redshift serve to constrain the lens mass, and if the lens mass is understood well enough it actually can 
constrain the redshift of the other sources. (basically, at what z_source does this source's observed lensing line up)

This is something that's actually done with huge clusters, where it's hard to get speczs on every object. I think 
I've seen this with some of the crazy massive clusters observed with JWST


Andrew

A quick comment (which doesn't actually suggest how things would be best implemented) and is written with explicit 
reference to NFW profiles, but applies to defining lenses by their convergence (not their mass) more generally...

Presumably, the reason why PyAutoLens uses (by default) kappa_s and theta_s [rs in the code, but it sounds like 
it is an angle] (not physical things like M200, c and z_l; or rho_s, r_s and z_l) is that they uniquely specify the 
mapping from image plane to source plane coordinates (i.e. the "deflection angles"). Whereas if specifying the 
3 numbers describing the physical mass distribution (how much it weighs, how concentrated it is, and how far along 
the line-of-sight it is) there are different combinations that give you the same deflection angles (i.e. the same 
mapping from positions in the image back to positions in the source plane). This means that, particularly 
when z_l and/or z_s is unknown, it makes sense to fit for kappa_s and theta_s. Of course, if we know the redshifts, 
then whether we fit for (kappa_s, theta_s), (M200, c), or (rho_s, r_s) doesn't really matter. 

The point I am trying to make, is that there are (in a single lens plane, single source plane case) good reasons to 
work directly with the convergence, rather than thinking first about the physical mass distribution, and then using 
redshifts to turn this into kappa(theta), from which we get alpha(theta) and hence the mapping from image plane to 
source plan positions that we need to "do the lensing".

I think @Jack's multi-source-plane case has caused some confusion because it breaks the property we had with a s
ingle source plane, that we could have got the same lensing effect from different mass distributions at different 
redshifts. As Qiuhan said earlier The NFW  profile parameterized by kappa_s  and theta_scale  has no physical meaning. 
It can be a halo of mass A in one strong lensing system or a halo of mass 2A in another strong lensing system whose 
sigma_crit is twice. But with two source planes, a change in z_l that doubles sigma_crit to one source plane, 
might only increase sigma_crit by a factor of 1.5 to the other source plane. So there is not a sense in which we 
can just fit for some "dimensionless lensing parameters" and then later use the lens redshift to convert this to 
physical lens parameters if we are so inclined. We need to know z_l in order to know how the convergence of the 
lens for source plane 1 relates to the convergence of the lens for source plane 2. And if we know the lens and 
source redshifts, we may as well parameterise the lens by physical parameters, rather than kappa_s and theta_s 
(because the benefit of kappa_s and theta_s has gone).

Coming from a simulation background, I definitely think of physical mass distributions first and then their lensing 
effect second. But I think the "kappa(theta) is primary, and we can worry about what actual mass distribution this 
corresponds to later" approach does make sense in some observational cases. And I'm hoping that hearing those two 
perspectives might help someone (or maybe everyone already knew this! :melting_face:) (edited) 



Jack

Thanks for the reasoned description Andrew! You are of course right that both approaches make sense in 
different cases - I apologize if I was being uncharitable in describing the "lensing units" approach before.

I think this is a frequent division between codes for "galaxy-scale" lensing and "cluster-scale" lensing: with 
galaxy-galaxy lensing often the "lensing units" modeling is simple and works great, but with cluster-scale lensing, 
with may different mass components and multiple source redshifts, the formalism behind the "lensing units" becomes 
confusing just how you describe and it makes more sense to parametrize by physical parameters.

The property that breaks in multiple-source lensing, as you describe, actually has a few interesting science corollaries:
The mass-sheet degeneracy is broken. With galaxy-scale lensing for cosmology, people fret quite a bit about the 
mass sheet degeneracy, but with multiple sources probing different sigma_crit's you have a lever arm to break this 
and constrain the physical mass distribution. This is a reason people don't talk much about the mass-sheet 
degeneracy in cluster-scale lensing

You can actually constrain cosmology with the lensing strength between planes! the ratio of lensing strength between 
planes is expressed as beta (below) for each pair of planes

it is a strict function of geometry (distances) - so comparing predicted to measured values of beta is a method for 
doing cosmography. You can sample over a cosmology for this, and have a similar constraining mechanism to BAO 
measurements. It's basically ratios of angular diameter distances

Here is an example paper doing # 2 with a small sample of cluster-scale lenses. We are currently waiting on 
spectroscopic follow-up of one golden system with many background sources, where we'd like to do this measurement.
https://arxiv.org/abs/2110.06232

arXiv.orgarXiv.org
Galaxy cluster strong lensing cosmography: cosmological constraints from a sample of regular galaxy clusters
Cluster strong lensing cosmography is a promising probe of the background geometry of the Universe and several studies 
have emerged, thanks to the increased quality of observations using space and ground-based telescopes. For the first 
time, we use a sample of five cluster strong lenses to measure the values of cosmological parameters and combine them 
with those from classical probes. In order to assess the degeneracies and the effectiveness of strong-lensing 
cosmography in constraining the background geometry of the Universe, we adopt four cosmological scenarios. We find 
good constraining power on the total matter density of the Universe ($Ω_{\rm m}$) and the equation of state of the dark e… Show more
"""
