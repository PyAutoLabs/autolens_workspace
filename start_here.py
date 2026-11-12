"""
PyAutoLens
===========

**PyAutoLens** models strong gravitational lenses to measure the distribution of matter and reconstruct distant galaxies. A strong lens forms when a foreground galaxy bends light from a more distant galaxy into arcs, multiple images or a ring.

![Schematic showing source light deflected by foreground galaxies towards JWST](https://raw.githubusercontent.com/PyAutoLabs/autolens_assistant/main/docs/images/strong_lens_schematic.png)

Light from the background source galaxy is deflected by the foreground galaxies on its way to the observer.

__Using This Guide__

This notebook gives an overview of the **PyAutoLens Python API**, including how to import `autolens`, create light and mass profiles, and perform lensing calculations. You can read through the examples or run the cells below, either locally or in your web browser using [Google Colab](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/start_here.ipynb).

If you use AI, the [PyAutoLens AI Assistant](https://github.com/PyAutoLabs/autolens_assistant) can give you an interactive walkthrough. Paste this prompt into Claude Code, Codex or another AI coding agent:

> I want to do strong lensing science with PyAutoLens
> (https://github.com/PyAutoLabs/PyAutoLens) and the
> autolens_assistant (https://github.com/PyAutoLabs/autolens_assistant).
>
> First clone the assistant repository, cd into it and follow its AGENTS.md.
> Begin the "start here" guide for a new user.

See [AI Options](https://github.com/PyAutoLabs/PyAutoScientist/blob/main/AI_OPTIONS.md) for tools and setup guidance.

__Google Colab Setup__

Run this cell to install PyAutoLens and download the files needed for this notebook on Google Colab.
If you’re running locally or in HTML, it leaves your installation unchanged.
"""

from autolens import setup_colab
setup_colab.for_autolens(raise_error_if_not_gpu=False)

"""
__Imports__

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
"""
from pathlib import Path
import matplotlib.pyplot as plt

import autolens as al
import autoarray as aa
import autolens.plot as aplt

"""
Lets illustrate a simple gravitational lensing calculation, creating an an image of a lensed galaxy using a 
light profile and mass profile.

__Grid__

The emission of light from a source galaxy, which is gravitationally lensed around the lens galaxy, is described 
using the `Grid2D` data structure, which is two-dimensional Cartesian grids of (y,x) coordinates.

We make and plot a uniform Cartesian grid:
"""
grid = al.Grid2D.uniform(
    shape_native=(150, 150),  # The [pixels x pixels] shape of the grid in 2D.
    pixel_scales=0.05,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
)

aplt.plot_grid(grid=grid, title="")

"""
__Light Profiles__

Our aim is to create an image of the source galaxy after its light has been deflected by the mass of the foreground
lens galaxy. We therefore need to ray-trace the `Grid2D`'s coordinates from the 'image-plane' to the 'source-plane'.

This uses analytic functions representing a galaxy's light and mass distributions, referred to as `LightProfile` and
`MassProfile` objects.

The most common light profile in Astronomy is the elliptical Sersic, which we create an instance of below:
"""
sersic_light_profile = al.lp.Sersic(
    centre=(0.0, 0.0),  # The light profile centre [units of arc-seconds].
    ell_comps=(
        0.2,
        0.1,
    ),  # The light profile elliptical components [can be converted to axis-ratio and position angle].
    intensity=0.005,  # The overall intensity normalisation [units arbitrary and are matched to the data].
    effective_radius=2.0,  # The effective radius containing half the profile's total luminosity [units of arc-seconds].
    sersic_index=4.0,  # Describes the profile's shape [higher value -> more concentrated profile].
)

"""
By passing the light profile the `grid`, we evaluate the light emitted at every (y,x) coordinate and therefore create 
an image of the Sersic light profile.
"""
image = sersic_light_profile.image_2d_from(grid=grid)

plt.imshow(image.native.array)  # Dont worry about the use of .native.array for now.

"""
__Plotting__

In-built plotting methods are provided for plotting objects and their properties, like the image of
a light profile we just created.

By using a `aplt.plot_array` to plot the light profile's image, the figured is improved. 

Its axis units are scaled to arc-seconds, a color-bar is added, its given a descriptive labels, etc.

The plot module is highly customizable and designed to make it straight forward to create clean and informative figures
for fits to large datasets.
"""
aplt.plot_array(array=sersic_light_profile.image_2d_from(grid=grid), title="Image")

"""
__Mass Profiles__

PyAutoLens uses MassProfile objects to represent a galaxy’s mass distribution and perform ray-tracing calculations.

Below we create an elliptical isothermal MassProfile and compute its deflection angles on our Cartesian grid, where 
the deflection angles describe how the lens galaxy’s mass bends the source’s light:
"""
isothermal_mass_profile = al.mp.Isothermal(
    centre=(0.0, 0.0),  # The mass profile centre [units of arc-seconds].
    ell_comps=(
        0.1,
        0.0,
    ),  # The mass profile elliptical components [can be converted to axis-ratio and position angle].
    einstein_radius=1.6,  # The Einstein radius [units of arc-seconds].
)

deflections = isothermal_mass_profile.deflections_yx_2d_from(grid=grid)

"""
The deflection angles are easily plotted using the **PyAutoLens** plot module.

(Many other lensing quantities are also easily plotted, for example the `convergence` and `potential`).
"""
deflections = isothermal_mass_profile.deflections_yx_2d_from(grid=grid)
deflections_y = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)
aplt.plot_array(array=deflections_y, title="Deflections Y")
deflections = isothermal_mass_profile.deflections_yx_2d_from(grid=grid)
deflections_x = aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)
aplt.plot_array(array=deflections_x, title="Deflections X")
aplt.plot_array(
    array=isothermal_mass_profile.convergence_2d_from(grid=grid), title="Convergence"
)
aplt.plot_array(
    array=isothermal_mass_profile.potential_2d_from(grid=grid), title="Potential"
)

"""
__Galaxy__

A `Galaxy` object is a collection of light profiles at a specific redshift.

This object is highly extensible and is what ultimately allows us to fit complex models to strong lens images.

We create two galaxies representing the lens and source galaxies shown in the strong lensing diagram above.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=sersic_light_profile,  # The foreground lens's light is typically observed in a strong lens.
    mass=isothermal_mass_profile,  # Its mass is what causes the strong lensing effect.
)

source_light_profile = al.lp.Exponential(
    centre=(
        0.3,
        0.2,
    ),  # The source galaxy's light is observed, appearing as multiple images around the lens galaxy.
    ell_comps=(
        0.1,
        0.0,
    ),  # However, the mass of the source does not impact the strong lensing effect.
    intensity=0.1,  # and is not included.
    effective_radius=0.5,
)

source_galaxy = al.Galaxy(redshift=1.0, light=source_light_profile)

"""
The `aplt.plot_array` function plots properties of the lens and source galaxies.
"""
aplt.plot_array(array=lens_galaxy.image_2d_from(grid=grid), title="Lens Galaxy Image")
aplt.plot_array(
    array=source_galaxy.image_2d_from(grid=grid), title="Source Galaxy Image"
)

"""
__Tracer__

The `Tracer` object is a collection of galaxies at different redshifts (often referred to as planes). 

It uses these galaxies to perform ray-tracing, using the mass profiles of the galaxies to bend the light of the source
galaxy(s) into the multiple images we observe in a strong lens system. 

This is shown below, where the image of the tracer shows a distinct Einstein ring of the source galaxy.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], cosmology=al.cosmo.Planck15())

image = tracer.image_2d_from(grid=grid)

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__JAX__

**PyAutoLens** runs on either **NumPy** (the default) or **JAX** (Google's
array library with GPU support and just-in-time compilation). JAX makes
lens modeling 10-100x faster on large grids — sometimes more on GPU.

You do not have to do anything to use it. JAX installs with `autolens` by
default (`pip install autolens`), and the analysis objects you'll meet in
the `__Lens Modeling__` section below use JAX automatically. The first time a model-fit starts you'll see a one-time log
line like `JAX: Applying vmap and jit to likelihood function -- may take a
few seconds.` — that's JAX compiling the likelihood function, after which
every evaluation re-uses the compiled code. If JAX is not installed, the
analysis warns once and falls back to NumPy automatically.

That is all a new user needs to know: install the extra, and lens modeling
is fast. Everything else — disabling JAX, writing `@jax.jit` code yourself
and how JAX changes the arrays inside results — is covered in the
`scripts/guides/using_jax.py` guide.

__Units__

The units used throughout the strong lensing literature vary, therefore lets quickly describe the units used in
**PyAutoLens**.

The `Tracer` object and all mass profiles describe their quantities in terms of angles, which are defined in units
of arc-seconds. To convert these to physical units (e.g. kiloparsecs), we use the redshift of the lens and source
galaxies and an input cosmology. A run through of all normal unit conversions is given in guides in the workspace
outlined below.

The use of angles in arc-seconds has an important property, it means that for a two-plane strong lens system 
(e.g. a lens galaxy at one redshift and source galaxy at another redshift) lensing calculations are independent of
the galaxies' redshifts and the input cosmology. This has a number of benefits, for example it makes it straight
forward to compare the lensing properties of different strong lens systems even when the redshifts of the galaxies
are unknown.

Multi-plane lensing is when there are more than two planes. The tracer fully supports this, if you input 3+ galaxies
with different redshifts into the tracer it will use their redshifts and its cosmology to perform multi-plane lensing
calculations that depend on them.

__Extensibility__

All of the objects we've introduced so far are highly extensible, for example a tracer can be made of many galaxies, a 
galaxy can be made up of any number of light profiles and many galaxy objects can be combined into a galaxies object.

Below, wecreate a `Tracer` with 3 galaxies at 3 different redshifts, forming a system with two distinct Einstein
rings! The mass distribution of the first galaxy has separate components for its stellar mass and dark matter, where
the stellar components use a `LightAndMassProfile` via the `lmp` module.
"""
lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05),
        intensity=0.5,
        effective_radius=0.3,
        sersic_index=3.5,
        mass_to_light_ratio=0.6,
    ),
    disk=al.lmp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.1),
        intensity=1.0,
        effective_radius=2.0,
        mass_to_light_ratio=0.2,
    ),
    dark=al.mp.NFWSph(centre=(0.0, 0.0), kappa_s=0.08, scale_radius=30.0),
)

lens_galaxy_1 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Exponential(
        centre=(0.00, 0.00),
        ell_comps=(0.05, 0.05),
        intensity=1.2,
        effective_radius=0.1,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.05, 0.05), einstein_radius=0.6
    ),
)

source_galaxy = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=0.7,
        effective_radius=0.1,
        sersic_index=1.5,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy])

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Lens Modeling__

Lens modeling is the process of fitting a physical model to strong-lensing data in order to infer the properties of
the lens and source galaxies.

The primary goal of **PyAutoLens** is to make lens modeling **simple, scalable to large datasets, and fast**, with
GPU acceleration provided via JAX.

The animation below illustrates the lens modeling workflow. Many models are fitted to the data iteratively,
progressively improving the quality of the fit until the model closely reproduces the observed image.

![Lens Modeling Animation](https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true "model")

**Credit: Amy Etherington**

The next documentation page guides you through lens modeling for a variety of lensing regimes (e.g. galaxy–galaxy lenses,
cluster-scale lenses) and data types (e.g. CCD imaging, interferometer data).

__Simulations__

Simulating strong lenses is often essential, for example to:

- Practice lens modeling before working with real data.
- Generate large training sets (e.g. for machine learning).
- Test lensing theory in a fully controlled environment.

The next documentation page guides you through how to simulate lenses for different types of strong 
lenses (e.g. galaxy–galaxy lenses, cluster-scale lenses) and different types of data (e.g. CCD imaging, interferometer data).

__Wrap Up__

This completes the introduction to **PyAutoLens**, including a brief overview of the core API for lensing calculations,
lens modeling, and data simulation.

__Where To Next__

**PyAutoLens** can analyse strong lens systems across a range of physical scales (e.g. galaxy, group, and cluster) and for 
different types of data (e.g. imaging, interferometer, and point-source observations). Depending on the scientific 
questions you are interested in, the analysis you perform may differ significantly.

The autolens_workspace contains a suite of example Jupyter Notebooks, organised by lens scale and dataset type.

__What Scale Lens?__

What size and scale of strong lens system are you expecting to work with? 

There are three scales to choose from:

- **Galaxy Scale**: Made up of a single lens galaxy lensing a single source galaxy, the simplest strong lens you can get!
  If you're interested in galaxy scale lenses, go to the question below called "What Data Type?".
  
- **Group Scale**: Strong Lens Groups contains 2-10 lens galaxies, normally with one main large galaxy responsible for the majority of lensing.
  They also typically lens just one source galaxy. If you are interested in groups, go to the `group/start_here.ipynb` notebook.
  
- **Cluster Scale**: Strong Lens Galaxy clusters often contained 20-50, or more, lens galaxies, lensing 10, or more, sources galaxies.
  If you are interested in clusters, go to `cluster/start_here.ipynb` notebook.

__What Data Type?__

If you are interested in galaxy-scale strong lenses, you now need to decide what type of strong lens data you are
interested in:

- **CDD Imaging**: For image data from telescopes like Hubble and James Webb, go to `imaging/start_here.ipynb`.

- **Interferometer**: For radio / sub-mm interferometer from instruments like ALMA, go to `interferometer/start_here.ipynb`.

- **Point Sources**: For strongly lensed point sources (e.g. lensed quasars, supernovae), go to `point_source/start_here.ipynb`.

__Google Colab__

You can also open and run each notebook directly in Google Colab, which provides a free cloud computing
environment with all the required dependencies already installed. 

This is a great way to get started quickly without needing to install **PyAutoLens** on your own machine,
so you can check its the right software for you before going through the installation process:

 - [imaging/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/imaging/start_here.ipynb): Galaxy scale strong lenses observed with CCD imaging (e.g. Hubble, James Webb).
 - [interferometer/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/interferometer/start_here.ipynb): Galaxy scale strong lenses observed with interferometer data (e.g. ALMA).
 - [point_source/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/point_source/start_here.ipynb): Galaxy scale strong lenses with a lensed point source (e.g. lensed quasars).
 - [group/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/group/start_here.ipynb): Group scale strong lenses where there are 2-10 lens galaxies.
 - [cluster/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/cluster/start_here.ipynb): Cluster scale strong lenses with 2+ lenses and 5+ source galaxies.

__Still Unsure?__

Each notebook is short and self-contained, and can be completed and adapted quickly to your particular task. 
Therefore, if you're unsure exactly which scale of lensing applies to you, or quite what data you want to use, you 
should just read through a few different notebooks and go from there.

__HowToLens Lectures__

If the start_here examples move too quickly — or you want a slower, more detailed grounding in strong lensing and lens
modeling — **HowToLens** is a separate four-chapter tutorial lecture series written for undergraduates, new PhD
students, or anyone new to the field. It lives in its own repository: https://github.com/PyAutoLabs/HowToLens.

The recommendation is you first look at the start_here example relevant to your science, then go through the lectures
to understand the concepts in more detail.

__Features__

Here is a brief overview of the advanced features of **PyAutoLens**. 

You won't look at these for a while, you should find your relevant start_here notebook and work through that first,
but a quick look through these will give you a sense of the breadth of **PyAutoLens**'s capabilities.

Brief one sentence descriptions of each feature are given, with more detailed descriptions below including 
links to the relevant workspace examples.

**Pixelizations**: Reconstructing the source galaxy on a mesh of pixels, to capture irregular structures like spiral arms.
**Point Sources**: Modeling point sources (e.g. quasars) observed in the strong lens imaging data.
**Interferometry**: Modeling of interferometer data (e.g. ALMA, LOFAR) directly in the uv-plane.
**Multi Gaussian Expansion (MGE)**: Flexible models for foreground lens light and the background source.
**Groups**: Modeling group-scale strong lenses with multiple lens galaxies and multiple source galaxies.
**Multi-Wavelength**: Simultaneous analysis of imaging and / or interferometer datasets observed at different wavelengths.
**Ellipse Fitting**: Fitting ellipses to determine a lens galaxy's ellipticity, position angle and centre.
**Shapelets**: Decomposing a galaxy into a set of shapelet orthogonal basis functions, capturing more complex structures than simple light profiles.
**Operated Light Profiles**: Assuming a light profile has already been convolved with the PSF, for when the PSF is a significant effect.
**Sky Background**: Including the background sky in the model to ensure robust fits to the outskirts of galaxies.

__Pixelizations__

Recover the background galaxy behind the arcs. A pixelization reconstructs source light on a mesh, providing
flexibility to describe spiral arms, star-forming clumps and irregular structure without imposing a simple
analytic light profile.

**Study the background galaxy.** Pixelized reconstructions are a key tool for source science: investigating
galaxy structure and measuring properties such as intrinsic flux, size and lensing magnification. Start with
[source_science.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/source_science.ipynb)
to see how to analyse the reconstructed source, calculate flux and magnification, propagate uncertainties and
export the reconstruction.

![Pixelization: recover the galaxy behind the arcs](images/features/pixelization_illustration.png)

Conceptual illustration, not observational data. The mesh and galaxy on the right show the same source
brightness distribution; removing mesh lines adds no information. Reconstruction requires a lens mass model,
and the recoverable detail depends on the data and modelling assumptions.

Source brightness is solved during the fit, with regularization controlling how freely neighbouring mesh pixels
can vary. This flexibility supports both source science and lens-mass modelling; it does not remove model
degeneracies or guarantee that fine structure is real.

Read the [pixelization feature introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/start_here.ipynb)
for use cases, mesh choices, regularization and worked examples.

For deeper teaching, see [the pixelization overview](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/overview/overview_5_pixelizations.ipynb)
and Chapter 3 of [HowToLens](https://github.com/PyAutoLabs/HowToLens).


__Point Sources__

There are many lenses where the background source is not extended but is instead a point-source, for example strongly
lensed quasars and supernovae.

For these objects, we do not want to model the source using a light profile, which implicitly assumes an extended
surface brightness distribution. 

Instead, we assume that our source is a point source with a centre (y,x), and ray-trace triangles at iteratively
higher resolutions to determine the source's exact locations in the image-plane:

<img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_0.png" width="200">
<img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_1.png" width="200">
<img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_2.png" width="200">
<img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_3.png" width="200">
<img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/point_4.png" width="200">

Note that the image positions above include the fifth central image of the strong lens, which is often not seen in 
strong lens imaging data. It is easy to disable this image in the point source modeling.

Checkout the`autolens_workspace/*/point_source` package to get started.

__Interferometry__

Modeling of interferometer data from submillimeter (e.g. ALMA) and radio (e.g. LOFAR) observatories:

![ALMA Image](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/paper/almacombined.png)

Visibilities data is fitted directly in the uv-plane, circumventing issues that arise when fitting a dirty image
such as correlated noise. This uses the non-uniform fast fourier transform algorithm
[nufftax](https://github.com/GragasLab/nufftax) to efficiently map the galaxy model images to the uv-plane.

Checkout the`autolens_workspace/*/interferometer` package to get started.


__Multi Gaussian Expansion (MGE)__

An MGE models a galaxy's light as a sum of Gaussians. In strong lensing, use it to model foreground lens light
that overlaps the arcs, or to describe the background source whose light is distorted by the lens mass model.

![MGE: separate the foreground lens light and model the background source](images/features/mge_illustration.png)

Illustration, not observational data. Amber represents foreground lens light; cyan represents source light.
The lens mass model connects the source to its arcs. Gaussian components are mathematical building blocks,
not separate galaxies; the components are fitted jointly.

An MGE can capture changes in a galaxy's shape with radius that a single simple light profile misses. Its
Gaussian intensities are solved by linear algebra during fitting, rather than all being sampled separately.
This also provides a flexible smooth source model for measuring source flux and magnification.

[He et al. (2024)](https://arxiv.org/abs/2403.16253) demonstrate MGE lens-light modelling on 38 HST SLACS lenses,
combined with a pixelized source reconstruction. The workspace also includes examples using an MGE source.

MGEs can additionally connect stellar mass models to observed light, describe compact emission alongside its
host galaxy, and support deprojection of galaxy light from two dimensions to three, with appropriate assumptions.

Read the [MGE feature introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/start_here.ipynb)
for the scientific use cases, detailed teaching and worked examples.

__Groups__

The strong lenses we've discussed so far have just a single lens galaxy responsible for the lensing. Group-scale
strong lenses are systems where there two or more  lens galaxies deflecting one or more background sources:

![Group](https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/group.png)

**PyAutoLens** has built in tools for modeling group-scale lenses, with no limit on the number of
lens and source galaxies!

Overviews of group and analysis are given in `notebooks/overview/overview_9_groups.ipynb`
The `autolens_workspace/*/group` package has example scripts for simulating datasets and lens modeling.


__Multi-Wavelength__

Modeling imaging datasets observed at different wavelengths (e.g. HST F814W and F150W) simultaneously or simultaneously
analysing imaging and interferometer data:

![g-band](https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/g_image.png)

![r-band](https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/r_image.png)

The appearance of the strong changes as a function of wavelength, therefore multi-wavelength analysis means we can learn
more about the different components in a galaxy (e.g a redder bulge and bluer disk) or when imaging and interferometer
data are combined, we can compare the emission from stars and dust.

Checkout the `autolens_workspace/*/multi_dataset` package to get started, however combining datasets is a more advanced
feature and it is recommended you first get to grips with the core API.


__Ellipse Fitting__

Ellipse fitting is a technique which fits many ellipses to a galaxy's emission to determine its ellipticity, position
angle and centre, without assuming a parametric form for its light (e.g. like a Seisc profile):

![ellipse](https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/ellipse.png)

This provides complementary information to parametric light profile fitting, for example giving insights on whether
the ellipticity and position angle are constant with radius or if the galaxy's emission is lopsided. 

There are also multipole moment extensions to ellipse fitting, which determine higher order deviations from elliptical 
symmetry providing even more information on the galaxy's structure.

The following paper describes the technique in detail: https://arxiv.org/html/2407.12983v1

Checkout `autolens_workspace/notebooks/features/ellipse_fitting.ipynb` to learn how to use ellipse fitting.


__Shapelets__

Shapelets provide a flexible model of galaxy light beyond a single smooth profile. A key application is
source science: reconstructing the resolved morphology of a background galaxy from its lensed images.
They can also model foreground light and help test how the chosen light representation affects a lens fit.

[Birrer, Amara & Refregier (2015)](https://arxiv.org/abs/1504.07629) demonstrate lens modelling with basis
sets on RX J1131-1231. This illustrates the method rather than a PyAutoLens application.

![Shapelet basis building blocks and an illustrative source model](images/features/shapelets_illustration.png)

*Conceptual illustration, not a fitted reconstruction or the exact sum of the displayed modes.*

Shapelets combine localized mathematical basis functions with linearly solved coefficients. The basis
order, scale and shared geometry control the structures a finite expansion can describe. Signed coefficients
are valid mathematical weights; negative brightness in the combined galaxy model is the physical concern.
Inspect the reconstructed light and residuals together, and test whether added flexibility follows signal
rather than noise.

The existing grid below shows individual polar basis functions. These patterns are mathematical building
blocks, not separate physical components of a galaxy:

![Individual polar shapelet basis functions](images/features/shapelets_basis.png)

Start with the [feature introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/shapelets/start_here.ipynb)
for scientific uses, basis construction, signed coefficients and limitations. The fit.py and modeling.py
examples show fixed-model fits, non-linear fitting, and alternative source and lens-light constructions.

__Operated Light Profiles__

An operated light profile is one where it is assumed to already be convolved with the PSF of the data, with the 
`Moffat` and `Gaussian` profiles common choices:

They are used for certain scientific applications where the PSF convolution is known to be a significant effect and
the knowledge of the PSF allows for detailed modeling abd deblending of the galaxy's light.

Checkout `autogalaxy_workspace/notebooks/features/operated_light_profiles.ipynb` to learn how to use operated profiles.

__Sky Background__

When an image of a galaxy is observed, the background sky contributes light to the image and adds noise:

For detailed studies of the outskirts of galaxies (e.g. stellar halos, faint extended disks), the sky background must be
accounted for in the model to ensure robust and accurate fits.

Checkout `autogalaxy_workspace/notebooks/features/sky_background.ipynb` to learn how to use include the sky
background in your model.


__Mass Models__

The examples above model the lens's mass with an isothermal profile, but **PyAutoLens** supports a wide
range of mass models:

Total mass profiles (e.g. the isothermal and power-law) represent the combined mass of the lens galaxy
with a single profile. Stellar + dark matter models, introduced below, separate these contributions.

Multipole perturbations (m=1, m=3 and m=4) extend any mass model with departures from elliptical symmetry, capturing
lopsidedness, boxiness / disciness and bar-like structures in the mass distribution. Accounting for this angular
complexity is critical for many science cases, as unmodeled angular structure can masquerade as other signals,
for example dark matter substructure.

The following paper measures angular mass complexity alongside dark matter substructure in JWST
strong lens imaging: https://arxiv.org/abs/2410.12987

The role of the m=1 multipole (lopsidedness) is detailed in: https://arxiv.org/abs/2407.12983

Checkout `autolens_workspace/scripts/guides/profiles` for the full range of mass profiles.


__Stellar + Dark Matter__

How much of a lens galaxy's mass is in stars, and how much is dark matter? Decomposed mass models combine
the galaxy's observed light with strong lensing to infer stellar mass-to-light ratios, estimate stellar and
dark matter masses within a chosen aperture, and investigate their relative distributions.

The stellar mass follows a light model with a mass-to-light ratio; a separate halo, often described by an
NFW profile, supplies the dark matter component. Lensing responds to their combined gravity. The inferred
split therefore depends on the adopted light model, mass-to-light prescription, halo model and priors.

[Nightingale et al. (2019)](https://arxiv.org/abs/1901.07801) used PyAutoLens to investigate stellar structures
and inner dark matter halos in massive elliptical galaxies with this approach.

![Observed lens light and inferred stellar plus dark matter components](images/features/stellar_dark_illustration.png)

The image is a conceptual illustration, not a measured decomposition. The observed arcs belong to the
background source; the mass components describe the foreground lens galaxy.

Start with the [feature introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/mass_stellar_dark/start_here.ipynb)
for the science and modelling assumptions. The basic modeling example uses priors near the simulated truth;
chaining.py and slam.py demonstrate staged fits that initialize the stellar mass model from lens-light fits.


__Double Source-Plane Lenses__

A double source-plane lens contains two background galaxies at different redshifts behind a foreground lens.
Their different lensing geometries let us investigate cosmological distance ratios, probe the lens mass
with two sets of arcs, and reconstruct both background galaxies for source science.

The nearer source also acts as an additional deflector for the farther source. A multi-plane model therefore
tracks both its light and its mass, using the galaxies' redshifts and the adopted cosmology to calculate the
appropriate deflections for each source.

[Collett & Auger (2014)](https://arxiv.org/abs/1403.5278) demonstrate the cosmological application using the
Jackpot lens, SDSS J0946+1006. Cosmological inference remains coupled to mass-model assumptions; the basic
modeling.py tutorial here fixes the cosmology and does not itself measure cosmological parameters.

![Two sets of lensed arcs and the three redshift planes that produce them](images/features/double_source_plane_illustration.png)

*Conceptual illustration, not a ray-traced prediction. Colours distinguish the two sources. Double
source-plane lenses can form incomplete arcs or multiple images, rather than two complete rings.*

Start with the [feature introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/start_here.ipynb)
for geometry, scientific uses and inference limitations. The basic modeling example initializes priors near
the simulated truth; chaining.py and slam.py demonstrate staged fits, including the nearer source's role as
an additional lens.


__Automated Pipelines / SLaM__

Fitting complex lens models (e.g. a decomposed mass model with a pixelized source) in one non-linear search is
often infeasible: the parameter space is too complex for the fit to converge reliably or efficiently.

Search chaining breaks the fit into a sequence of simpler searches, where the results of each search initialize
the model of the next. The Source, Light and Mass (SLaM) pipelines are **PyAutoLens**'s pre-built implementation
of this approach: they first build a robust model of the source, then the lens's light, and finally its mass,
gradually increasing model complexity. The SLaM pipelines have been used in many published **PyAutoLens** analyses
and are the recommended way to perform detailed lens modeling of large samples.

Checkout `autolens_workspace/*/guides/modeling/slam_start_here.py` to get started with the SLaM pipelines.


__Dark Matter Subhalos__

Small dark matter halos can perturb the shapes of lensed arcs even when they contain too little starlight to
be seen directly. **PyAutoLens** searches for these otherwise invisible structures by fitting lens models with
and without an additional halo, comparing their Bayesian evidence and inferring the candidate's position and mass
under the adopted halo model.

Use this feature to search for candidate subhalos, investigate arc anomalies, and build detection results for
population studies of dark matter. Pair it with **sensitivity mapping**: simulated halos of different masses and
positions establish what a dataset could detect, making non-detections scientifically informative too.

![An invisible subhalo perturbs a lensed arc](images/features/subhalo_detection_illustration.png)

*Conceptual illustration, not observational data. The distortion is exaggerated for clarity; the dashed marker
represents an inferred halo position, not light emitted by dark matter.*

An evidence increase is conditional on the lens and source models being compared. Candidates require checks of
the main lens mass distribution, source reconstruction and observational systematics. A halo along the line of
sight can also perturb an arc; this example models the perturber at the lens redshift.

For a scientific application and the importance of testing lens-model flexibility, see
[Nightingale et al., Scanning For Dark Matter Subhalos in Hubble Space Telescope Imaging of 54 Strong Lenses](https://arxiv.org/abs/2209.10566).

Start with the [subhalo detection introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/detect/start_here.ipynb)
for the SLaM workflow, grid search and evidence interpretation, then explore
[sensitivity mapping](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/start_here.ipynb).

__Line-of-Sight Halos__

Line-of-sight halos contribute to strong lensing as a population: the small deflections from many halos at
different distances combine to perturb the resolved arcs. The `los_halos` feature simulates this population
through multi-plane ray tracing, including halos both in front of and behind the main lens.

Use these mock images to investigate how the halo population affects lensing, test dark matter inference
methods, and assess the consequences of attributing all perturbations to subhalos at the main lens redshift.
[He et al. (2022)](https://doi.org/10.1093/mnras/stac191) demonstrate how forward modelling resolved arcs
can constrain a cutoff in the halo mass function and, within an assumed dark matter model, the particle mass.

![A population of line-of-sight halos and its combined lensing imprint](images/features/los_halos.png)

*Conceptual illustration, not a computed prediction. The halo symbols represent unseen mass; the arc
distortions illustrate the combined population effect rather than a detection of an individual halo.*

The examples sample a halo population within a light cone, place truncated NFW halos on redshift planes,
and include compensating convergence sheets to account for the population's mean matter density. PSF
convolution and noise turn the lensed source image into mock imaging data. Compare multiple realizations
to distinguish population-level effects from the particular halos drawn in one simulation.

Start with the [line-of-sight halo introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/los_halos/start_here.ipynb)
for population assumptions, multi-plane geometry and mean-density compensation. It links the standard
simulator and a JAX route for batched mock-image generation. This folder supplies simulations, not a
complete halo-detection or population-inference workflow.

__Sensitivity Mapping__

**What could this lens have revealed, even when no subhalo was found?**

Sensitivity mapping measures which dark matter halos a dataset and modeling procedure could detect. It injects
halos into simulated observations, fits each simulation with and without a halo, and compares the Bayesian evidence.
This calibrates both detections and non-detections, enabling population studies of the abundance and masses of
small halos when combined with a halo-population model.

Use it to interpret a non-detection or compare the reach of different observations. Additional simulations with
changed noise levels or angular resolution can help assess observing strategies; the introductory example does
not perform that comparison.

![Illustrative halo recoverability around a lensed arc](images/features/sensitivity_mapping_illustration.png)

*Conceptual illustration, not a computed sensitivity map. The overlay depicts recoverability at one illustrative
injected mass, not detected halos or the four-cell result of the introductory script.*

Start with [sensitivity mapping](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/start_here.ipynb):
a demonstration with an MGE source, four injected positions and one fixed injected halo mass. Each simulated dataset
is fitted twice. The recovery fits allow mass to vary and use narrow position priors around each injected location.
For calibration using reconstructed source morphology, see
[slam_source_pixelized.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/slam_source_pixelized.ipynb).

Sensitivity depends on the simulation, priors, model family and detection threshold. Calibrate it consistently with
the detection analysis; a positive evidence difference alone is not sufficient to claim recovery. A population
analysis needs more than this small demonstration.

For the scientific connection between detection limits and substructure population constraints, see
[Vegetti & Koopmans (2009)](https://arxiv.org/abs/0903.4752).

__Potential Correction__

**What mass structure is your smooth lens model missing?**

Potential correction, also called gravitational imaging, reconstructs small changes to the lensing potential on a
pixelized mesh while reconstructing the background source. Use it to investigate coherent arc residuals, locate
candidate mass perturbations for subsequent physical modeling, and explore departures from a simple smooth lens
model without first choosing a particular halo density profile.

![Lensed arcs and an inferred correction relative to a smooth lens model](images/features/potential_correction_illustration.png)

*Conceptual illustration, not a measured mass map. Positive and negative corrections mean more or less projected
mass relative to the smooth baseline; a negative correction does not mean negative total mass.*

A reconstructed feature is not by itself a confirmed dark-matter detection. Its interpretation depends on the
main lens model, source reconstruction, mask, mesh and regularization, as well as observational systematics.
Potential correction reconstructs structure for the dataset being fitted; sensitivity mapping instead measures
recovery across injected perturbations.

For a scientific application, [Powell et al. (2025)](https://www.nature.com/articles/s41550-025-02651-2) uses
very-high-resolution radio observations of JVAS B1938+666. This illustrates the method's reach, not a mass-detection
limit promised by the introductory imaging example. The PyAutoLens implementation credits the
[Cao et al. potential-correction software](https://github.com/caoxiaoyue/lensing_potential_correction);
retain its [citation and reproduction materials](https://github.com/caoxiaoyue/potential_correction_paper) when using it.

Start with the [potential correction introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/potential_correction/start_here.ipynb)
for simulated imaging with a known smooth starting model, joint reconstruction, convergence corrections,
iterative refinement and evidence sampling. Continue to the
[likelihood walkthrough](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/potential_correction/likelihood_function.ipynb)
or the [interferometer example](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/potential_correction/start_here.ipynb).

__Graphical Models__

The examples above fit each strong lens dataset one-by-one. However, many lens properties are shared across a
sample (e.g. population-level mass distributions, cosmological parameters), and fitting lenses independently does
not exploit this shared structure.

Graphical models fit multiple datasets simultaneously, explicitly defining which parameters are unique to each
lens and which are shared across the sample. For example, the Hubble constant can be inferred jointly from many
time-delay lenses, with each lens having its own mass model but all sharing a single H0. Hierarchical extensions
assume parameters are drawn from a parent distribution (e.g. the population's distribution of mass slopes), whose
properties are inferred from the full sample, extracting significantly more information than one-by-one fitting.

Checkout `autolens_workspace/*/guides/modeling/advanced/graphical.py` to learn how to fit a graphical model
and `autolens_workspace/*/guides/modeling/advanced/hierarchical.py` for hierarchical models.


__Weak Lensing__

In the weak lensing regime, a foreground mass distribution (e.g. a galaxy cluster) subtly distorts the shapes of
many background galaxies, coherently aligning their ellipticities, without producing the multiple images and
giant arcs of strong lensing.

**PyAutoLens** fits weak lensing shear catalogues using the same mass profiles and non-linear search tools as
strong lensing: model-independent mass maps, parametric halo model fits and tangential shear profiles. Its
quickstart example fits the real JWST-era shape catalogue of Abell 2744, and strong and weak lensing data can
be combined in joint analyses of the same cluster.

Checkout `autolens_workspace/*/weak/start_here.py` to fit your first weak lensing dataset.
"""
