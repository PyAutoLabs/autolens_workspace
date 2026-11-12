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

**Experiment with Gemini in Colab.** This notebook includes dedicated instructions from the
[PyAutoLens AI Assistant](https://github.com/PyAutoLabs/autolens_assistant) to help Gemini explain the examples
and suggest changes using the notebook's API. Try the prompts below, run the changed cells and inspect the results.
For your own lensing project, the full assistant provides dedicated lensing workflows and scientific guidance;
use the starting prompt above to work with it.

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

We first plot a coarse grid so you can distinguish its individual coordinates on screen:
"""
grid_for_plot = al.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.25)

fig, ax = plt.subplots()
aplt.plot_grid(grid=grid_for_plot, title="", ax=ax)
ax.set(xlabel="", ylabel="")
plt.show()

"""
For the light and mass calculations below, we use a finer grid covering the same 7.5 arcsecond field of view.

**Try asking Gemini:** "Explain how shape_native and pixel_scales set the field of view. Show me a coarser
display grid that covers the same area, keeping the calculation grid unchanged."
"""
grid = al.Grid2D.uniform(
    shape_native=(150, 150),  # The [pixels x pixels] shape of the grid in 2D.
    pixel_scales=0.05,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
)

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
**Try asking Gemini:** "Help me change the Sersic effective_radius, one value at a time. Which cells should
I rerun, and what change should I look for in the image?"

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
**Try asking Gemini:** "Help me move the source galaxy's centre and rerun the tracer image. Explain how
the arcs change as the source moves away from alignment with the lens."

You can explore these examples with Gemini here. For dedicated help turning them into an analysis of your own
data, continue with the [PyAutoLens AI Assistant](https://github.com/PyAutoLabs/autolens_assistant).

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

This guide helps you choose a **PyAutoLens example notebook** for your observations. Start with the scale of your lens, then choose your data type. Each example includes Python code you can read, run and adapt.

If you use AI, describe your lens and observations to the [PyAutoLens AI Assistant](https://github.com/PyAutoLabs/autolens_assistant). It can help you choose an example and work through it interactively.

__What scale lens?__

Choose the example that matches the galaxies contributing to the lensing. The distinction is the mass model and fitting approach, not a strict galaxy-count threshold.

![Four conceptual lensing scenes: one dominant galaxy, a comparable-mass galaxy pair, a small group and a rich cluster. Gold marks foreground galaxies and blue marks lensed background light. Indicative Einstein radii: galaxy approximately 1 arcsecond; multi-galaxy 1–3 arcseconds; group 3–10 arcseconds; cluster 10–50 arcseconds.](https://raw.githubusercontent.com/PyAutoLabs/autolens_workspace/main/images/new_user_guide/lens_scales.png)

*AI-generated conceptual illustration, not observations or a calculated lens model. Panels are not shown at the same angular scale.*

__Galaxy__

Indicative θ_E_: ~1″

One galaxy dominates the lensing. Start with your data type below.

[Choose your data type below](#what-dataset-type)

__Multi-galaxy__

Indicative θ_E_: ~1–3″

Several galaxies contribute significantly to the lensing, without a dominant shared halo. Model their light and mass together and reconstruct an extended background source.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/multi_galaxy/start_here.ipynb)

__Group__

Indicative θ_E_: ~3–10″

A group of galaxies lenses an extended background source. The example introduces an optional shared dark-matter halo and ways to model fainter members.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/group/start_here.ipynb)

__Cluster__

Indicative θ_E_: ~10–50″

Many member galaxies lens multiple background sources at different redshifts. This example fits their multiple-image positions; it does not fit the lens galaxies’ light.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/cluster/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/cluster/start_here.ipynb)

**Einstein radius (θE)** is a characteristic angular scale of strong lensing, approximately the ring radius for an aligned source and a symmetric lens. These values are rough guides in arcseconds (″), not category boundaries; they depend on the mass distribution and lens/source distances. Complex lenses can have several critical curves.

<details class="sources"><summary>About these scale estimates</summary><p>These are illustrative ranges, not survey limits or predicted radii for the artwork. Multi-galaxy systems overlap galaxy and group scales. The workspace’s lensing guide uses order-one arcsecond galaxy lenses, few-arcsecond multi-galaxy/group lenses and tens-of-arcseconds cluster lenses.</p><p>Examples grounding the scale: <a href="https://www.aanda.org/articles/aa/pdf/2025/05/aa51868-24.pdf">Euclid galaxy-scale lenses</a>; <a href="https://doi.org/10.1051/0004-6361/200811473">SL2S groups with 3–8″ Einstein radii</a>; <a href="https://doi.org/10.1051/0004-6361/202349060">a cluster lens with an effective Einstein radius around 43″</a>. For a complex lens, an effective Einstein radius is often defined from the area enclosed by its tangential critical curve for a specified source redshift.</p></details>

<a id="what-dataset-type"></a>

__What dataset type?__

For galaxy-scale lenses, choose the example for your observations. For multi-galaxy, group or cluster systems, begin with the scale-specific guide above.

![Four conceptual dataset representations: a pixelated lensed image, samples in the interferometer u–v plane, stacked spectral-channel images, and four compact quasar images around a lens galaxy.](https://raw.githubusercontent.com/PyAutoLabs/autolens_workspace/main/images/new_user_guide/dataset_types.png)

*AI-generated schematic, not measured data. The interferometer panel shows where visibilities are sampled, not their amplitudes or phases; cube slices represent different spectral channels.*

__Imaging__

Images from telescopes such as Hubble or JWST. Fit the lens light, mass and extended source.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/imaging/start_here.ipynb)

__Interferometer__

<img src="images/new_user_guide/interferometry.png" alt="Interferometer conceptual illustration" width="360">

Radio or submillimetre observations, such as ALMA data. Fit the measured visibilities directly.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/interferometer/start_here.ipynb)

__Data cubes__

<img src="images/new_user_guide/data_cubes.png" alt="Data cubes conceptual illustration" width="360">

Spectral-line channels, such as an ALMA CO cube. Fit the channels together with a shared lens model.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/datacube/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/interferometer/features/datacube/start_here.ipynb)

__Point sources__

<img src="images/new_user_guide/point_sources.png" alt="Point sources conceptual illustration" width="360">

Multiple images of a compact source, such as a quasar or supernova. Start with the point-source example.

[Notebook ](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.12.1/notebooks/point_source/start_here.ipynb)

Read each notebook on GitHub, open it locally, or use its Google Colab link to run it in your browser. In Colab, run the setup cell first to install the software and prepare the example.

__Still unsure?__

Tell the [AI Assistant](https://github.com/PyAutoLabs/autolens_assistant) what objects are in your field and what observations you have. It can help you identify a starting example.

__Learn the science__

[HowToLens](https://github.com/PyAutoLabs/HowToLens) teaches gravitational lensing and Bayesian modelling through worked notebooks. Use it for a slower explanation of the concepts behind these examples.

__Next steps__

Once you have chosen an example, explore the [Features guide](https://pyautolens.readthedocs.io/en/latest/overview/overview_3_features.html) to find the modelling tools relevant to your science.

__Features__

Explore the modelling tools you can add to your analysis. Each introduction explains the scientific use cases and links to worked examples.

For your first dataset and starting notebook, use the [New User Guide](https://autolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html).

<table class="feature-gallery"><tbody><tr><td width="25%"><a href="#pixelizations"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/pixelization_illustration.png" alt="Pixelizations illustration" width="220"><br>Pixelizations</a></td><td width="25%"><a href="#multi-gaussian-expansion-mge"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/mge_illustration.png" alt="Multi Gaussian Expansion (MGE) illustration" width="220"><br>MGE</a></td><td width="25%"><a href="#dark-matter-subhalos"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/subhalo_detection_illustration.png" alt="Dark Matter Subhalos illustration" width="220"><br>Dark Matter Subhalos</a></td><td width="25%"><a href="#sensitivity-mapping"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/sensitivity_mapping_illustration.png" alt="Sensitivity Mapping illustration" width="220"><br>Sensitivity Mapping</a></td></tr></tbody></table>

*Conceptual illustrations, not measured data or guaranteed reconstructions.*

<details class="feature-more-images"><summary>More feature illustrations</summary><table class="feature-gallery"><tbody><tr><td width="25%"><a href="#shapelets"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/shapelets_illustration.png" alt="Shapelets illustration" width="220"><br>Shapelets</a></td><td width="25%"><a href="#point-sources"><img src="images/new_user_guide/point_sources.png" alt="Point Sources illustration" width="220"><br>Point Sources</a></td><td width="25%"><a href="#interferometry"><img src="images/new_user_guide/interferometry.png" alt="Interferometry illustration" width="220"><br>Interferometry</a></td><td width="25%"><a href="#data-cubes"><img src="images/new_user_guide/data_cubes.png" alt="Data Cubes illustration" width="220"><br>Data Cubes</a></td></tr><tr><td width="25%"><a href="#multi-wavelength"><img src="images/new_user_guide/multi_wavelength.png" alt="Multi-Wavelength illustration" width="220"><br>Multi-Wavelength</a></td><td width="25%"><a href="#linear-light-profiles"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/linear_light_profiles.png" alt="Linear Light Profiles illustration" width="220"><br>Linear Light Profiles</a></td><td width="25%"><a href="#extra-galaxies"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/extra_galaxies.png" alt="Extra Galaxies illustration" width="220"><br>Extra Galaxies</a></td><td width="25%"><a href="#no-lens-light"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/no_lens_light.png" alt="No Lens Light illustration" width="220"><br>No Lens Light</a></td></tr><tr><td width="25%"><a href="#operated-light-profiles"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/operated_light_profile.png" alt="Operated Light Profiles illustration" width="220"><br>Operated Light Profiles</a></td><td width="25%"><a href="#sky-background"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/sky_background.png" alt="Sky Background illustration" width="220"><br>Sky Background</a></td><td width="25%"><a href="#stellar-dark-matter"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/stellar_dark_illustration.png" alt="Stellar + Dark Matter illustration" width="220"><br>Stellar + Dark Matter</a></td><td width="25%"><a href="#scaling-relations"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/scaling_relation.png" alt="Scaling Relations illustration" width="220"><br>Scaling Relations</a></td></tr><tr><td width="25%"><a href="#multi-galaxy-lenses-groups-and-clusters"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/group_halo.png" alt="Multi-Galaxy Lenses, Groups and Clusters illustration" width="220"><br>Multi-Galaxy Lenses, Groups and Clusters</a></td><td width="25%"><a href="#double-source-plane-lenses"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/double_source_plane_illustration.png" alt="Double Source-Plane Lenses illustration" width="220"><br>Double Source-Plane Lenses</a></td><td width="25%"><a href="#weak-lensing"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/strong_weak_lensing.png" alt="Weak Lensing illustration" width="220"><br>Weak Lensing</a></td><td width="25%"><a href="#line-of-sight-halos"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/los_halos.png" alt="Line-of-Sight Halos illustration" width="220"><br>Line-of-Sight Halos</a></td></tr><tr><td width="25%"><a href="#potential-correction"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/potential_correction_illustration.png" alt="Potential Correction illustration" width="220"><br>Potential Correction</a></td></tr></tbody></table></details>

__Reconstruct the source__

<div id="pixelizations" class="feature-entry"><h3>Pixelizations</h3><p>Reconstruct the background galaxy on a mesh to study spiral arms, star-forming clumps and irregular structure. Pixelization is a key tool for <strong>source science</strong>: measuring intrinsic flux, size and magnification alongside the lens mass.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/start_here.ipynb">Start with pixelization</a>.</p>
</div>

<div id="multi-gaussian-expansion-mge" class="feature-entry"><h3>Multi Gaussian Expansion (MGE)</h3><p>Model foreground lens light that overlaps the arcs, or reconstruct a flexible smooth source, using a sum of Gaussians. An MGE captures changes in shape with radius while solving component brightnesses together. Source examples include flux and magnification measurements.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/start_here.ipynb">Explore multi gaussian expansion (mge)</a>.</p>
</div>

<div id="shapelets" class="feature-entry"><h3>Shapelets</h3><p>Study source morphology or flexible foreground light with a sum of spatial basis functions. Shapelets capture structure beyond a simple smooth profile while retaining a compact description of the basis geometry. Inspect the combined light and residuals as you increase flexibility.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/shapelets/start_here.ipynb">Explore shapelets</a>.</p>
</div>

__Model foreground light and neighbours__

<div id="linear-light-profiles" class="feature-entry"><h3>Linear Light Profiles</h3><p>Fit models with many light components using fewer sampled parameters. Component brightnesses are solved together at each likelihood evaluation, leaving the search to explore geometry and lens mass. Useful for lens light, source light and MGE models; solved brightnesses change with the model.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/start_here.ipynb">Explore linear light profiles</a>.</p>
</div>

<div id="extra-galaxies" class="feature-entry"><h3>Extra Galaxies</h3><p>Account for neighbours whose light blends with the arcs or whose gravity changes the lensing. Choose a light model, mass model or both for each companion. Masking its emission does not remove its gravitational effect.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/extra_galaxies/start_here.ipynb">Explore extra galaxies</a>.</p>
</div>

<div id="no-lens-light" class="feature-entry"><h3>No Lens Light</h3><p>Fit the lens mass and source when foreground light is negligible or has already been subtracted. The lens still deflects light. For subtracted images, check how residuals and subtraction uncertainty affect the result.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/no_lens_light/start_here.ipynb">Explore no lens light</a>.</p>
</div>

<div id="operated-light-profiles" class="feature-entry"><h3>Operated Light Profiles</h3><p>Separate compact emission from extended galaxy light using a component that already includes the imaging PSF response. The surrounding galaxy still receives its appropriate instrumental response. Interferometer examples instead transform the compact sky emission into visibilities.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/operated_light_profile/start_here.ipynb">Explore operated light profiles</a>.</p>
</div>

<div id="sky-background" class="feature-entry"><h3>Sky Background</h3><p>Fit an uncertain sky level alongside faint lens light and arcs. This helps distinguish galaxy outskirts from residual background, while keeping sky photon noise in the noise model. The example fits a constant level.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/sky_background/start_here.ipynb">Explore sky background</a>.</p>
</div>

**Advanced: ellipse fitting.** Measure foreground-light shape with the [ellipse introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/ellipse/start_here.ipynb), then follow the maintained PyAutoGalaxy examples.

__Measure lens mass__

<div id="mass-models" class="feature-entry"><h3>Mass Models</h3><p>Test how the assumed mass distribution affects the inferred lens and source. Choose total-mass profiles or separate stellar and dark components, and use multipoles to explore departures from elliptical symmetry. Angular structure can otherwise resemble a substructure signal.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/profiles">Explore mass models</a>.</p>
</div>

<div id="stellar-dark-matter" class="feature-entry"><h3>Stellar + Dark Matter</h3><p>Measure how stars and dark matter contribute to a lens galaxy’s mass. Combine a stellar component tied to observed light with a separate halo to infer mass-to-light ratios and aperture masses. Lensing measures their combined gravity, so the split depends on the model and priors.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/mass_stellar_dark/start_here.ipynb">Explore stellar + dark matter</a>.</p>
</div>

<div id="scaling-relations" class="feature-entry"><h3>Scaling Relations</h3><p>Include a population of companion galaxies without fitting every mass independently. Tie mass normalizations to measured luminosities through a shared relation, while giving influential companions more freedom. The tied masses depend on that population assumption.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/scaling_relation/start_here.ipynb">Explore scaling relations</a>.</p>
</div>

<div id="multi-galaxy-lenses-groups-and-clusters" class="feature-entry"><h3>Multi-Galaxy Lenses, Groups and Clusters</h3><p>Model systems with several deflectors: <strong>multi-galaxy lenses</strong> have co-dominant galaxies; <strong>groups</strong> can include a shared halo; <strong>clusters</strong> use multiple-image positions from many sources. Choose the scale and observables that match your data.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/start_here.ipynb">Explore multi-galaxy lenses, groups and clusters</a>.</p>
</div>

<div id="double-source-plane-lenses" class="feature-entry"><h3>Double Source-Plane Lenses</h3><p>Use two background galaxies at different redshifts to probe lens mass and reconstruct both sources. Their geometry can also constrain cosmological distance ratios. Multi-plane modelling includes the nearer source’s deflection of the farther one; the introductory fit keeps cosmology fixed.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/start_here.ipynb">Explore double source-plane lenses</a>.</p>
</div>

<div id="weak-lensing" class="feature-entry"><h3>Weak Lensing</h3><p>Constrain mass across a wider field using the coherent shape distortions of background galaxies. Fit shear catalogues alone or combine them with central strong-lensing constraints, preserving each dataset’s uncertainties and source-redshift information.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/start_here.ipynb">Explore weak lensing</a>.</p>
</div>

__Probe dark matter and build analyses__

<div id="dark-matter-subhalos" class="feature-entry"><h3>Dark Matter Subhalos</h3><p>Search for unseen halos through their perturbations to lensed arcs. Compare fits with and without a halo to identify candidates and infer their properties. Check candidates against lens-model flexibility and systematics, then use sensitivity mapping to interpret both detections and non-detections.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/start_here.ipynb">Explore dark matter subhalos</a>.</p>
<p>Continue to <a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/detect/start_here.ipynb">subhalo detection</a> for the search workflow.</p>
</div>

<div id="line-of-sight-halos" class="feature-entry"><h3>Line-of-Sight Halos</h3><p>Simulate the combined effect of a halo population between observer and source. Multi-plane ray tracing includes halos in front of and behind the main lens, providing mock images for dark matter studies and tests of lens-redshift assumptions. These examples generate simulations, not population-inference results.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/los_halos/start_here.ipynb">Explore line-of-sight halos</a>.</p>
</div>

<div id="sensitivity-mapping" class="feature-entry"><h3>Sensitivity Mapping</h3><p>Measure which halos your data and modelling procedure could detect. Inject a halo into simulated data, fit with and without it, and repeat across positions or masses. This calibrates non-detections as well as detections; the introductory example uses four positions at one injected mass.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/start_here.ipynb">Explore sensitivity mapping</a>.</p>
</div>

<div id="potential-correction" class="feature-entry"><h3>Potential Correction</h3><p>Investigate arc residuals by reconstructing small corrections to a smooth lensing potential alongside the source. This gravitational-imaging approach locates candidate mass structure without first choosing a halo profile. A reconstructed feature still needs physical modelling and systematic checks.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/potential_correction/start_here.ipynb">Explore potential correction</a>.</p>
</div>

<div id="automated-pipelines-slam" class="feature-entry"><h3>Automated Pipelines / SLaM</h3><p>Build complex lens models through a sequence of simpler fits. The Source, Light and Mass pipelines use earlier results to initialize later searches, progressively refining the source, foreground light and mass model.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/modeling/slam_start_here.ipynb">Explore automated pipelines / slam</a>.</p>
</div>

<div id="graphical-models" class="feature-entry"><h3>Graphical Models</h3><p>Study populations of lenses by fitting datasets with shared parameters. Hierarchical models infer a population distribution while retaining an individual model for each lens—for example, a distribution of mass slopes or a common Hubble constant.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/modeling/advanced/graphical.ipynb">Explore graphical models</a>.</p>
</div>

__Extend the analysis across observations__

<div id="point-sources" class="feature-entry"><h3>Point Sources</h3><p>Add flux and time-delay constraints to a point-source lens model, or deblend compact images from foreground light before measuring them.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/start_here.ipynb">Point-source examples</a>.</p>
</div>

<div id="interferometry" class="feature-entry"><h3>Interferometry</h3><p>Extend radio and submillimetre modelling with flexible source light, pixelized reconstructions and source-science measurements while fitting the visibilities.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/tree/main/notebooks/interferometer/features">Interferometer features</a>.</p>
</div>

<div id="data-cubes" class="feature-entry"><h3>Data Cubes</h3><p>Map a lensed source across a spectral line by reconstructing each channel with a shared lens model. Each channel keeps its own brightness, sampling and noise. These examples reconstruct emission; a dynamical interpretation needs an additional model.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/datacube/start_here.ipynb">Explore data cubes</a>.</p>
</div>

<div id="multi-wavelength" class="feature-entry"><h3>Multi-Wavelength</h3><p>Compare stellar, dust and line emission across observations while fitting a shared lens model. Keep each dataset’s brightness distribution, instrumental response and noise distinct; link only the parameters your scientific model shares.</p>
<p><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/start_here.ipynb">Explore multi-wavelength</a>.</p>
</div>

<section class="directory" id="directory"><h2>All workspace links</h2><p class="minor">Choose your dataset to browse every feature introduction.</p><details><summary>Group · 13 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/double_source_plane_lens/start_here.ipynb">Double Source Plane Lens</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/mass_stellar_dark/start_here.ipynb">Mass Stellar Dark</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/operated_light_profile/start_here.ipynb">Operated Light Profile</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/shapelets/start_here.ipynb">Shapelets</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/sky_background/start_here.ipynb">Sky Background</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/subhalo/detect/start_here.ipynb">Subhalo / Detect</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/subhalo/start_here.ipynb">Subhalo</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/group_halo/start_here.ipynb">Group Halo</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/linear_light_profiles/start_here.ipynb">Linear Light Profiles</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/multi_gaussian_expansion/start_here.ipynb">Multi Gaussian Expansion</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/no_lens_light/start_here.ipynb">No Lens Light</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/pixelization/start_here.ipynb">Pixelization</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/scaling_relation/start_here.ipynb">Scaling Relation</a></li></ul></details><details><summary>Imaging · 17 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/start_here.ipynb">Double Source Plane Lens</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/ellipse/start_here.ipynb">Ellipse</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/los_halos/start_here.ipynb">Los Halos</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/mass_stellar_dark/start_here.ipynb">Mass Stellar Dark</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/operated_light_profile/start_here.ipynb">Operated Light Profile</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/potential_correction/start_here.ipynb">Potential Correction</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/shapelets/start_here.ipynb">Shapelets</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/sky_background/start_here.ipynb">Sky Background</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/detect/start_here.ipynb">Subhalo / Detect</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/sensitivity/start_here.ipynb">Subhalo / Sensitivity</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/subhalo/start_here.ipynb">Subhalo</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/extra_galaxies/start_here.ipynb">Extra Galaxies</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/start_here.ipynb">Linear Light Profiles</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/start_here.ipynb">Multi Gaussian Expansion</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/no_lens_light/start_here.ipynb">No Lens Light</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/start_here.ipynb">Pixelization</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/scaling_relation/start_here.ipynb">Scaling Relation</a></li></ul></details><details><summary>Interferometer · 12 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/operated_light_profile/start_here.ipynb">Operated Light Profile</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/potential_correction/start_here.ipynb">Potential Correction</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/shapelets/start_here.ipynb">Shapelets</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/subhalo/detect/start_here.ipynb">Subhalo / Detect</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/subhalo/sensitivity/start_here.ipynb">Subhalo / Sensitivity</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/subhalo/start_here.ipynb">Subhalo</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/datacube/start_here.ipynb">Datacube</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/start_here.ipynb">Extra Galaxies</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/start_here.ipynb">Linear Light Profiles</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/start_here.ipynb">Multi Gaussian Expansion</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/pixelization/start_here.ipynb">Pixelization</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/scaling_relation/start_here.ipynb">Scaling Relation</a></li></ul></details><details><summary>Multi dataset · 8 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/start_here.ipynb">Dataset Offsets</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/start_here.ipynb">Imaging And Interferometer</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_point_source/start_here.ipynb">Imaging And Point Source</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/one_by_one/start_here.ipynb">One By One</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/pixelization/start_here.ipynb">Pixelization</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/start_here.ipynb">Same Wavelength</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/slam/start_here.ipynb">Slam</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/start_here.ipynb">Wavelength Dependence</a></li></ul></details><details><summary>Multi galaxy · 13 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/double_source_plane_lens/start_here.ipynb">Double Source Plane Lens</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/mass_stellar_dark/start_here.ipynb">Mass Stellar Dark</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/operated_light_profile/start_here.ipynb">Operated Light Profile</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/shapelets/start_here.ipynb">Shapelets</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/sky_background/start_here.ipynb">Sky Background</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/subhalo/detect/start_here.ipynb">Subhalo / Detect</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/subhalo/start_here.ipynb">Subhalo</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/extra_galaxies/start_here.ipynb">Extra Galaxies</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/linear_light_profiles/start_here.ipynb">Linear Light Profiles</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/multi_gaussian_expansion/start_here.ipynb">Multi Gaussian Expansion</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/no_lens_light/start_here.ipynb">No Lens Light</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/pixelization/start_here.ipynb">Pixelization</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/scaling_relation/start_here.ipynb">Scaling Relation</a></li></ul></details><details><summary>Point source · 4 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/deblending/start_here.ipynb">Deblending</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/extra_galaxies/start_here.ipynb">Extra Galaxies</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/multiple_sources/start_here.ipynb">Multiple Sources</a></li><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/scaling_relation/start_here.ipynb">Scaling Relation</a></li></ul></details><details><summary>Weak · 1 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/weak/features/strong_lensing/start_here.ipynb">Strong Lensing</a></li></ul></details></section>

"""
