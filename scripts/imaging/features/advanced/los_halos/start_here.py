"""
Features: Line-of-Sight Halos
============================

Strong gravitational lenses can reveal small dark matter halos through the distortions they produce in
resolved arcs and Einstein rings, even when those halos contain no visible stars. Some halos orbit the main
lens galaxy; others lie elsewhere between us and the background source. The `los_halos` feature simulates
this second population and its effect on lensed images.

__Scientific Use Cases__

Use these simulations to explore how halo masses, concentrations and redshifts affect the arcs; generate
mock datasets for testing dark matter inference methods; and investigate how line-of-sight structure could
be mistaken for substructure in the main lens galaxy. Ensembles of simulated lenses also let you study how
changes to the assumed halo population alter the distribution of lensing perturbations.

[He et al. (2022)](https://doi.org/10.1093/mnras/stac191) demonstrate the scientific motivation: forward
modelling resolved strong-lensing images can constrain a cutoff in the halo mass function and, through an
assumed dark matter model, the dark matter particle mass. The examples here supply simulations for such
investigations, rather than the paper's complete population-inference workflow. Changing the dark matter
model requires appropriate population assumptions; drawing a different random realization alone does not
change the underlying model.

![Line-of-sight halos and their imprint on lensed arcs](../../../../../images/features/los_halos.png)

Illustration: halos at different distances can perturb the light from a background source. The halo symbols
and arc distortions are schematic, not a computed prediction or directly visible dark matter.

__Why Redshift Matters__

The main lens produces the prominent arcs, while smaller halos add perturbations along the light's path.
Multi-plane ray tracing follows this path through successive redshift planes: a deflection at one plane
changes where the ray passes through subsequent planes.

Redshift therefore matters as well as projected position. A halo between the observer and the lens does not
have the same effect as an otherwise identical halo between the lens and source. Moving every perturber to
the main lens plane generally changes the predicted image. This is the distinction between this feature and
a calculation that places a dark subhalo at the main lens redshift.

__Population Assumptions and Individual Realizations__

The simulator samples halos within a light cone from an assumed cosmological halo mass function, which
specifies their abundance as a function of mass. It assigns concentrations with scatter, represents the
halos with spherical truncated NFW mass profiles, and groups them onto redshift planes.

The supplied example uses a main lens at redshift 0.5, a source at redshift 1.0 and halo masses from
10^7 to 10^10 solar masses. These are example choices, not universal limits of the method. Read the mass
function, concentration relation, light-cone size and redshift construction alongside the code before
interpreting a simulated perturbation.

A realization is one random draw from this population. A striking feature in one image is not a prediction
for every lens, and a weak perturbation in another does not show that the population is absent. Compare
many realizations when studying population-level effects. Keep track of the halo-sampling seed separately
from the image-noise seed so that changes in the mass distribution and changes in noise can be distinguished.

__Accounting for the Mean Matter Density__

Adding a population of halos without accounting for its mean density would add extra focusing on top of
the smooth cosmological background. The simulator includes a compensating negative-convergence sheet on
each halo plane to subtract the population's mean contribution, leaving its spatial fluctuations to perturb
the images.

These sheets are bookkeeping relative to the smooth background, not physical objects with negative mass.
Their normalization belongs to the chosen halo population and light-cone construction; retain that
compensation when comparing realizations or changing the population assumptions.

__From Halos to Mock Imaging__

The examples combine the halo population with a main-lens PowerLaw mass profile, external shear and a
SersicCore source. They trace the source light through the mass distribution, convolve the resulting image
with a point-spread function (PSF), and add noise to produce mock imaging data.

The main lens model, source structure, PSF and noise all affect how apparent a halo perturbation is. Retain
these choices when comparing simulations, or vary them deliberately to test their influence. For scientific
applications, also check that the image sampling, halo mass range, light-cone extent and redshift-plane
resolution are adequate for the perturbations being studied.

This folder generates datasets; it does not fit them or establish a halo detection. Detectability requires
an inference workflow that also accounts for uncertainty in the lens and source models.

__Choose An Example__

- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/los_halos/simulator.ipynb) — Start here to follow halo sampling, multi-plane ray tracing, mean-density compensation and mock-image generation. The outputs include the dataset, saved tracer and halo diagnostics for inspecting a realization.
- [simulator_jax.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/los_halos/simulator_jax.ipynb) — Explore a JAX implementation of the same physical setup, including batched mock-image generation. Halo sampling runs in NumPy; the image-generation path packs halos into fixed-shape arrays for compilation and batching. Changing array shapes or capacities can require recompilation.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the
workspace root.
"""
