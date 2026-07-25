"""
Simulator: Group Halo
=====================

This script simulates the `Imaging` dataset fitted by the `group_halo` feature tutorial: a group-scale lens
with a genuine **group-scale dark matter halo**, a brightest group galaxy (BGG) and two tidally truncated
member galaxies, lensing a single extended background source.

The defining question of the group regime — and of the `modeling.py` tutorial this simulator feeds — is
whether the data *requires* the halo: some groups genuinely do, others are adequately described by their
member galaxies alone. This simulator's truth INCLUDES the halo, so the tutorial's halo model should win the
model comparison. To build intuition for the opposite case, set ``include_group_halo = False`` below,
re-simulate, and watch the members-only model win instead.

__The mass components (and the truncation convention)__

 - **Group-scale dark matter halo** (~10^13-10^14 M_sun): a dPIE with a large core and its truncation fixed
   large — the Lenstool-literature convention for host halos (an NFW / gNFW is the physically preferred
   alternative; see `cluster/mass_parameterizations.py`).
 - **BGG + 2 member galaxies**: dPIE profiles with **vanishing cores (r_core = 0)** and finite truncation
   radii ``r_cut``. Truncation is the signature of the group and cluster regimes: it encodes tidal stripping
   of the members' outer halos by the shared host potential. (At galaxy and multi-galaxy scale, with no host
   halo, profiles are untruncated — see `multi_galaxy/`.)

Lens light is deliberately omitted so the `modeling.py` tutorial isolates the mass question; the standard
group light handling (MGE per galaxy) is covered by `group/start_here.py` and `group/modeling.py`.

__Contents__

- **Dataset Paths:** Where the dataset is written.
- **Simulation Switch:** The `include_group_halo` flag that flips the tutorial's verdict.
- **Grid / PSF / Simulator:** Standard imaging simulation setup.
- **Group Halo:** The group-scale dPIE dark-matter halo.
- **BGG + Members:** Truncated members placed exactly on the tied scaling relation.
- **Source Galaxy / Ray Tracing / Dataset:** Simulate and write the dataset.
- **Tracer json + Centres:** Truth records for the modeling script.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The dataset is output to `/autolens_workspace/dataset/group/group_halo`.
"""
dataset_type = "group"
dataset_name = "group_halo"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Simulation Switch__

The truth simulated below includes the group halo. Flip this to `False` (and re-run this script) to produce
a members-only truth, and watch the `modeling.py` model comparison flip its verdict.
"""
include_group_halo = True

"""
__Grid / PSF / Simulator__
"""
grid = al.Grid2D.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
)

member_centres = [(0.0, 0.0), (3.6, 2.4), (-3.0, 3.4)]  # BGG first

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=member_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

psf = al.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Redshifts / Cosmology__
"""
redshift_lens = 0.5
redshift_source = 1.0

H0 = 67.66
Om0 = 0.30966

"""
__Group Halo__

The group-scale dark matter halo: an elliptical dPIE centred near (but deliberately not exactly on) the BGG,
with a large core and truncation fixed large — the Lenstool-style host-halo convention. Its dispersion of
500 km/s corresponds to a group-scale (~10^13-10^14 M_sun) halo.
"""
halo = al.Galaxy(
    redshift=redshift_lens,
    mass=al.mp.dPIEMass(
        centre=(0.3, -0.2),
        ellipticity=0.3,
        angle_pos=60.0,
        sigma=500.0,
        r_core=8.0,
        r_cut=200.0,
        redshift_object=redshift_lens,
        redshift_source=redshift_source,
        H0=H0,
        Om0=Om0,
    ),
)

"""
__BGG + Members__

The BGG and two members: dPIE profiles with vanishing cores (r_core = 0; the dPIE is analytic there) and
finite truncations — tidally stripped subhalos inside the group potential.

The member truths are placed EXACTLY on the modern tied scaling relation the `modeling.py` tutorial
fits — sigma_i = sigma_BGG * (L_i/L_BGG)^alpha and r_cut_i = r_cut_BGG * (L_i/L_BGG)^beta_cut with
alpha = 0.25 and beta_cut = 1 + gamma - 2*alpha = 0.7 (gamma = 0.2) — so the tied model can recover the
truth and the halo-vs-no-halo comparison is not contaminated by member mis-specification. The luminosity
ratios below are the same ones `modeling.py` uses.
"""
member_luminosity_ratios = [1.0, 0.25, 0.15]  # relative to the BGG

alpha = 0.25
gamma = 0.2
beta_cut = 1.0 + gamma - 2.0 * alpha  # 0.7

bgg_sigma = 280.0  # km/s
bgg_r_cut = 8.0  # arcsec

members = []
for centre, luminosity_ratio in zip(member_centres, member_luminosity_ratios):
    members.append(
        al.Galaxy(
            redshift=redshift_lens,
            mass=al.mp.dPIEMassSph(
                centre=centre,
                sigma=bgg_sigma * luminosity_ratio**alpha,
                r_core=0.0,
                r_cut=bgg_r_cut * luminosity_ratio**beta_cut,
                redshift_object=redshift_lens,
                redshift_source=redshift_source,
                H0=H0,
                Om0=Om0,
            ),
        )
    )

"""
__Source Galaxy__
"""
source_galaxy = al.Galaxy(
    redshift=redshift_source,
    bulge=al.lp.SersicCore(
        centre=(0.2, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=2.0,
        effective_radius=0.3,
        sersic_index=1.0,
    ),
)

"""
__Ray Tracing + Dataset__
"""
lens_galaxies = members + ([halo] if include_group_halo else [])

tracer = al.Tracer(galaxies=lens_galaxies + [source_galaxy])

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Tracer json + Centres__
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

al.output_to_json(
    obj=al.Grid2DIrregular(member_centres),
    file_path=Path(dataset_path, "member_centres.json"),
)

"""
Finished.
"""
