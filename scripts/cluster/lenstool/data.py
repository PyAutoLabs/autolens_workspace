"""
Lenstool Users: Data Preparation (SMACS J0723)
==============================================

This script prepares everything needed to repeat a *published Lenstool cluster analysis* in
**PyAutoLens**: the strong-lensing model of the first JWST cluster, SMACS J0723.3-7327, by
Mahler et al. 2023 (ApJ 945, 49; arXiv:2207.07101).

Mahler et al. released their complete Lenstool workflow — the optimized parameter file
(``best.par``), the model definition (``input.par``), the multiple-image constraints
(``arcs.dat``) and the cluster-member catalogue (``galcat.cat``) — at
https://github.com/guillaumemahler/SMACS0723-mahler2022 (ICLv2 release). This script downloads
those files (plus a public RELICS HST image for visualization), translates them into the CSV
formats the PyAutoLens cluster workflow reads, and records every unit / sign convention involved,
so that ``modeling.py`` can (a) reconstruct the published best-fit model exactly and (b) refit it
from scratch.

If you are a Lenstool user: each output CSV corresponds to one input you already maintain —

 - ``arcs.dat``   → ``point_datasets.csv``   (multiple-image positions + redshifts + noise)
 - ``galcat.cat`` → ``members.csv``          (cluster members: centres, shapes, magnitudes)
 - ``best.par``   → ``halos.csv`` + ``members_best.csv``  (every optimized ``potential`` section)

__Attribution__

The model files are © Mahler et al. (GPL-licensed repository) and are downloaded at runtime with
attribution rather than redistributed here. Cite Mahler et al. 2023 (and RELICS, Coe et al. 2019,
for the imaging) in any work that uses them.

__Contents__

- **Paths + URLs:** where the Mahler files and the RELICS image live.
- **Downloads:** cached downloads of the four Lenstool files and the F814W image.
- **Coordinate Convention:** Lenstool's relative-arcsec frame, verified against the data.
- **Parse arcs.dat:** the multiple-image constraints, with per-system redshifts.
- **Parse galcat.cat:** the 144 red-sequence cluster members.
- **Parse best.par:** all 149 optimized dPIE potentials + model-optimized source redshifts.
- **Image Cutout:** a field-sized cutout of the RELICS F814W image in the Lenstool frame.
- **Write CSVs:** the PyAutoLens-side data products.

__Coordinate Convention__

Lenstool works in a relative tangent-plane frame centred on the ``runmode`` ``reference``
coordinate (here RA0, Dec0 = 110.826989, -73.454723):

    x_lt = -(RA - RA0) * cos(Dec0) * 3600      [arcsec, positive toward WEST]
    y_lt = +(Dec - Dec0) * 3600                [arcsec, positive toward North]

This is verified against the data itself: member #1 of ``galcat.cat`` (RA = 110.8355945) maps to
x = -8.820", matching its ``potential`` section in ``best.par`` (x_centre = -8.822). Everything
this script writes uses (y, x) = (y_lt, x_lt), so all PyAutoLens positions, model centres and
critical curves live in Lenstool's own frame and can be compared number-for-number with the
``.par`` file. Only the *sign of x vs RA* differs from the usual sky convention — remember it if
you overlay results on a WCS-aligned image.

__Position Uncertainties__

Lenstool's positional likelihood uses a single ``sigposArcsec`` from ``input.par`` — here
0.44271887" — not the per-image shape columns of ``arcs.dat``. We propagate that value into the
``positions_noise`` column of ``point_datasets.csv`` so the PyAutoLens chi-squared is defined
identically.

__Source Redshifts__

Five systems have spectroscopic redshifts carried by ``arcs.dat`` (systems 1, 2, 5: MUSE grism
values 1.449, 1.3779, 1.425; system 3: MUSE 1.9914; system 7: JWST/NIRSpec 5.17). The other 16
systems were optimized by Lenstool (``z_m_limit``); their best-fit values are read from the
``best.par`` header and treated as *fixed* inputs here — the refit in ``modeling.py`` frees the
mass model but not the source redshifts, which keeps the comparison to the published mass
parameters clean (and the multi-plane solve tractable).
"""

import re
import urllib.request
from pathlib import Path

import numpy as np

"""
__Paths + URLs__
"""
DATASET_PATH = Path("dataset") / "cluster" / "smacs0723"
LENSTOOL_PATH = DATASET_PATH / "lenstool_model"

MAHLER_BASE = (
    "https://raw.githubusercontent.com/guillaumemahler/SMACS0723-mahler2022/main/ICLv2"
)
MAHLER_FILES = ["arcs.dat", "galcat.cat", "input.par", "best.par", "README.txt"]

RELICS_F814W_URL = (
    "https://archive.stsci.edu/hlsps/relics/smacs0723m73/images/60mas-resolution/"
    "hlsp_relics_hst_acs-60mas_smacs0723m73_f814w_v1_drc.fits"
)

REFERENCE_RA = 110.826989
REFERENCE_DEC = -73.454723

MAG0 = 19.12  # potfile reference magnitude (input.par)
SIGPOS_ARCSEC = 0.44271887242357316  # input.par sigposArcsec

CUTOUT_ARCSEC = 220.0  # full width of the image cutout
PIXEL_SCALE = 0.06  # RELICS 60mas imaging

"""
__Downloads__

Each file is downloaded once and cached on disk; delete a file to force a re-download. The 96 MB
F814W mosaic is only fetched if the cutout does not already exist (and can be deleted afterwards —
set ``KEEP_FULL_MOSAIC = False``).
"""
KEEP_FULL_MOSAIC = False

LENSTOOL_PATH.mkdir(parents=True, exist_ok=True)

for name in MAHLER_FILES:
    path = LENSTOOL_PATH / name
    if not path.exists():
        print(f"Downloading {name} from the Mahler et al. repository...")
        urllib.request.urlretrieve(f"{MAHLER_BASE}/{name}", path)

"""
__Coordinate Convention__ (see module docstring)
"""


def lenstool_yx_from(ra: float, dec: float) -> tuple:
    """Convert (RA, Dec) in degrees to Lenstool's relative (y, x) in arcsec."""
    x = -(ra - REFERENCE_RA) * np.cos(np.deg2rad(REFERENCE_DEC)) * 3600.0
    y = (dec - REFERENCE_DEC) * 3600.0
    return (y, x)


"""
__Parse best.par__

``best.par`` is Lenstool's optimized output: a header of comment lines carrying the per-system
optimized redshifts (``#<system> z:<value> dlsds:<...>``), then one ``potential`` section per mass
component. This cluster has 149: five individually-optimized halos (labelled O1-O5: the
cluster-scale halo, the BCG, two light-concentration clumps and one galaxy modelled outside the
scaling relation) and 144 cluster members whose parameters Lenstool derived from the scaling
relation. Every section carries the *same five numbers you would type into PyAutoLens*:

    Lenstool ``potential``            →  ``dPIEMass.from_lenstool`` argument
    ---------------------------------------------------------------------
    x_centre / y_centre  [arcsec]     →  centre=(y, x)
    ellipticity  (a²-b²)/(a²+b²)      →  ellipticity
    angle_pos  [deg]                  →  angle_pos
    core_radius / cut_radius [arcsec] →  r_core / r_cut
    v_disp  [km/s]  (sigma_LT!)       →  sigma
    z_lens                            →  redshift_object
"""


def parse_best_par(path: Path) -> tuple:
    """Return (halo_list, z_by_system) parsed from a Lenstool best.par file."""
    text = path.read_text()

    z_by_system = {}
    for match in re.finditer(r"^#(\d+)\.0 z:([\d.]+)", text, re.M):
        z_by_system[int(match.group(1))] = float(match.group(2))

    halos = []
    for section in re.finditer(r"^potential\s+(\S+)\n(.*?)^\s*end", text, re.M | re.S):
        label, body = section.group(1), section.group(2)
        params = dict(re.findall(r"^\s*(\S+)\s+(-?[\d.]+)", body, re.M))
        halos.append(
            {
                "label": label,
                "profile": int(float(params["profile"])),
                "x": float(params["x_centre"]),
                "y": float(params["y_centre"]),
                "ellipticity": float(params["ellipticity"]),
                "angle_pos": float(params["angle_pos"]),
                "r_core": float(params["core_radius"]),
                "r_cut": float(params["cut_radius"]),
                "sigma": float(params["v_disp"]),
                "z_lens": float(params["z_lens"]),
            }
        )
    return halos, z_by_system


halos, z_by_system = parse_best_par(LENSTOOL_PATH / "best.par")

named_halos = [h for h in halos if h["label"].startswith("O")]
member_halos = [h for h in halos if not h["label"].startswith("O")]

print(
    f"best.par: {len(halos)} potentials "
    f"({len(named_halos)} individually-optimized + {len(member_halos)} scaling members); "
    f"{len(z_by_system)} model-optimized source redshifts."
)
assert all(h["profile"] == 81 for h in halos), "expected dPIE (profile 81) throughout"

"""
__Parse arcs.dat__

One row per multiple image: ``id RA Dec a b theta z mag``. The id encodes ``system.image``. A z of
0.0 means "no spectroscopic redshift" — those systems take their model-optimized value from the
best.par header instead.
"""
image_rows = []
for line in (LENSTOOL_PATH / "arcs.dat").read_text().splitlines():
    parts = line.split()
    if len(parts) < 7 or parts[0].startswith("#"):
        continue
    system = int(parts[0].split(".")[0])
    y, x = lenstool_yx_from(ra=float(parts[1]), dec=float(parts[2]))
    z_spec = float(parts[6])
    redshift = z_spec if z_spec > 0 else z_by_system[system]
    image_rows.append((system, y, x, redshift, z_spec > 0))

systems = sorted({row[0] for row in image_rows})
n_spec = len({r[0] for r in image_rows if r[4]})
print(
    f"arcs.dat: {len(image_rows)} images across {len(systems)} systems "
    f"({n_spec} systems with spectroscopic redshifts)."
)

"""
__Parse galcat.cat__

One row per cluster member: ``id RA Dec a b theta mag [z]``. Commented rows (leading ``#``) were
excluded by Mahler et al. and are skipped here too. The shape columns give the light ellipse:
Lenstool converts (a, b) to its mass ellipticity as e = (a² - b²) / (a² + b²), and we store that
value so the refit can fix each member's geometry exactly as Lenstool did. The luminosity that
enters the scaling relation is *relative* to the potfile pivot mag0 = 19.12:

    L / L0 = 10 ** (0.4 * (mag0 - mag))
"""
member_rows = []
for line in (LENSTOOL_PATH / "galcat.cat").read_text().splitlines():
    parts = line.split()
    if len(parts) < 7 or parts[0].startswith("#"):
        continue
    y, x = lenstool_yx_from(ra=float(parts[1]), dec=float(parts[2]))
    a, b = float(parts[3]), float(parts[4])
    ellipticity = (a**2 - b**2) / (a**2 + b**2)
    angle_pos = float(parts[5])
    mag = float(parts[6])
    luminosity = 10.0 ** (0.4 * (MAG0 - mag))
    member_rows.append((y, x, ellipticity, angle_pos, mag, luminosity))

print(f"galcat.cat: {len(member_rows)} cluster members (relative to mag0 = {MAG0}).")

"""
__Write CSVs__
"""
with open(DATASET_PATH / "point_datasets.csv", "w") as f:
    f.write("name,y,x,positions_noise,redshift\n")
    for system, y, x, redshift, _ in image_rows:
        f.write(f"point_{system},{y:.6f},{x:.6f},{SIGPOS_ARCSEC},{redshift}\n")

with open(DATASET_PATH / "members.csv", "w") as f:
    f.write("y,x,ellipticity,angle_pos,mag,luminosity\n")
    for row in member_rows:
        f.write(",".join(f"{v:.6f}" for v in row) + "\n")

with open(DATASET_PATH / "halos.csv", "w") as f:
    f.write("label,y,x,ellipticity,angle_pos,r_core,r_cut,sigma,z_lens\n")
    for h in named_halos:
        f.write(
            f"{h['label']},{h['y']:.6f},{h['x']:.6f},{h['ellipticity']:.6f},"
            f"{h['angle_pos']:.6f},{h['r_core']:.6f},{h['r_cut']:.6f},"
            f"{h['sigma']:.6f},{h['z_lens']}\n"
        )

with open(DATASET_PATH / "members_best.csv", "w") as f:
    f.write("label,y,x,ellipticity,angle_pos,r_core,r_cut,sigma,z_lens\n")
    for h in member_halos:
        f.write(
            f"{h['label']},{h['y']:.6f},{h['x']:.6f},{h['ellipticity']:.6f},"
            f"{h['angle_pos']:.6f},{h['r_core']:.6f},{h['r_cut']:.6f},"
            f"{h['sigma']:.6f},{h['z_lens']}\n"
        )

print("Wrote point_datasets.csv, members.csv, halos.csv, members_best.csv.")

"""
__Image Cutout__

A 220" cutout of the RELICS F814W mosaic centred on the Lenstool reference coordinate, stored
**flipped in x** so that array coordinates match the Lenstool frame defined above (x positive
toward West). The cutout exists purely for visualization — none of the modeling uses pixel data.
"""
cutout_path = DATASET_PATH / "data.fits"

import os

if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
    print(
        "PYAUTO_SMALL_DATASETS=1: skipping the 96 MB RELICS mosaic download / cutout "
        "(visualization-only product; the modeling data products above are complete)."
    )
elif not cutout_path.exists():
    from astropy.io import fits
    from astropy.wcs import WCS

    mosaic_path = DATASET_PATH / "f814w_mosaic.fits"
    if not mosaic_path.exists():
        print("Downloading the RELICS F814W mosaic (96 MB, one-off)...")
        urllib.request.urlretrieve(RELICS_F814W_URL, mosaic_path)

    with fits.open(mosaic_path) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)

    x_pix, y_pix = wcs.world_to_pixel_values(REFERENCE_RA, REFERENCE_DEC)
    half = int(CUTOUT_ARCSEC / 2.0 / PIXEL_SCALE)
    y0, x0 = int(round(float(y_pix))), int(round(float(x_pix)))
    cutout = data[y0 - half : y0 + half, x0 - half : x0 + half]

    # Flip x so +x points West, matching the Lenstool relative frame.
    cutout = cutout[:, ::-1]

    header = fits.Header()
    header["PIXSCALE"] = PIXEL_SCALE
    header["COMMENT"] = "RELICS F814W cutout in the Lenstool relative frame (+x West)"
    fits.PrimaryHDU(cutout.astype(np.float32), header=header).writeto(
        cutout_path, overwrite=True
    )
    print(
        f'Wrote {cutout_path} ({cutout.shape[0]}x{cutout.shape[1]} @ {PIXEL_SCALE}"/px).'
    )

    if not KEEP_FULL_MOSAIC:
        mosaic_path.unlink()
        print("Deleted the full mosaic (set KEEP_FULL_MOSAIC = True to keep it).")

print("\nData preparation complete.")
