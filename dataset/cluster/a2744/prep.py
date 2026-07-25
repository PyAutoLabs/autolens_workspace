"""
Regenerates the Abell 2744 dataset CSVs from the published Bergamini et al.
2023 (A&A 670, A60) lens-model inputs on VizieR. See README.md for the schema
and selection; run from the workspace root:

    python dataset/cluster/a2744/prep.py
"""

import csv
import re
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
RA0, DEC0 = 3.5875, -30.3972  # projected cluster core, shared with scripts/weak/
COSD = np.cos(np.deg2rad(DEC0))

VIZIER = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A%2bA/670/A60/{table}&-out.max=9999&-out.all"

POSITIONS_NOISE = 0.5  # arcsec — standard image-plane uncertainty in cluster models
MEMBER_RADIUS_MAX = 80.0  # arcsec — scaling-tier selection radius
BCG_RADIUS_MAX = 60.0  # arcsec — main galaxies must be core members


def project(ra, dec):
    return (dec - DEC0) * 3600.0, (ra - RA0) * COSD * 3600.0  # (y, x)


def fetch_tsv(table):
    with urllib.request.urlopen(VIZIER.format(table=table)) as response:
        text = response.read().decode()
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        try:
            int(parts[0])
        except (ValueError, IndexError):
            continue
        rows.append(parts)
    return rows


# --- Multiple images (table A1): gold systems, one knot per family ---

systems = {}
for parts in fetch_tsv("tablea1"):
    image_id = parts[1].strip()
    match = re.match(r"^(.*?)([a-z])$", image_id)
    system_id = match.group(1) if match else image_id
    systems.setdefault(system_id, []).append(
        (image_id, float(parts[3]), float(parts[4]), float(parts[5]), parts[6].strip())
    )

gold = {
    k: v for k, v in systems.items() if len(v) >= 3 and all(i[4] == "3" for i in v)
}

chosen = {}
for k in gold:
    family = k.split(".")[0]
    if family not in chosen or k < chosen[family]:
        chosen[family] = k
selected = [chosen[f] for f in sorted(chosen, key=float)]

with open(OUT / "point_datasets.csv", "w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["name", "y", "x", "positions_noise", "redshift"])
    for i, s in enumerate(selected):
        for image_id, ra, dec, z, qf in sorted(gold[s]):
            y, x = project(ra, dec)
            writer.writerow([f"point_{i}", f"{y:.4f}", f"{x:.4f}", POSITIONS_NOISE, z])

redshift_source_max = max(gold[s][0][3] for s in selected)

# --- Cluster members (table B1): BCGs + scaling tier ---

members = []
for parts in fetch_tsv("tableb1"):
    try:
        mag = float(parts[5])
    except ValueError:
        continue
    y, x = project(float(parts[3]), float(parts[4]))
    members.append((parts[2].strip(), y, x, mag, float(np.hypot(y, x))))

core = sorted([m for m in members if m[4] < BCG_RADIUS_MAX], key=lambda m: m[3])
bcg_1, bcg_2 = core[0], core[1]
mag_ref = bcg_1[3]

with open(OUT / "scaling_galaxies.csv", "w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["y", "x", "luminosity"])
    for member_id, y, x, mag, radius in sorted(members, key=lambda m: m[3]):
        if member_id in (bcg_1[0], bcg_2[0]) or radius > MEMBER_RADIUS_MAX:
            continue
        writer.writerow([f"{y:.4f}", f"{x:.4f}", f"{10.0 ** (-0.4 * (mag - mag_ref)):.5f}"])

# --- Named-galaxy model CSVs ---

with open(OUT / "mass.csv", "w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        ["galaxy", "attr_name", "profile_class", "y", "x", "sigma", "r_core", "r_cut",
         "mass_at_200", "redshift_object", "redshift_source", "H0", "Om0", "redshift"]
    )
    # sigma is Lenstool's fiducial v_disp (km/s); 290 km/s corresponds to a lens
    # strength b0 ~ 3.0" at (z_l = 0.308, z_s = 5.662). sigma and r_cut are
    # initialization placeholders — the modeling script promotes them to priors.
    # r_core is FIXED at 0 (vanishing core, the standard convention for BCGs and
    # members alike; PyAutoLens's dPIE is analytic at r_core = 0).
    # H0 / Om0 are the (fixed) cosmology constants entering the b0 normalization;
    # they must be present so they load as fixed values rather than free priors.
    for i, bcg in enumerate((bcg_1, bcg_2)):
        writer.writerow(
            [f"lens_{i}", "mass", "dPIEMassSph", f"{bcg[1]:.4f}", f"{bcg[2]:.4f}",
             290.0, 0.0, 20.0, "", 0.308, redshift_source_max, 67.66, 0.30966, 0.308]
        )
    writer.writerow(
        ["host_halo", "dark", "NFWMCRLudlowSph", f"{bcg_1[1]:.4f}", f"{bcg_1[2]:.4f}",
         "", "", "", 2e15, 0.308, redshift_source_max, "", "", 0.308]
    )

with open(OUT / "point.csv", "w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["galaxy", "attr_name", "profile_class", "y", "x", "redshift"])
    for i, s in enumerate(selected):
        positions = np.array([project(ra, dec) for _, ra, dec, _, _ in gold[s]])
        writer.writerow(
            [f"source_{i}", f"point_{i}", "Point",
             f"{positions[:, 0].mean():.4f}", f"{positions[:, 1].mean():.4f}",
             gold[s][0][3]]
        )

print(f"selected systems: {selected}")
print(f"BCGs: {bcg_1[0]} at ({bcg_1[1]:.2f}, {bcg_1[2]:.2f}), "
      f"{bcg_2[0]} at ({bcg_2[1]:.2f}, {bcg_2[2]:.2f})")
