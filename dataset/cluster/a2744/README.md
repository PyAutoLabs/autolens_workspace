# Abell 2744 cluster strong-lensing dataset

Real strong-lensing constraints for the Hubble Frontier Fields cluster
**Abell 2744** ("Pandora's Cluster", z = 0.308), built from the published
lens-model inputs of **Bergamini et al. 2023 (A&A 670, A60)**, retrieved from
VizieR catalogue `J/A+A/670/A60`:

- `point_datasets.csv` — multiple-image positions and spectroscopic redshifts.
  From table A1, keeping the 7 "gold" systems (≥3 images, all quality flag 3;
  one knot per source family): 25 images of sources at z = 1.688 to 5.662.
  Positions are (y, x) arc-second offsets about the projected cluster core
  (RA, Dec) = (3.5875, -30.3972) — the same centre used by the weak-lensing
  examples (`scripts/weak/`), so strong and weak constraints share a frame.
  `positions_noise = 0.5"` is the standard image-plane positional uncertainty
  adopted by published cluster models.
- `scaling_galaxies.csv` — 188 cluster members within 80" of the core, from
  table B1, with luminosities `L = 10^(-0.4 (mF160W - m_BCG))` relative to
  the brightest core member (the BCG, member 36034).
- `mass.csv` / `point.csv` — the named-galaxy model CSVs: the two brightest
  core members as individually-modelled dPIE galaxies, one NFW host halo
  centred on the BCG, and one `Point` per source (centres initialised at the
  mean of each system's observed images).
- `data.fits` — not committed; `scripts/cluster/start_here.py` downloads an
  HST H-band cutout from the CDS hips2fits service on first run, for
  visualization only.

`prep.py` regenerates the CSVs from VizieR. The multiple-image and member
catalogues are © the original authors; cite Bergamini et al. 2023 (and for
the HFF data, Lotz et al. 2017) in any publication using them.
