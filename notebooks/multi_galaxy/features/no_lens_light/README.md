The `no_lens_light` folder contains example scripts for fitting a multi-galaxy strong lens whose co-dominant
deflectors have no visible light — either because it was subtracted before modeling, or because the galaxies are
undetected at the observed wavelength.

The dataset is the `simple` lens of `multi_galaxy/simulator.py` with the light removed and nothing else changed:
the same two `Isothermal` deflectors, the same Einstein radii (1.0" and 0.8"), the same shear, the same source. So
every claim these scripts make about what you lose is measured against a dataset that differs in exactly one
respect.

# The multi-galaxy difference

At galaxy scale, removing the lens light is close to a free win: fewer parameters, faster fits, and the one lens
is conventionally near (0.0", 0.0") anyway. Here it costs something the galaxy-scale example never has to pay.

**Without light, nothing in the image says where the deflectors are.** There are two of them, neither at the
origin. `multi_galaxy/modeling.py` fixes each mass centre to the galaxy's observed light; with the light gone the
centres have to be freed and anchored on information from outside this image — another band, the subtraction that
produced the data, or a catalogue position.

The parameter count makes the trade concrete for the model these scripts fit:

| Model | Free parameters |
|---|---|
| `multi_galaxy/modeling.py` (with lens light, fixed mass centres) | 20 |
| No lens light, **free** anchored mass centres (what `modeling.py` here fits) | 16 |
| No lens light, mass centres still fixed | 12 |

The saving is 4 parameters, not the 8 the light removed — and the 4 that came back are mass centres, which are
degenerate with Einstein radius, which is already degenerate between the two deflectors. Fixing the centres is
defensible when your external astrometry is good; it is a stated model assumption either way.

# Files

- `simulator`: Simulating the two mass-only co-dominant deflectors and the lensed source.
- `modeling`: Fitting the model, and where the deflector centres come from when there is no light to read.
- `slam`: The Source, Light and Mass pipeline with the lens-light stages removed — the production path.

# Related

- `multi_galaxy/modeling`: the same lens with its light, and the baseline the numbers above are quoted against.
- `imaging/features/no_lens_light`: the galaxy-scale version, where the centre problem does not arise.
- `group/features/no_lens_light`: the group-scale version, where the mass-only model spans tiers of galaxies.
- `imaging/features/pixelization`: a pixelized source, which combines naturally with a mass-only model.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
