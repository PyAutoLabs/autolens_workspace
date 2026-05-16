---
name: al_load_results
description: Load a single PyAutoLens lens model-fit's results from its output folder via the .json and .fits files. Use when a user wants to inspect, plot, or further analyse one (or a handful of) completed fits by pointing at the output directory. Not for bulk analysis of >100 fits — see the planned al_load_results_many skill instead.
---

# Loading a finished lens model fit

This skill helps a user pull a completed PyAutoLens fit back into memory and start interpreting it. It is the entry point for *"the search finished — what did it actually learn about my lens system?"*

Loading is rarely the goal in itself. The user already wants to look at *something*: the mass model, the source reconstruction, the residuals, the posterior, or a comparison to another fit. The skill's job is to figure out which, load only what is needed, and route the user into the workspace material that teaches that piece of the science. Read this skill as a conversation, not a recipe — the procedural checklist at the bottom is for the agent's own sanity, not for the user.

The canonical reference for everything below is `scripts/guides/results/start_here.py` and its notebook sibling `notebooks/guides/results/start_here.ipynb`. Follow what those files do; don't invent new patterns.

## Orient — what loading results actually means

When this skill activates, deliver a short orientation before asking for a path or touching code.

Loading results means taking the output folder that `search.fit(...)` wrote (typically `output/imaging/<dataset>/modeling/<unique_hash>/`) and reconstructing the lens model objects from disk. By default, what comes back describes the **maximum log likelihood** result — the single best-fitting model the non-linear search found. The full posterior is also there, in `files/samples.csv`, which loads as a `Samples` object whenever you want errors, posterior medians, or upper / lower sigma bounds.

The most common file-to-object mappings are:

- `files/tracer.json` → `Tracer` (the max-log-likelihood lens model: galaxies, light profiles, mass profiles, redshifts).
- `files/model.json` → the fitted `af.Collection` model definition.
- `files/samples.csv` → `Samples` (full posterior of accepted samples).
- `image/dataset.fits` → the dataset (`Imaging`, `Interferometer` or `PointDataset`, depending on the analysis).
- `image/fit.fits` → model image, residuals, normalised residuals, chi-squared map.

Loaded objects can be recombined into a fit on demand. For example `al.FitImaging(dataset=dataset, tracer=tracer)` rebuilds the full `FitImaging` so you can inspect log likelihood, residuals, linear-profile intensities, or inversion outputs — none of which live inside `tracer.json` on its own.

Tailor the next sentence to what the user has told you:

- *HST, JWST, Euclid, CCD imaging:* "If you've modelled HST / JWST / Euclid imaging, this lets you inspect the inferred mass model, source reconstruction, residuals and posterior without re-running the search."
- *ALMA, JVLA, visibilities:* "For interferometer data, you can load the completed fit products and inspect the tracer, samples, and interferometer-specific outputs after the search has finished."
- *Sharing with a collaborator:* "You can hand the whole output folder to another PyAutoLens user — provided their environment is compatible, they can load the same `.json`, `.csv` and `.fits` products without re-running anything."
- *Unclear:* just ask. "Point me at the fit folder (something like `output/imaging/<dataset>/modeling/<hash>/`) and tell me what you want to look at."

If the user has hundreds of finished fits, this is the wrong skill — see "When this isn't the right skill" below.

## Ask — what does the user want to learn?

Before loading anything, ask what they want out of the result. The answer chooses the branch and tells you how deep to go. Useful prompts:

- *"Are you trying to understand the inferred **mass model** — the deflection field, convergence, Einstein radius?"*
- *"Or the **source** — its reconstructed light or pixelised inversion?"*
- *"Or the **posterior** — errors on parameters, median values, covariances?"*
- *"Or **how well the fit actually matches the data** — residual maps, chi-squared, model images?"*
- *"Or are you **comparing this fit to another run**?"*

If they have already said in their opening message (*"load fit X and show me residuals"*), skip the question and go straight to the relevant branch. If they say *"I'm new to lensing"* or *"what's a Tracer?"*, slow down — frame the physics each time before reaching for an API call. If they just point at a fit folder with no question attached, ask the disambiguating question once.

Also ask for the path if they have not given one: something of the form `output/imaging/<dataset>/modeling/<unique_hash>/`. If they point at a parent (e.g. `output/imaging/<dataset>/modeling/`), list its contents and ask which hashed sub-folder.

## Branches — what to do once you know the question

### If they want the lens / mass model — load the `Tracer`

The `Tracer` is PyAutoLens's representation of the lens system: an ordered set of `Galaxy` objects with their light profiles, mass profiles and redshifts. Operationally, it is the thing that knows how to compute deflection angles, convergence, magnification, critical curves, and how to ray-trace image-plane positions back to the source plane.

Load it from `files/tracer.json` with the `from_json` pattern shown in the canonical guide:

```python
from autoconf.dictable import from_json
tracer = from_json(file_path=".../files/tracer.json")
```

For everything you might do with the loaded `Tracer` — evaluating a full model image, tracing grids to the source plane, computing convergence / potential / deflections, sub-grid choices, `slim` vs `native` arrays, accessing galaxies via `tracer.galaxies` or `tracer.planes` — read `scripts/guides/tracer.py` (notebook: `notebooks/guides/tracer.ipynb`). Don't write new tracer-manipulation code without checking the guide first.

For magnification maps, critical curves, caustics, and image-position overlays, read `scripts/guides/plot/examples/visuals.py` and `scripts/guides/plot/examples/plotters.py`. For multi-plane ray tracing (more than two redshift planes), read `scripts/guides/advanced/multi_plane.py`.

Don't mutate the loaded `Tracer` in place. If the user wants a modified tracer — different parameter values, a swapped profile — build a new `al.Tracer` from copied `Galaxy` / profile objects so the original max-log-likelihood result stays intact.

Conceptual overview: https://pyautolens.readthedocs.io/en/latest/overview/overview_1_start_here.html
API reference: https://pyautolens.readthedocs.io/en/latest/api/galaxy.html

Once the `Tracer` is loaded, ask the user if they want to dig into how deflections are computed, or how ray-tracing maps image-plane grids to source-plane grids — those are natural follow-up conversations.

### If they want individual galaxies or profile components

Often what the user actually wants is one piece of the system: the lens galaxy's mass profile by itself, the source galaxy's light, the lens bulge separately from its disc, the convergence of a dark-matter halo. These all live inside the loaded `Tracer`.

Read `scripts/guides/galaxies.py` (notebook: `notebooks/guides/galaxies.ipynb`) before manipulating them. It covers:

- pulling image-plane and source-plane galaxies out of a tracer,
- evaluating an individual light profile's image via `light_profile.image_2d_from(grid=...)`,
- evaluating mass-profile convergence via `mass_profile.convergence_2d_from(grid=...)`,
- evaluating mass-profile deflections via `mass_profile.deflections_yx_2d_from(grid=...)`,
- accessing parameters and redshifts.

`scripts/guides/tracer.py` is the companion reference for any lower-level method that lives on the profile itself.

API reference: https://pyautolens.readthedocs.io/en/latest/api/galaxy.html

If the user is new to PyAutoLens's split between `Tracer`, `Galaxy`, `LightProfile` and `MassProfile`, this is a good moment to invite a follow-up question — that hierarchy is worth a paragraph of explanation if it's their first time meeting it.

### If they want the posterior — load `Samples`

The `Samples` object holds every accepted sample from the non-linear search. It is how you get parameter errors, posterior medians, upper / lower sigma instances, and any derived quantity that needs uncertainty propagation (Einstein radius, mass-to-light ratio, half-light radius of the source, …).

Load with:

```python
import autofit as af
samples = af.SamplesNest.from_table(filename=".../files/samples.csv", model=model)
```

where `model` is loaded from `files/model.json` via `from_json`.

What you can do with it is covered in the samples section of `scripts/guides/results/start_here.py`, with deeper examples in `scripts/guides/results/aggregator/samples.py`:

- `samples.max_log_likelihood()` — the single best-fitting instance,
- `samples.median_pdf()` — the median PDF instance,
- `samples.values_at_upper_sigma(sigma=...)` / `samples.values_at_lower_sigma(...)` — error bounds,
- errors on derived quantities, e.g. computing an Einstein radius for every sample to get an uncertainty on it,
- corner / posterior plots, where supported by the plotting API.

PyAutoFit Samples API reference: https://pyautofit.readthedocs.io/en/latest/api/samples.html

One subtlety worth flagging for the user: **linear light profile intensities and inversion quantities are solved during the fit itself, not sampled by the non-linear search.** If they want those quantities with uncertainties they have to recreate the appropriate `FitImaging` (or inversion) object — see the next branch.

### If they want to compare model to data — rebuild the fit

When the user asks about residuals, normalised residuals, chi-squared maps, model images per galaxy, or *"is this a good fit?"*, they want a fit object, not just a `Tracer`. Two paths:

**Read the saved fit FITS products.** `image/fit.fits`, `image/tracer.fits`, `image/galaxy_images.fits`, `image/model_galaxy_images.fits` and `image/source_plane_images.fits` are written at the end of the fit. Load them with `al.Array2D.from_fits(...)` (or the relevant API the guide shows). This is fast and read-only — fine for "show me the residuals saved during the run."

**Recreate the fit object.** Load `image/dataset.fits` into the appropriate dataset (`al.Imaging.from_fits(...)`), load the `Tracer`, then build the fit:

```python
fit = al.FitImaging(dataset=dataset, tracer=tracer)
```

This is what `scripts/guides/results/start_here.py` does in its FitImaging section. Use this path when the user wants the *live* fit object — linear intensities, inversion state, log likelihood recomputed — rather than the static images saved at fit time.

For plotting, point at `scripts/guides/plot/examples/plotters.py` (notebook: `notebooks/guides/plot/examples/plotters.ipynb`).

Fitting API reference: https://pyautolens.readthedocs.io/en/latest/api/fitting.html
Plotting API reference: https://pyautolens.readthedocs.io/en/latest/api/plot.html

If the user wants to compare a saved model image (loaded from FITS) against a freshly recomputed one (`tracer.image_2d_from(grid=dataset.grid)`), do that and surface any difference. They should usually match; mismatches indicate version drift or a subtly different grid.

### If they want physical units

PyAutoLens reports most quantities in internal angular units — arcseconds and dimensionless lensing quantities. If the user wants kpc, solar masses, Einstein masses, or anything physical, they need cosmological conversions.

`scripts/guides/results/start_here.py` has the units section; for mass-to-light ratios and other physical-unit conversions, read `scripts/guides/units/mass_to_light_ratio_units.py`.

When you quote a number to the user, be explicit about the unit. *"Einstein radius = 1.2 arcsec"* is fine; *"Einstein radius = 1.2"* without a unit is a bug.

### If the fit uses a pixelisation or inversion

A pixelised source reconstruction is not a property of the saved `Tracer` alone — it requires the dataset and the fit object. The pixelisation state lives inside `FitImaging` after the inversion is solved. Read the relevant section of `scripts/guides/results/start_here.py`, and for plotting:

- `scripts/guides/plot/advanced/plotters_pixelization.py`
- `scripts/guides/plot/advanced/plotters_double_einstein_ring.py`

Don't try to read the pixelisation state out of `tracer.json` — it isn't there. Rebuild the fit.

## Sharing a finished fit with a collaborator

The entire output folder for a fit (e.g. `output/imaging/<dataset>/modeling/<unique_hash>/`) is portable. Hand the whole folder to a collaborator and, provided they have a compatible PyAutoLens environment, they can use this same skill to load it without rerunning the non-linear search.

The files that matter for another user are:

- `files/tracer.json`, `files/model.json`, `files/samples.csv`, `files/samples_summary.json`, `files/search.json`, `files/settings.json`, `files/cosmology.json`,
- the `image/*.fits` products,
- the human-readable `model.info` and `model.results` summaries.

Tell whoever you're sharing with: preserve the folder structure as-is, and note the PyAutoLens / workspace version used to produce the fit. If their environment is incompatible with serialised objects from a different version, they may need to match the version (or rerun parts of the analysis).

## When this isn't the right skill

This is a single-fit, in-memory loader. It is the wrong tool for bulk analysis. If the user has more than ~100 finished fits and wants to query, summarise or aggregate across them, the routes designed for that case live elsewhere in the workspace:

- `scripts/guides/results/aggregator/` — generator-based, memory-bounded iteration.
- `scripts/guides/results/workflow/` — bulk CSV / PNG / FITS summary makers.
- `scripts/guides/results/database/` — `.sqlite`-backed querying for very large samples.

A dedicated `al_load_results_many` skill is planned to wrap those routes. Mention it by name when redirecting, and warn that the simple `.json` / `.fits` route holds everything in memory and is slow at scale.

## Skill combinations

This skill is the loading half of any deeper inspection workflow. The interesting work happens when its outputs feed into other skills:

- **Loaded `Tracer` + a plotting skill** = critical curves, caustics, magnification maps, and image-position overlays plotted on the data — none of which any single workspace script does end-to-end.
- **Loaded `Samples` + a comparison skill** (`al_compare_fits`, planned) = posterior comparison between two model runs (e.g. SIE vs. NFW), with parameter-wise uncertainty propagation.
- **Loaded `FitImaging` + a re-fitting skill** (`al_refit_with_perturbation`, planned) = perturb a parameter, recompute residuals, see whether the fit is robust to that change.

Surface these when the user's question would benefit from chaining. The emergent power of PyAutoLens-Assistant is that the user can do things one script alone can't.

## Reference card

### Output folder layout

```
output/imaging/<dataset_name>/modeling/<unique_hash>/
    files/
        tracer.json            ← max log likelihood Tracer
        model.json             ← fitted af.Collection model
        samples.csv            ← full non-linear search samples
        samples_summary.json   ← max log likelihood parameters + errors
        samples_info.json      ← samples metadata
        search.json            ← non-linear search configuration
        settings.json          ← search settings
        cosmology.json         ← cosmology used for the fit
        covariance.csv         ← parameter covariance matrix
    image/
        dataset.fits           ← data, noise-map, PSF
        fit.fits               ← model image, residuals, chi-squared map
        tracer.fits            ← tracer image-plane images per galaxy
        source_plane_images.fits
        model_galaxy_images.fits
        galaxy_images.fits
        dataset.png, fit.png, tracer.png   ← static visualisations
    model.info                 ← human-readable model summary
    model.results              ← human-readable fit summary
    search.summary             ← search run summary
    metadata                   ← run metadata
```

Sub-paths vary slightly for interferometer / multi-wavelength fits — defer to the canonical guide if the layout differs.

### Workspace guides (read these first)

- Loading: `scripts/guides/results/start_here.py`, `notebooks/guides/results/start_here.ipynb`
- Tracer: `scripts/guides/tracer.py`, `notebooks/guides/tracer.ipynb`
- Galaxies and profiles: `scripts/guides/galaxies.py`, `notebooks/guides/galaxies.ipynb`
- Plotting (arrays, tracers, fits): `scripts/guides/plot/examples/plotters.py`
- Visuals (critical curves, caustics, overlays): `scripts/guides/plot/examples/visuals.py`
- Multi-plane ray tracing: `scripts/guides/advanced/multi_plane.py`
- Pixelisation / inversion plotting: `scripts/guides/plot/advanced/plotters_pixelization.py`, `scripts/guides/plot/advanced/plotters_double_einstein_ring.py`
- Physical units: `scripts/guides/units/mass_to_light_ratio_units.py`
- Samples deep dive: `scripts/guides/results/aggregator/samples.py`

### ReadTheDocs (human-facing reference)

- PyAutoLens main: https://pyautolens.readthedocs.io/en/latest/
- Workspace tour: https://pyautolens.readthedocs.io/en/latest/general/workspace.html
- Galaxy / Tracer API: https://pyautolens.readthedocs.io/en/latest/api/galaxy.html
- Fitting API: https://pyautolens.readthedocs.io/en/latest/api/fitting.html
- Plotting API: https://pyautolens.readthedocs.io/en/latest/api/plot.html
- PyAutoFit Samples API: https://pyautofit.readthedocs.io/en/latest/api/samples.html

### Installed source (for exact API behaviour)

When signatures or implementation details matter, locate the active installed package the user's environment is using:

```bash
PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
  python -c "import autolens, inspect, pathlib; print(pathlib.Path(inspect.getfile(autolens)).parent)"
```

Inspect files under that path. If the workspace and installed versions diverge, trust the installed source for runtime behaviour, but keep any tutorial edits consistent with the workspace scripts.

## Agent procedure (slim checklist)

The user-facing material above is the substance of this skill. The list below is for the agent's own sanity when running the skill end-to-end — it is not a script to read aloud.

1. **Orient.** Deliver the orientation paragraph, tailored to the user's data type. Mention the canonical guide path. Flag the >100-fits caveat only if relevant.
2. **Ask.** Find out which branch (mass model / galaxies / samples / fit / units / pixelisation), unless the user has already said.
3. **Read the canonical guide.** Open `scripts/guides/results/start_here.py` (and the notebook when useful) and reuse its API calls verbatim. Don't invent new loading patterns — if the guide doesn't show something, surface that gap to the user rather than guessing.
4. **Identify the fit folder.** Ask if the user has not given one. If they point at a parent (e.g. `output/imaging/<dataset>/modeling/`), list its contents and ask which hashed sub-folder.
5. **Load only what was requested.** Don't eagerly load every artefact; match the branch.
6. **Print every file path you read.** The user needs to verify them and catch wrong-folder mistakes early.
7. **Handle missing files gracefully.** If a fit was killed mid-run, `image/` may be incomplete. Say so; offer to fall back to what IS present (e.g. samples without fit images) rather than crashing silently.
8. **Stay read-only.** Never modify anything inside `output/`. Derived artefacts go to a working directory the user nominates.
9. **Invite a follow-up.** Before signing off, point at the most relevant adjacent skill or workspace script for what they would do next.
