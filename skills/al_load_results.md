---
name: al_load_results
description: Load a single PyAutoLens lens model-fit's results from its output folder via the .json and .fits files. Use when a user wants to inspect, plot, or further analyse one (or a handful of) completed fits by pointing at the output directory.
---

# Load Lens Model Results (Single Fit)

This skill lets the agent load a completed PyAutoLens model-fit's results back into Python from the files that `search.fit(...)` wrote to disk. It should also help users learn what those results are, where they live in the workspace, how to read the matching notebooks / scripts, and how to share completed fit folders with other PyAutoLens users.

Everything it does is built around the patterns in `scripts/guides/results/start_here.py` and `notebooks/guides/results/start_here.ipynb` — these are the canonical references and the agent should read them before running any loading code.

The scope is deliberately narrow: **one fit at a time, in memory**. Bulk analysis of many fits is a separate concern with separate APIs (aggregator / workflow / database); a future `al_load_results_many` skill will wrap those.

## When to use this skill

- Inspecting a single completed fit (or a handful) in memory.
- Reproducing plots from a finished run without re-fitting.
- Pulling parameter values, errors, or derived quantities for one fit.
- Comparing a fitted model image / residuals to the input data.

## When NOT to use this skill

- Bulk analysis of a sample of fits, especially **more than ~100**. The simple `.json` / `.fits` loading path holds every object in memory and is slow for large samples. The bulk routes under `scripts/guides/results/` are designed for this:
  - `scripts/guides/results/aggregator/` — generator-based, memory-bounded.
  - `scripts/guides/results/workflow/` — bulk CSV / PNG / FITS summary makers.
  - `scripts/guides/results/database/` — `.sqlite`-backed querying for very large samples.

  A dedicated `al_load_results_many` skill is planned to wrap these. If the user has >100 fits, surface the trade-off explicitly before loading anything.

## What the agent should say when this skill activates

When this skill activates, give the user a short orientation before asking for a path or running code. The message should do four things:

- Explain what loading lens modeling results means.
- Give the most relevant workspace paths and ReadTheDocs URLs for learning more.
- If the user's science case is clear, include one concrete scientific example.
- Mention that a completed output folder can be shared with another PyAutoLens user, who can load the same `.json`, `.csv` and `.fits` products without rerunning the fit.

Use wording like:

> Loading lens modeling results means taking a completed PyAutoLens fit from `output/` and reconstructing the objects written by `search.fit(...)`, such as the maximum likelihood `Tracer`, model, samples, dataset and FITS fit products. The best local starting points are `notebooks/guides/results/start_here.ipynb` and `scripts/guides/results/start_here.py`; for the underlying API, see the PyAutoLens docs at https://pyautolens.readthedocs.io/en/latest/, especially the Galaxy / Tracer, fitting and plotting API pages.

Then adapt the next sentence to the user if possible:

- If they mention HST, Euclid, JWST, CCD or imaging data: "For example, if you have modeled HST imaging, this skill can load the completed fit from hard disk so you can inspect the inferred mass model, source reconstruction, residuals and posterior samples without rerunning the non-linear search."
- If they mention ALMA, JVLA or visibilities: "For example, if you have modeled interferometer data, this skill can load the completed fit products so you can inspect the inferred tracer, samples and interferometer-specific outputs after the search has finished."
- If they mention sharing / collaboration: "You can also send the completed fit folder to another PyAutoLens user; provided they have a compatible PyAutoLens environment, they can load the same result files and reproduce your result inspection."
- If no science case is clear: "Tell me the completed fit folder, for example `output/imaging/<dataset>/modeling/<unique_hash>/`, and I can load only the result objects needed for your question."

If the user has a large sample of fits (>100), also mention that the aggregator / workflow / database routes under `scripts/guides/results/` are usually a better fit, and that a dedicated `al_load_results_many` skill is planned for that case.

## Learning resources to show users

When the skill activates, show a concise set of links / paths tailored to the task. Do not dump every reference every time; choose the smallest useful subset.

### Core result loading

- Notebook: `notebooks/guides/results/start_here.ipynb`
- Script: `scripts/guides/results/start_here.py`
- Folder README: `notebooks/guides/results/README.md`, `scripts/guides/results/README.md`
- ReadTheDocs overview: https://pyautolens.readthedocs.io/en/latest/
- Workspace tour: https://pyautolens.readthedocs.io/en/latest/general/workspace.html

### Understanding the loaded `Tracer`

- Notebook: `notebooks/guides/tracer.ipynb`
- Script: `scripts/guides/tracer.py`
- Galaxy / Tracer API docs: https://pyautolens.readthedocs.io/en/latest/api/galaxy.html
- PyAutoLens overview with `Tracer` examples: https://pyautolens.readthedocs.io/en/latest/overview/overview_1_start_here.html

### Inspecting galaxies and components

- Notebook: `notebooks/guides/galaxies.ipynb`
- Script: `scripts/guides/galaxies.py`
- Galaxy / Tracer API docs: https://pyautolens.readthedocs.io/en/latest/api/galaxy.html

### Samples, parameter errors and posterior analysis

- Notebook: `notebooks/guides/results/aggregator/samples.ipynb`
- Script: `scripts/guides/results/aggregator/samples.py`
- PyAutoFit Samples API docs: https://pyautofit.readthedocs.io/en/latest/api/samples.html

### Fits, residuals and plotting

- Notebook: `notebooks/guides/plot/examples/plotters.ipynb`
- Script: `scripts/guides/plot/examples/plotters.py`
- Fitting API docs: https://pyautolens.readthedocs.io/en/latest/api/fitting.html
- Plotting API docs: https://pyautolens.readthedocs.io/en/latest/api/plot.html

### Bulk result libraries

For many fits, point users at these instead of this single-fit loading workflow:

- Notebooks: `notebooks/guides/results/aggregator/`, `notebooks/guides/results/workflow/`, `notebooks/guides/results/database/start_here.ipynb`
- Scripts: `scripts/guides/results/aggregator/`, `scripts/guides/results/workflow/`, `scripts/guides/results/database/start_here.py`

## Output folder layout (what you will be reading)

Each completed fit lives at a path of the form:

```
output/imaging/<dataset_name>/modeling/<unique_hash>/
    files/
        tracer.json            ← max log likelihood Tracer
        model.json             ← fitted af.Collection model
        samples.csv            ← full non-linear search samples
        samples_summary.json   ← max log likelihood parameter values + errors
        samples_info.json      ← metadata about the samples
        search.json            ← non-linear search configuration
        settings.json          ← search settings
        cosmology.json         ← cosmology used for the fit
        covariance.csv         ← parameter covariance matrix
    image/
        dataset.fits           ← data, noise-map and PSF
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

Sub-paths vary slightly for interferometer / multi-wavelength fits — defer to the guide if the layout differs.

## Sharing results with other PyAutoLens users

A completed fit can be shared by sending the whole hashed output folder, for example `output/imaging/<dataset_name>/modeling/<unique_hash>/`. The key files for another user are usually `files/tracer.json`, `files/model.json`, `files/samples.csv`, `files/samples_summary.json`, `files/search.json`, `files/settings.json`, `files/cosmology.json`, the `image/*.fits` products, and the human-readable `model.info` / `model.results` summaries.

When helping a user share results, tell them to preserve the folder structure and note the PyAutoLens / workspace version used to produce the fit. The receiving user can then use this skill to load the result without rerunning the non-linear search, provided their environment is compatible with the serialized objects.

## Steps

1. **Orient the user first.** Give the short user-facing summary from "What the agent should say when this skill activates". Include the most relevant local notebook / script paths and ReadTheDocs URLs from "Learning resources to show users". If the user's science case is clear, add a specific example of how loading completed results helps their data analysis. Make sure to mention the >100-fits caveat and the planned `al_load_results_many` skill when appropriate.

2. **Read the canonical guide.** Open `scripts/guides/results/start_here.py` and, when useful for user-facing explanations, `notebooks/guides/results/start_here.ipynb`. Reuse the script's API calls verbatim. Do not invent new loading patterns — if the guide does not show how to load something, surface that gap to the user rather than guessing. This guide also contains the first layer of post-load context: samples, fits, tracers, galaxies, units, linear light profiles and pixelizations.

3. **Identify the fit folder.** If the user has not given a path, ask for one of the form `output/imaging/<dataset>/modeling/<unique_hash>/`. If they point at a parent directory (e.g. just `output/imaging/<dataset>/modeling/`), list its contents and ask which hashed sub-folder.

4. **Load only what was requested.** Use the patterns from the guide:

   - `from_json(file_path=".../files/tracer.json")` → `Tracer`
   - `from_json(file_path=".../files/model.json")` → `af.Collection`
   - `af.SamplesNest.from_table(filename=".../files/samples.csv", model=model)` → `Samples`
   - `al.Imaging.from_fits(...)` for `image/dataset.fits`
   - `al.Array2D.from_fits(...)` (or the relevant API in the guide) for the per-galaxy / per-plane FITS products

   Don't eagerly load every artefact — match what the user asked about.

5. **Print every file path being read** so the user can verify them and spot wrong-folder mistakes early.

6. **Handle missing files gracefully.** If a fit was killed mid-run, the `image/` folder may be incomplete or absent. Say so explicitly and offer to fall back to whatever IS present (e.g. samples but no fit images) rather than crashing silently.

7. **Stay read-only.** This skill never modifies anything inside `output/`. If the user wants derived artefacts saved somewhere, write them outside the fit's folder (e.g. into a working directory the user nominates).

## After loading: what to do with each object

Loading is usually only the first step. After an object is loaded, route follow-up analysis through the workspace guides below before writing new code. Prefer the local `scripts/guides/` examples because the user is running from `autolens_workspace` and these scripts should match the workspace conventions, datasets, plotting API and tutorial style.

### If a `Tracer` was loaded

Read `scripts/guides/tracer.py` before manipulating or interpreting the tracer. It is the main reference for:

- evaluating a full model image with `tracer.image_2d_from(grid=...)`;
- tracing an image-plane grid to source-plane grids with `tracer.traced_grid_2d_list_from(grid=...)`;
- computing scalar lensing quantities such as convergence and potential;
- computing vector quantities such as deflection angles;
- using different grid choices, sub-grids, irregular position grids and `slim` / `native` array representations;
- accessing galaxies and profile attributes through `tracer.galaxies` and `tracer.planes`.

For magnification maps, critical curves, caustics and image-position overlays, also read:

- `scripts/guides/plot/examples/visuals.py`
- `scripts/guides/plot/examples/plotters.py`

For multi-plane ray tracing or tracers with more than two redshift planes, read:

- `scripts/guides/advanced/multi_plane.py`

Do not mutate the loaded result object in-place unless the user explicitly asks for exploratory scratch work. For derived or modified tracers, create a new `al.Tracer` from copied or newly constructed `Galaxy` / profile objects so the original maximum log likelihood result remains reproducible.

### If `Galaxy`, light profile or mass profile components are needed

Read `scripts/guides/galaxies.py` when the user asks to inspect or manipulate individual lens/source galaxies or profile components inside a tracer. It covers extracting image-plane and source-plane galaxies, evaluating individual bulge / disk / mass components, and accessing their parameters.

Use `scripts/guides/tracer.py` as the companion reference for lower-level profile methods like:

- light profile images via `image_2d_from(grid=...)`;
- mass profile convergence via `convergence_2d_from(grid=...)`;
- mass profile deflections via `deflections_yx_2d_from(grid=...)`.

### If `Samples` were loaded

Use the samples section of `scripts/guides/results/start_here.py` first, then read `scripts/guides/results/aggregator/samples.py` for deeper examples. These show how to:

- access the maximum log likelihood instance with `samples.max_log_likelihood()`;
- access the median PDF instance with `samples.median_pdf()`;
- compute upper / lower sigma instances with `samples.values_at_upper_sigma(...)` and `samples.values_at_lower_sigma(...)`;
- derive errors on fitted and derived quantities, for example an Einstein radius;
- make posterior / corner plots where supported by the plotting API.

If the user asks for a quantity involving a linear light profile intensity or pixelized source reconstruction, do not rely on the raw `Samples` instance alone. Recreate the appropriate `FitImaging` / fit object as shown in `scripts/guides/results/start_here.py`, because linear-profile intensities and inversion quantities are solved during the fit and may not be represented directly as ordinary sampled parameters.

### If imaging, fit images or residual maps were loaded

Use `scripts/guides/results/start_here.py` for loading the FITS products, then use `scripts/guides/plot/examples/plotters.py` for plotting arrays and fit products. Common follow-up tasks include plotting or summarising:

- data, noise map and PSF from `image/dataset.fits`;
- model image, residual map, normalized residual map and chi-squared map from `image/fit.fits`;
- per-galaxy or per-plane images from `image/tracer.fits`, `image/galaxy_images.fits` or `image/model_galaxy_images.fits`.

If the user asks to compare loaded FITS products against recomputed model images, load the `Tracer`, load or reconstruct the dataset grid, compute `tracer.image_2d_from(grid=dataset.grid)`, and compare against the appropriate model-image HDU.

### If physical units or cosmological quantities are requested

Use the units guidance in `scripts/guides/results/start_here.py` first. For mass-to-light and physical-unit conversion examples, read:

- `scripts/guides/units/mass_to_light_ratio_units.py`

Be explicit about whether a reported quantity is in PyAutoLens internal angular units (usually arcseconds / dimensionless lensing quantities) or converted physical units.

### If pixelizations or inversions are involved

Pixelized source reconstructions and inversion-derived quantities are not handled by a loaded `Tracer` alone. Use the relevant sections of `scripts/guides/results/start_here.py` and the plotting examples:

- `scripts/guides/plot/advanced/plotters_pixelization.py`
- `scripts/guides/plot/advanced/plotters_double_einstein_ring.py`

Recreate the fit / inversion object when necessary rather than trying to infer inversion state from `tracer.json`.

## Documentation and source-code references

Use references in this order:

1. **Workspace guides first:** `scripts/guides/` is the best starting point for agents running inside `autolens_workspace`, because these files are local, executable and written in the same tutorial style as the scripts the user is editing.
2. **PyAutoLens ReadTheDocs for human-facing documentation:** use https://pyautolens.readthedocs.io/en/latest/index.html when the user wants conceptual documentation, API docs, tutorials outside the workspace, or links they can read in a browser.
3. **Installed / active source for exact API behavior:** when exact signatures, class locations or implementation details matter, inspect the active source imported by the user's environment. From this workspace, use:

   ```bash
   PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -c "import autolens, inspect, pathlib; print(pathlib.Path(inspect.getfile(autolens)).parent)"
   ```

   Then inspect the relevant files under that path. If the workspace version and installed library version differ, trust the active installed source for runtime behavior, but keep any tutorial edits consistent with the workspace scripts.

## Reference

- **Canonical loading guide:** `notebooks/guides/results/start_here.ipynb`, `scripts/guides/results/start_here.py` — single source of truth for the simple-loading API and first post-load routing.
- **Tracer operations:** `notebooks/guides/tracer.ipynb`, `scripts/guides/tracer.py`
- **Galaxy and profile operations:** `notebooks/guides/galaxies.ipynb`, `scripts/guides/galaxies.py`
- **Plotting arrays, tracers and fits:** `notebooks/guides/plot/examples/plotters.ipynb`, `scripts/guides/plot/examples/plotters.py`
- **Critical curves, caustics and overlays:** `scripts/guides/plot/examples/visuals.py`
- **Multi-plane ray tracing:** `scripts/guides/advanced/multi_plane.py`
- **Pixelization / inversion plotting:** `scripts/guides/plot/advanced/plotters_pixelization.py`, `scripts/guides/plot/advanced/plotters_double_einstein_ring.py`
- **Physical units / mass-to-light examples:** `scripts/guides/units/mass_to_light_ratio_units.py`
- **PyAutoLens documentation:** https://pyautolens.readthedocs.io/en/latest/
- **Workspace tour:** https://pyautolens.readthedocs.io/en/latest/general/workspace.html
- **Galaxy / Tracer API docs:** https://pyautolens.readthedocs.io/en/latest/api/galaxy.html
- **Fitting API docs:** https://pyautolens.readthedocs.io/en/latest/api/fitting.html
- **Plotting API docs:** https://pyautolens.readthedocs.io/en/latest/api/plot.html
- **PyAutoFit Samples API docs:** https://pyautofit.readthedocs.io/en/latest/api/samples.html
- **Bulk-analysis alternatives (NOT this skill's scope):**
  - `scripts/guides/results/aggregator/` — generator-based iteration over many fits.
  - `scripts/guides/results/workflow/` — bulk CSV / PNG / FITS summary makers.
  - `scripts/guides/results/database/` — `.sqlite`-backed querying for very large samples.
- **Future skill:** `al_load_results_many` (planned) will wrap the bulk routes above. Reference it by name when redirecting users away from this skill for large samples.
