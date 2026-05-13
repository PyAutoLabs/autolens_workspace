---
name: al_load_results
description: Load a single PyAutoLens lens model-fit's results from its output folder via the .json and .fits files. Use when a user wants to inspect, plot, or further analyse one (or a handful of) completed fits by pointing at the output directory.
---

# Load Lens Model Results (Single Fit)

This skill lets the agent load a completed PyAutoLens model-fit's results back into Python from the files that `search.fit(...)` wrote to disk. Everything it does is built around the patterns in `scripts/guides/results/start_here.py` — that script is the canonical reference and the agent should read it before running any loading code.

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

In one or two sentences, tell the user what just became available — for example:

> I can now load a completed lens model-fit's results from its output folder (Tracer, fitted model, samples, dataset, and FITS imaging products) directly from the `.json` and `.fits` files written by `search.fit(...)`. If you have a large sample of fits (>100), the aggregator / workflow / database routes under `scripts/guides/results/` are usually a better fit — a dedicated `al_load_results_many` skill is planned for that case.

Then ask the user which fit folder to load from, if they have not already given one.

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

## Steps

1. **Tell the user what's available.** Use the 1–2 sentence message above. Make sure to mention the >100-fits caveat and the planned `al_load_results_many` skill.

2. **Read the canonical guide.** Open `scripts/guides/results/start_here.py` and reuse its API calls verbatim. Do not invent new loading patterns — if the guide does not show how to load something, surface that gap to the user rather than guessing.

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

## Reference

- **Canonical guide:** `scripts/guides/results/start_here.py` — single source of truth for the simple-loading API.
- **Bulk-analysis alternatives (NOT this skill's scope):**
  - `scripts/guides/results/aggregator/` — generator-based iteration over many fits.
  - `scripts/guides/results/workflow/` — bulk CSV / PNG / FITS summary makers.
  - `scripts/guides/results/database/` — `.sqlite`-backed querying for very large samples.
- **Future skill:** `al_load_results_many` (planned) will wrap the bulk routes above. Reference it by name when redirecting users away from this skill for large samples.
