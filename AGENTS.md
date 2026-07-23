# PyAutoLens Workspace — Agent Instructions

This is the tutorial and example workspace for **PyAutoLens**, a Python library for strong
gravitational lens modeling. These are the canonical, agent-agnostic instructions for this repo.

## Repository Structure

- `scripts/` — Runnable Python scripts, organised by topic:
  - `imaging/` — CCD imaging (HST, JWST, Euclid) lens modeling
  - `interferometer/` — ALMA / JVLA uv-plane lens modeling
  - `multi/` — Multi-wavelength simultaneous modeling
  - `point_source/` — Point-source (e.g. lensed quasar) modeling
  - `group/` — Group-scale lenses (multiple lens galaxies)
  - `cluster/` — Cluster-scale lenses
  - `weak/` — Weak lensing
  - `guides/` — API guides: `modeling/`, `results/`, `plot/`, `profiles/`, `units/`, `hpc/`,
    `advanced/`
- `notebooks/` — Jupyter notebook versions, generated from `scripts/` (do not edit directly)
- `config/` — PyAutoLens configuration YAML files
- `dataset/` — Example imaging, interferometer, and point-source datasets
- `output/` — Model-fit results (generated at runtime, not committed)

SLaM (Source, Light and Mass) pipelines are not a top-level directory; they live under
`features/slam/` (and similar feature subfolders) within a given topic. The HowToLens tutorial
lecture series is a **separate repo** (https://github.com/PyAutoLabs/HowToLens), not part of this
workspace.

## Running Scripts

Scripts are run **from the repository root** so relative paths to `dataset/` and `output/`
resolve correctly:

```bash
python scripts/imaging/start_here.py
```

Each topic folder has a `start_here.py` that is the canonical, always-current reference for that
topic. Some other scripts in a folder depend on results produced by its `start_here.py`.

### Standard imports

```python
import autofit as af
import autolens as al
import autolens.plot as aplt
```

For the canonical end-to-end modeling workflow (load dataset → mask → over-sample → compose model
→ configure search → analysis → fit), read `scripts/imaging/start_here.py` rather than relying on
a recipe here — that script is kept current with the API.

## Testing

On CI, every PR is gated by three workflows on Python **3.12 and 3.13**: `smoke_tests.yml` (the
smoke runner below — the definition of green), `navigator_check.yml` (PyAutoHands's reusable
navigator-catalogue check; see *Notebooks vs Scripts*), and `url_check.yml` (link checking). The
smoke and navigator jobs check out **PyAutoHands** as a sibling and run the PyAuto* libraries from
the **same-named branch** of each source repo, so a workspace PR is validated against matching
library branches.

The smoke-test runner verifies that the listed scripts (and notebooks) still execute end-to-end:

```bash
python .github/scripts/run_smoke.py
```

Run it from the repo root. It is driven by `smoke_tests.txt` (scripts) and `smoke_notebooks.txt`
(notebooks) in the workspace root, with per-entry environment from `config/build/profile_smoke.yaml`.
The runner continues through failures, prints a `[PASS]` / `[FAIL (exit N)]` line per entry with
its captured stdout+stderr, ends with a `=== Smoke test summary: P/T passed ===` line listing each
failure, and exits non-zero if any entry failed. (There is no `run_all_scripts.sh` and no
`failed/` directory — those do not exist.)

`profile_smoke.yaml` applies fast-mode defaults to every entry, notably:

- `PYAUTO_TEST_MODE=2` — skips the non-linear search's sampling, turning a run into a fast
  structural / integration check that the model composes correctly and the script/pipeline runs
  end-to-end, without paying for inference. (`1` = reduced iterations; `0` = normal.)
- `PYAUTO_SMALL_DATASETS=1` — caps grids/masks to 15×15 px so simulators and downstream
  computation are dramatically faster.
- Plus skip/fast flags (`PYAUTO_SKIP_FIT_OUTPUT`, `PYAUTO_SKIP_VISUALIZATION`,
  `PYAUTO_SKIP_CHECKS`, `PYAUTO_FAST_PLOTS`).

A script that fails under these flags indicates a real problem (broken import, renamed API, etc.).

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default home/source-tree cache locations, point
them at writable directories:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/imaging/start_here.py
```

## Notebooks vs Scripts

Notebooks in `notebooks/` are **generated** from the `.py` files in `scripts/`. **Always edit the
`.py` scripts, never the `.ipynb` notebooks directly.** The `# %%` marker alternates between code
and markdown cells.

### Generating notebooks

After updating scripts, regenerate the notebooks using the PyAutoHands tool (run from the
workspace root):

```bash
pip install ipynb-py-convert
git clone https://github.com/PyAutoLabs/PyAutoHands.git ../PyAutoHands
PYTHONPATH=../PyAutoHands/autohands python3 ../PyAutoHands/autohands/generate.py autolens
```

Commit the regenerated notebooks alongside the script changes.

## SLaM Pipelines

SLaM (Source, Light and Mass) pipelines are advanced automated modeling workflows. When working on
any SLaM script, **read `scripts/guides/modeling/slam_start_here.py` first** — it is the canonical
reference for pipeline structure, inline function signatures, docstring style, and model-building
patterns. All other SLaM scripts are documented relative to it.

## Bulk-edit safety

When editing the same region across many scripts in one pass (adding a section, renaming a symbol,
updating an import block), only rewrite the targeted region. **Never produce a whole-file write
unless you have read the entire current contents of that file** — a whole-file write based on a
header skim silently deletes every section below the header. (This rule exists because of a real
incident where a header-insert pass instead replaced ~80% of 17 scripts with the header alone.)

A guard tracks this: `.script_sizes.json` records the byte size of every `scripts/**/*.py`. Run
`scripts/check_sizes.sh` to flag any script that shrank by >50% since the snapshot. If shrinkage is
intentional, confirm with `ALLOW_SHRINK=1` and refresh via `scripts/check_sizes.sh --update` in the
same diff. Prefer targeted `Edit` over whole-file `Write`; after a bulk pass, run
`scripts/check_sizes.sh` before committing.

## Scientific Context

For strong-lensing background — concepts (mass models, source reconstruction, mass-sheet
degeneracy, multipoles, substructure, time-delay cosmography), named entities (SLACS, H0liCOW,
Euclid Q1, …), and bibliography — consult the `autolens_assistant` literature wiki at
https://github.com/PyAutoLabs/autolens_assistant (`wiki/literature/` — concepts, entities,
sources). If `autolens_assistant` is cloned as a sibling, read it locally at
`../autolens_assistant/wiki/literature/`. Pull from it on demand when an example or narrative
script would benefit from background — explaining a mass-model choice, citing a survey, framing a
pipeline phase.

## Related Repos

The PyAutoLens stack (all on the `PyAutoLabs` GitHub org):

- https://github.com/PyAutoLabs/PyAutoNerves — configuration handling
- https://github.com/PyAutoLabs/PyAutoArray — arrays, grids, masks
- https://github.com/PyAutoLabs/PyAutoFit — model composition + non-linear search
- https://github.com/PyAutoLabs/PyAutoGalaxy — light/mass profiles, galaxies
- https://github.com/PyAutoLabs/PyAutoLens — tracer, ray-tracing, lensing fits
- https://github.com/PyAutoLabs/PyAutoHands — notebook generation + CI
- https://github.com/PyAutoLabs/HowToLens — tutorial lecture series that teaches strong lensing
  from first principles; the starting point for beginners new to lensing
- https://github.com/PyAutoLabs/autolens_assistant — science-assistant workspace (literature
  wiki; see Scientific Context above)

For local development, these are typically cloned as siblings of this repo (`../PyAutoLens`,
`../PyAutoGalaxy`, `../PyAutoHands`, etc.).

## Task Workflows

**`[API Update]` issues:** read the PR diff, identify every renamed/moved/removed/changed public API,
search **all** `.py` files in `scripts/` for the old API, and update each (preserving behaviour,
docstrings, comments). Run `python .github/scripts/run_smoke.py` and fix `[FAIL]` entries until the
summary passes; leave any script you can't fix unchanged and list it under **"Could not update"**.
Regenerate the notebooks (see *Generating notebooks*) once scripts pass.

**General (non-API) issues:** read the issue and any linked plan; create/modify only files in
`scripts/` (never edit `notebooks/` directly); preserve docstrings, comments, and tutorial prose;
test with `run_smoke.py`; regenerate notebooks after.

**PR description:** summarise what changed and why, list the scripts touched, confirm notebooks were
regenerated, and add a "Could not update" section for any still-failing scripts.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->
