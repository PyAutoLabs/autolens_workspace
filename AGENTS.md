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

The smoke-test runner verifies that the listed scripts (and notebooks) still execute end-to-end:

```bash
python .github/scripts/run_smoke.py
```

Run it from the repo root. It is driven by `smoke_tests.txt` (scripts) and `smoke_notebooks.txt`
(notebooks) in the workspace root, with per-entry environment from `config/build/env_vars.yaml`.
The runner continues through failures, prints a `[PASS]` / `[FAIL (exit N)]` line per entry with
its captured stdout+stderr, ends with a `=== Smoke test summary: P/T passed ===` line listing each
failure, and exits non-zero if any entry failed. (There is no `run_all_scripts.sh` and no
`failed/` directory — those do not exist.)

`env_vars.yaml` applies fast-mode defaults to every entry, notably:

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

After updating scripts, regenerate the notebooks using the PyAutoBuild tool (run from the
workspace root):

```bash
pip install ipynb-py-convert
git clone https://github.com/PyAutoLabs/PyAutoBuild.git ../PyAutoBuild
PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py autolens
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

- https://github.com/PyAutoLabs/PyAutoConf — configuration handling
- https://github.com/PyAutoLabs/PyAutoArray — arrays, grids, masks
- https://github.com/PyAutoLabs/PyAutoFit — model composition + non-linear search
- https://github.com/PyAutoLabs/PyAutoGalaxy — light/mass profiles, galaxies
- https://github.com/PyAutoLabs/PyAutoLens — tracer, ray-tracing, lensing fits
- https://github.com/PyAutoLabs/PyAutoBuild — notebook generation + CI
- https://github.com/PyAutoLabs/HowToLens — tutorial lecture series that teaches strong lensing
  from first principles; the starting point for beginners new to lensing
- https://github.com/PyAutoLabs/autolens_assistant — science-assistant workspace (literature
  wiki; see Scientific Context above)

For local development, these are typically cloned as siblings of this repo (`../PyAutoLens`,
`../PyAutoGalaxy`, `../PyAutoBuild`, etc.).

## Task Workflows

### API Update tasks

When assigned an issue titled `[API Update]`:

1. Read the PR diff in the issue body. Identify every renamed, moved, removed, or changed public
   API (functions, classes, method signatures, parameter names, import paths).
2. Search **all** `.py` files in `scripts/` for usages of the old API.
3. Update each file to the new API, preserving existing behaviour, docstrings, and comments.
4. Run `python .github/scripts/run_smoke.py` (from the repo root) to test.
5. Read the failure output for any `[FAIL]` entries and fix the affected scripts.
6. Repeat 4–5 until the summary reports all passed.
7. If a script cannot be fixed (ambiguous change, missing dependency), leave it unchanged and list
   it in the PR description under **"Could not update"** with the reason.
8. After all scripts pass, regenerate the notebooks (see "Generating notebooks").

### General Issue tasks

When assigned a general (non-API) issue:

1. Read the issue description and any linked plan or AI prompt.
2. Identify which scripts need to be created or modified.
3. Only edit files in `scripts/`. Never edit `notebooks/` directly.
4. Preserve all docstrings, comments, and tutorial explanations.
5. Test with `python .github/scripts/run_smoke.py` after changes.
6. Regenerate notebooks after all scripts pass.

### PR description

When opening your PR, include:

- A summary of what changed and why.
- A list of all scripts you updated or created.
- Confirmation that notebooks were regenerated.
- A "Could not update" section for any scripts that still fail, with the error and your assessment.

## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset for AI workflow", or
  any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

```bash
git fetch origin
git reset --hard origin/main
git clean -fd
```

This applies equally to humans, local agents, cloud agents, and any other tool. The "Initial commit
— fresh start" pattern this prevents has cost ~40 commits of redundant rework each time it has
happened.
