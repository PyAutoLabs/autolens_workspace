# Copilot Coding Agent Instructions

You are working on the **autolens_workspace**, a tutorial/example repository for the PyAutoLens library.

## Key Rules

- Only edit files in `scripts/`. Never edit files in `notebooks/` (those are auto-generated).
- After making changes, always test with `bash run_all_scripts.sh`.
- If tests fail, read the log in `failed/<path>.log`, fix the script, and re-run.
- Do not add new scripts unless the issue specifically asks for it. For API updates, only update existing ones to match API changes.
- Preserve all docstrings, comments, and tutorial explanations. Only change code that uses the old API.
- When working on SLaM pipeline scripts, read `scripts/guides/modeling/slam_start_here.py` first — it is the canonical reference.

## Testing

`run_all_scripts.sh` sets `PYAUTO_TEST_MODE=1` automatically. Every script should pass in this mode. A script that fails in test mode indicates a real problem (broken import, wrong function name, etc.).

## Notebook Generation

After all scripts pass testing, regenerate the notebooks:

```bash
pip install ipynb-py-convert
git clone https://github.com/Jammy2211/PyAutoBuild.git ../PyAutoBuild
PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py autolens
```

Run this from the workspace root. Commit the regenerated notebooks alongside the script changes.

## PR Description

When opening your PR, include:
- A summary of which APIs changed and how
- A list of all scripts you updated
- Confirmation that notebooks were regenerated
- A "Could not update" section for any scripts that still fail, with the error and your assessment of why
