#!/usr/bin/env python3
"""
Regenerate the workspace catalogue (Phase 2 only — no notebook rebuild).

This is the lightweight entrypoint the ``navigator_check`` staleness job uses:
it calls PyAutoBuild's ``navigator.write_catalogue`` for the current workspace
checkout, producing ``llms-full.txt`` and ``workspace_index.json`` *without*
running the full notebook generation and *without* needing the science stack
(only ``pyyaml``).

Repo-agnostic — no repository name is hardcoded. Configuration is via
environment variables so porting to another workspace is a one-line change in
the workflow:

  NAVIGATOR_PROJECT   the generator project name (default: ``autolens``).
  PYAUTOBUILD_DIR     path to PyAutoBuild's ``autobuild`` package directory
                      (default: ``../PyAutoBuild/autobuild`` relative to CWD,
                      matching the local sibling-clone layout). In CI this is
                      set to the checked-out PyAutoBuild path.

Run from the workspace root::

    python .github/scripts/regenerate_navigator.py
"""

import os
import sys
from pathlib import Path


def main():
    project = os.environ.get("NAVIGATOR_PROJECT", "autolens")

    pyautobuild_dir = os.environ.get(
        "PYAUTOBUILD_DIR", str(Path.cwd().parent / "PyAutoBuild" / "autobuild")
    )
    pyautobuild_dir = str(Path(pyautobuild_dir).resolve())
    if not Path(pyautobuild_dir).is_dir():
        sys.exit(
            f"PyAutoBuild autobuild dir not found: {pyautobuild_dir}\n"
            "Set PYAUTOBUILD_DIR to the checked-out PyAutoBuild/autobuild path."
        )

    sys.path.insert(0, pyautobuild_dir)

    # navigator imports generate.py lazily, whose module-level argparse expects
    # a project positional; supply it via argv so the import resolves cleanly.
    sys.argv = ["generate.py", project]

    import navigator

    navigator.write_catalogue(Path.cwd(), project)


if __name__ == "__main__":
    main()
