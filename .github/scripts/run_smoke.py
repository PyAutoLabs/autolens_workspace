"""
Run the workspace smoke test suite: the scripts listed in `smoke_tests.txt` and
the notebooks listed in `smoke_notebooks.txt`.

Nothing about discovery, exclusion, environment resolution, per-entry timeouts,
notebook execution or reporting is implemented here. This is a thin shim over
PyAutoHands' `autohands/run_python.py` and `autohands/run.py` — the same entry
points PyAutoHeart's workspace-validation uses — so the PR gate and the
validation runner cannot drift apart.

This file used to be a 356-line copy of that machinery, one of ten across the
workspace repos. Each of the last three fixes to it — the env-resolver fork
(PyAutoHands#185), the per-script timeout and process-group kill
(PyAutoHands#226/#227), the jupyter guard — had to be swept across every copy by
hand, while the HowTo repos needed none of them precisely because they hold no
logic.

The notebook leg is preserved in full, promoted into PyAutoHands#261 rather than
reimplemented here:

  * `--no-write-back` executes a throwaway copy, so a smoke run never modifies
    the committed notebooks. (The shared runner pins the kernel's cwd to the
    repo root already, which is what this file's staged-copy-at-root trick
    existed to achieve — so that trick is superseded, not ported.)
  * `--retry-from scripts` regenerates a failing notebook from its source `.py`
    and retries ONCE — the stale-notebook recovery. A TIMEOUT is never retried,
    a clean `sys.exit(0)` skip guard is already a PASS and never reaches it, and
    the retry's verdict replaces the first attempt's so one notebook yields one
    result.
  * A missing notebook toolchain is one FAIL and the run continues: the shared
    runner invokes `sys.executable run_notebook.py`, never a bare `jupyter`, so
    the abort-with-no-summary failure mode is structurally absent.

`config/build/no_run.yaml` is deliberately NOT applied to either allowlist. It
is policy for the release mega-run and notebook generation; the smoke lists are
policy for this gate, and an entry legitimately appears in both
(PyAutoHands#262).

`--report-dir` is REQUIRED, not cosmetic. The shared runners only propagate
failures (`sys.exit(1)`) when a report was built; without it the suite runs to
completion and always exits 0 — a vacuously green gate.

Both legs always run, and the exit code is the worst of the two: a failing
notebook must not be masked by passing scripts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = "autolens_workspace"

# CI puts PyAutoHands/autohands on PYTHONPATH (PyAutoHeart's reusable
# smoke-tests.yml clones it alongside the dependency chain); for local runs,
# fall back to the sibling checkout.
try:
    import build_util
except ImportError:  # pragma: no cover - local-run fallback
    sys.path.insert(0, str(WORKSPACE.parent / "PyAutoHands" / "autohands"))
    import build_util

AUTOHANDS = Path(build_util.__file__).resolve().parent

SCRIPT_LIST = WORKSPACE / "smoke_tests.txt"
NOTEBOOK_LIST = WORKSPACE / "smoke_notebooks.txt"
REPORT_DIR = WORKSPACE / "test-results"


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(AUTOHANDS), env.get("PYTHONPATH", "")) if p
    )
    return env


def _run(argv: list[str]) -> int:
    # The shared runners resolve config/build/ relative to the cwd.
    return subprocess.run(argv, cwd=str(WORKSPACE), env=_env()).returncode


def main() -> int:
    rc = 0

    if SCRIPT_LIST.exists():
        rc |= _run([
            sys.executable,
            str(AUTOHANDS / "run_python.py"),
            PROJECT,
            "scripts",
            "--list", str(SCRIPT_LIST),
            "--report-dir", str(REPORT_DIR),
        ])
    else:
        print(f"No {SCRIPT_LIST.name}; script leg skipped.")

    if NOTEBOOK_LIST.exists():
        rc |= _run([
            sys.executable,
            str(AUTOHANDS / "run.py"),
            PROJECT,
            "notebooks",
            "--list", str(NOTEBOOK_LIST),
            "--no-write-back",
            "--retry-from", "scripts",
            "--report-dir", str(REPORT_DIR),
        ])
    else:
        print(f"No {NOTEBOOK_LIST.name}; notebook leg skipped.")

    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
