"""
Run the workspace smoke test suite.

Reads `smoke_tests.txt` (Python scripts) and `smoke_notebooks.txt`
(Jupyter notebooks) from the workspace root, plus
`config/build/profile_smoke.yaml` for per-entry env var overrides, then
runs each listed entry with the appropriate environment. Continues
through failures and exits non-zero if any entry failed.

Notebook execution uses `jupyter nbconvert --to notebook --execute`.
If `jupyter` is not installed the notebook entries are reported as
per-entry failures (exit 127) rather than aborting the run: the runner's
contract is to continue through failures and always end with the summary
line. CI images always ship jupyter, so this only bites a local sweep --
where an abort would print a raw traceback and silently discard coverage
of every remaining entry.
On failure the runner regenerates the single failing notebook from its
source `.py` script via PyAutoHands's `py_to_notebook` and retries
once — this catches stale notebooks where the script has moved on but
the on-disk `.ipynb` wasn't refreshed by `/pre_build`'s
`generate.py`. Whole-workspace regeneration stays the responsibility
of `generate.py`; smoke only regenerates the single notebook in front
of it so the recovery is cheap.

The env resolution itself is NOT implemented here: it is PyAutoHands's
`autohands/env_config.py`, imported below. This file used to carry a copy, and
the copy had already drifted (its `load_env_config` hardcoded
`config/build/profile_smoke.yaml`, so the PR gate was structurally unable to read
the release profile — the seed incident's failure mode 4/7). One resolver
means the PR gate and the release runner cannot disagree about what a script's
environment is. See PyAutoHands docs/env_profile_redesign.md §5 (#161 step 2).

Mirrors the logic of the `/smoke-test` skill so CI and local runs stay
in sync.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
SMOKE_FILE = WORKSPACE / "smoke_tests.txt"
NOTEBOOK_FILE = WORKSPACE / "smoke_notebooks.txt"
ENV_VARS_FILE = WORKSPACE / "config" / "build" / "profile_smoke.yaml"
SCRIPTS_DIR = WORKSPACE / "scripts"
NOTEBOOKS_DIR = WORKSPACE / "notebooks"

# Exit code reported for a notebook entry when the `jupyter` executable is
# absent, mirroring the shell's "command not found". Distinct from any code
# nbconvert itself returns, so run_notebook can recognise the case.
JUPYTER_MISSING_RC = 127
JUPYTER_MISSING_MSG = (
    "jupyter not found on PATH: cannot execute notebooks.\n"
    "Install it with `pip install jupyter` to cover the notebook entries; "
    "the script entries above are unaffected.\n"
)

# CI puts PyAutoHands/autohands on PYTHONPATH (PyAutoHeart's reusable
# smoke-tests.yml clones it alongside the dependency chain); for local runs,
# fall back to the sibling checkout.
try:
    from env_config import build_env_for_script, load_env_config
except ImportError:  # pragma: no cover - local-run fallback
    sys.path.insert(0, str(WORKSPACE.parent / "PyAutoHands" / "autohands"))
    from env_config import build_env_for_script, load_env_config

from build_util import is_clean_skip_exit  # PyAutoHands/autohands on PYTHONPATH


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def load_cfg() -> dict | None:
    """Parsed env profile, or None when the workspace has none.

    None flows through build_env_for_script -> None -> subprocess inherits the
    parent environment, which is what the old local copy's empty-config path
    did by hand.
    """
    if not ENV_VARS_FILE.exists():
        return None
    return load_env_config(ENV_VARS_FILE)


def run_script(script_rel: str, cfg: dict | None) -> tuple[str, int, float, str]:
    env = build_env_for_script(Path(script_rel), cfg)
    script_path = SCRIPTS_DIR / script_rel
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(WORKSPACE),
        env=env,
        capture_output=True,
        text=True,
    )
    return (
        script_rel,
        result.returncode,
        time.time() - t0,
        result.stdout + result.stderr,
    )


def execute_notebook(nb_path: Path, env: dict) -> tuple[int, str]:
    # Write the executed copy to a throwaway path so the on-disk notebook
    # under notebooks/ is never modified — checked-in notebooks stay clean.
    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_nb_"))
    # `jupyter nbconvert --execute` runs the kernel in the *notebook's own
    # directory*, but the workspace convention (and the script smoke, which runs
    # with cwd=WORKSPACE) is that relative `dataset/` paths resolve from the repo
    # root. Stage a temporary copy of the notebook at the workspace root so the
    # kernel's working directory is the root and root-relative paths resolve.
    staged = WORKSPACE / f".smoke_run_{os.getpid()}_{nb_path.name}"
    shutil.copyfile(nb_path, staged)
    try:
        try:
            result = subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--output-dir",
                    str(tmp_dir),
                    "--output",
                    nb_path.name,
                    str(staged),
                ],
                cwd=str(WORKSPACE),
                env=env,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            # `jupyter` is not installed. Report a per-entry failure instead of
            # letting the exception escape main(), which would abort the run
            # with a raw traceback, print no summary line, and leave every
            # remaining entry silently uncovered.
            return JUPYTER_MISSING_RC, JUPYTER_MISSING_MSG
    finally:
        staged.unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result.returncode, result.stdout + result.stderr


def regenerate_notebook(nb_rel: str) -> Path:
    """Regenerate a notebook from its source `.py` into a temp dir.

    The regenerated copy lives in /tmp; the on-disk `notebooks/` tree is
    never modified, so a smoke run leaves the worktree clean.
    """
    from build_util import py_to_notebook  # PyAutoHands/autohands on PYTHONPATH

    script_path = SCRIPTS_DIR / Path(nb_rel).with_suffix(".py")
    if not script_path.exists():
        raise FileNotFoundError(f"No source script at {script_path}")
    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_regen_"))
    tmp_script = tmp_dir / script_path.name
    shutil.copy(script_path, tmp_script)
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_dir)
        generated = py_to_notebook(tmp_script)
    finally:
        os.chdir(old_cwd)
    return generated


def run_notebook(nb_rel: str, cfg: dict | None) -> tuple[str, int, float, str]:
    env = build_env_for_script(Path(nb_rel), cfg)
    nb_path = NOTEBOOKS_DIR / nb_rel
    t0 = time.time()

    if not nb_path.exists():
        return nb_rel, 1, 0.0, f"Notebook not found: {nb_path}\n"

    rc, output = execute_notebook(nb_path, env)
    if rc == JUPYTER_MISSING_RC:
        # No jupyter, so regenerating the notebook and retrying cannot help.
        # Checked before the skip guard so a missing tool is never a PASS.
        return nb_rel, rc, time.time() - t0, output
    if rc != 0 and is_clean_skip_exit(output):
        # Optional-dependency skip guard (`sys.exit(0)`): a clean exit 0 as a
        # `.py` script, so the notebook form is a PASS too (PyAutoHands#198).
        rc = 0
    if rc == 0:
        return nb_rel, 0, time.time() - t0, output

    print("  notebook failed; regenerating from source script and retrying...")
    try:
        nb_path = regenerate_notebook(nb_rel)
    except Exception as exc:
        output += f"\n[regenerate_notebook] {exc}\n"
        return nb_rel, rc, time.time() - t0, output

    rc2, output2 = execute_notebook(nb_path, env)
    output += "\n--- regenerated from script and retried ---\n" + output2
    return nb_rel, rc2, time.time() - t0, output


def main() -> int:
    cfg = load_cfg()
    scripts = load_lines(SMOKE_FILE)
    notebooks = load_lines(NOTEBOOK_FILE)

    if not scripts and not notebooks:
        print("No smoke tests listed.")
        return 0

    failures: list[tuple[str, int, str]] = []
    total = 0

    if scripts:
        print(f"Running {len(scripts)} script smoke test(s) from {SMOKE_FILE.name}\n")
        for rel in scripts:
            print(f"::group::script: {rel}")
            name, rc, elapsed, output = run_script(rel, cfg)
            print(output, end="")
            status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
            print(f"\n[{status}] {name} — {elapsed:.1f}s")
            print("::endgroup::")
            total += 1
            if rc != 0:
                failures.append((f"script: {name}", rc, output))

    if notebooks:
        print(
            f"\nRunning {len(notebooks)} notebook smoke test(s) from {NOTEBOOK_FILE.name}\n"
        )
        for rel in notebooks:
            print(f"::group::notebook: {rel}")
            name, rc, elapsed, output = run_notebook(rel, cfg)
            print(output, end="")
            status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
            print(f"\n[{status}] {name} — {elapsed:.1f}s")
            print("::endgroup::")
            total += 1
            if rc != 0:
                failures.append((f"notebook: {name}", rc, output))

    passed = total - len(failures)
    print(f"\n=== Smoke test summary: {passed}/{total} passed ===")
    for name, rc, _ in failures:
        print(f"  FAIL  {name}  (exit {rc})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
