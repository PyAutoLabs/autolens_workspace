"""
Run the workspace smoke test suite.

Reads `smoke_tests.txt` (Python scripts) and `smoke_notebooks.txt`
(Jupyter notebooks) from the workspace root, plus
`config/build/env_vars.yaml` for per-entry env var overrides, then
runs each listed entry with the appropriate environment. Continues
through failures and exits non-zero if any entry failed.

Notebook execution uses `jupyter nbconvert --to notebook --execute`.
On failure the runner regenerates the single failing notebook from its
source `.py` script via PyAutoBuild's `py_to_notebook` and retries
once — this catches stale notebooks where the script has moved on but
the on-disk `.ipynb` wasn't refreshed by `/pre_build`'s
`generate.py`. Whole-workspace regeneration stays the responsibility
of `generate.py`; smoke only regenerates the single notebook in front
of it so the recovery is cheap.

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

import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
SMOKE_FILE = WORKSPACE / "smoke_tests.txt"
NOTEBOOK_FILE = WORKSPACE / "smoke_notebooks.txt"
ENV_VARS_FILE = WORKSPACE / "config" / "build" / "env_vars.yaml"
SCRIPTS_DIR = WORKSPACE / "scripts"
NOTEBOOKS_DIR = WORKSPACE / "notebooks"


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


def load_env_config() -> dict:
    if not ENV_VARS_FILE.exists():
        return {"defaults": {}, "overrides": []}
    return yaml.safe_load(ENV_VARS_FILE.read_text()) or {}


def pattern_matches(pattern: str, rel_path: str) -> bool:
    if "/" in pattern:
        return pattern in rel_path
    return Path(rel_path).stem == pattern


def build_env(rel_path: str, cfg: dict) -> dict:
    env = os.environ.copy()
    defaults = cfg.get("defaults") or {}
    env.update({k: str(v) for k, v in defaults.items()})
    for override in cfg.get("overrides") or []:
        if pattern_matches(override["pattern"], rel_path):
            for key in override.get("unset", []):
                env.pop(key, None)
            for key, val in (override.get("set") or {}).items():
                env[key] = str(val)
    return env


def run_script(script_rel: str, cfg: dict) -> tuple[str, int, float, str]:
    env = build_env(script_rel, cfg)
    script_path = SCRIPTS_DIR / script_rel
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(WORKSPACE),
        env=env,
        capture_output=True,
        text=True,
    )
    return script_rel, result.returncode, time.time() - t0, result.stdout + result.stderr


def execute_notebook(nb_path: Path, env: dict) -> tuple[int, str]:
    # Write the executed copy to a throwaway path so the on-disk notebook
    # under notebooks/ is never modified — checked-in notebooks stay clean.
    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_nb_"))
    try:
        result = subprocess.run(
            [
                "jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--output-dir", str(tmp_dir),
                "--output", nb_path.name,
                str(nb_path),
            ],
            cwd=str(WORKSPACE),
            env=env,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result.returncode, result.stdout + result.stderr


def regenerate_notebook(nb_rel: str) -> Path:
    """Regenerate a notebook from its source `.py` into a temp dir.

    The regenerated copy lives in /tmp; the on-disk `notebooks/` tree is
    never modified, so a smoke run leaves the worktree clean.
    """
    from build_util import py_to_notebook  # PyAutoBuild/autobuild on PYTHONPATH

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


def run_notebook(nb_rel: str, cfg: dict) -> tuple[str, int, float, str]:
    env = build_env(nb_rel, cfg)
    nb_path = NOTEBOOKS_DIR / nb_rel
    t0 = time.time()

    if not nb_path.exists():
        return nb_rel, 1, 0.0, f"Notebook not found: {nb_path}\n"

    rc, output = execute_notebook(nb_path, env)
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
    cfg = load_env_config()
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
        print(f"\nRunning {len(notebooks)} notebook smoke test(s) from {NOTEBOOK_FILE.name}\n")
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
