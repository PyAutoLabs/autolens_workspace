#!/usr/bin/env python3
"""
Navigator / workspace catalogue checker.

Two independent checks, runnable locally or in CI:

  (a) Path existence (HARD FAIL)
      Scan the workspace's navigator / instruction files for repo-relative
      ``scripts/...`` and ``notebooks/...`` references and confirm each resolves
      on disk. A reference may be a literal path (must exist) or a glob
      containing ``*`` (must match at least one file). ``output/...`` and
      ``dataset/...`` references are ignored (runtime / data, not example code).
      In ``workspace_index.json`` only the authoritative ``path`` / ``notebook``
      fields are validated; ``cross_refs`` are best-effort docstring references
      and are not required to resolve.

  (b) Banner-comment lint (warn or fail, see ``--banners``)
      Scan ``scripts/**/*.py`` for banner-style separator comments — a comment
      line that is ``#`` followed only by a run (>= 4) of ``-``, ``=``, ``#`` or
      ``*``. The workspace style is ``\"\"\"__Section__\"\"\"`` docstrings, not
      ``# -----`` banners. The ``===`` underline beneath a docstring title is not
      a ``#`` comment and is never flagged.

This script is intentionally repo-agnostic: it hardcodes no repository name and
operates on the current working directory, so it ports to the galaxy / fit
workspaces unchanged. Run it from the workspace root::

    python .github/scripts/check_navigator.py
    python .github/scripts/check_navigator.py --banners=warn

An optional ignore file (default ``.navigator_check_ignore``) lists paths or
globs (one per line, ``#`` comments allowed) that are exempt from BOTH checks.
"""

import argparse
import fnmatch
import re
import sys
from pathlib import Path

# Files scanned for path references. ``scripts/**/README.md`` is expanded at
# runtime. Missing files are simply skipped (e.g. a workspace without one of
# these), so the same list ports across workspaces.
REFERENCE_FILES = [
    "AGENTS.md",
    "CLAUDE.md",
    ".github/copilot-instructions.md",
    "llms.txt",
    "llms-full.txt",
    "workspace_index.json",
]

# Path tokens we care about: example code / notebooks under the workspace. The
# negative lookbehind anchors the token at a path boundary so a longer path such
# as ".github/scripts/run_smoke.py" is NOT matched as "scripts/run_smoke.py".
_PATH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:scripts|notebooks)/[A-Za-z0-9_./*-]+"
)

# In workspace_index.json only the authoritative "path" / "notebook" fields are
# checked. "cross_refs" (and prose in "summary") are best-effort docstring
# references that may legitimately not resolve, so they are not validated here.
_JSON_AUTHORITATIVE_KEY_RE = re.compile(r'^\s*"(?:path|notebook)"\s*:')

# A banner separator comment: '#', optional whitespace, then ONLY a run (>=4) of
# the separator characters. Does not match '# ----- Section -----' (has text) or
# the docstring '===' underline (not a '#' comment).
_BANNER_RE = re.compile(r"^\s*#\s*[-=#*]{4,}\s*$")

DEFAULT_IGNORE_FILE = ".navigator_check_ignore"


def load_ignore(root: Path, ignore_file: str):
    """Load ignore patterns (paths / globs). Returns a list of POSIX strings."""
    path = root / ignore_file
    patterns = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def is_ignored(rel_path: str, patterns) -> bool:
    """True if ``rel_path`` matches an ignore pattern (literal or glob)."""
    return any(
        rel_path == pat or fnmatch.fnmatch(rel_path, pat) for pat in patterns
    )


def reference_files(root: Path):
    """Yield the existing reference files, including scripts/**/README.md."""
    for name in REFERENCE_FILES:
        candidate = root / name
        if candidate.exists():
            yield candidate
    for readme in sorted((root / "scripts").rglob("README.md")):
        yield readme


def extract_path_tokens(text: str, json_authoritative_only: bool = False):
    """
    Yield (line_number, token) for every scripts/.. or notebooks/.. token in the
    text. Works uniformly for markdown links and inline-code spans, since all we
    need is the token substring on its line.

    When ``json_authoritative_only`` is set (for workspace_index.json), only
    lines holding the authoritative ``"path"`` / ``"notebook"`` keys are scanned,
    so best-effort ``cross_refs`` and prose ``summary`` values are skipped.
    """
    for lineno, line in enumerate(text.splitlines(), start=1):
        if json_authoritative_only and not _JSON_AUTHORITATIVE_KEY_RE.match(line):
            continue
        for match in _PATH_TOKEN_RE.findall(line):
            # Strip trailing punctuation that commonly abuts a token in prose
            # (e.g. "see scripts/foo.py." or a token closing a code span).
            token = match.rstrip(".,;:)`\"'")
            yield lineno, token


def check_paths(root: Path, ignore_patterns):
    """Return a list of (file, line, token) misses. Empty list == all good."""
    misses = []
    for ref in reference_files(root):
        rel_ref = ref.relative_to(root).as_posix()
        try:
            text = ref.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"WARNING: could not read {rel_ref}: {exc}", file=sys.stderr)
            continue
        json_only = ref.suffix == ".json"
        for lineno, token in extract_path_tokens(text, json_only):
            if is_ignored(token, ignore_patterns):
                continue
            if "*" in token:
                if not any(root.glob(token)):
                    misses.append((rel_ref, lineno, token))
            else:
                if not (root / token).exists():
                    misses.append((rel_ref, lineno, token))
    return misses


def check_banners(root: Path, ignore_patterns):
    """Return a list of (file, line) banner-comment hits."""
    hits = []
    for script in sorted((root / "scripts").rglob("*.py")):
        rel = script.relative_to(root).as_posix()
        if is_ignored(rel, ignore_patterns):
            continue
        try:
            lines = script.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"WARNING: could not read {rel}: {exc}", file=sys.stderr)
            continue
        for lineno, line in enumerate(lines, start=1):
            if _BANNER_RE.match(line):
                hits.append((rel, lineno))
    return hits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        help="Workspace root to check (default: current directory).",
    )
    parser.add_argument(
        "--banners",
        choices=["warn", "fail"],
        default="fail",
        help="Banner-comment lint mode: 'fail' (nonzero exit) or 'warn'.",
    )
    parser.add_argument(
        "--ignore-file",
        default=DEFAULT_IGNORE_FILE,
        help=f"Ignore file of paths/globs (default: {DEFAULT_IGNORE_FILE}).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    ignore_patterns = load_ignore(root, args.ignore_file)

    # (a) Path existence — always a hard failure.
    path_misses = check_paths(root, ignore_patterns)
    if path_misses:
        print(f"Path check: {len(path_misses)} missing reference(s):")
        for ref, lineno, token in path_misses:
            print(f"  {ref}:{lineno} -> missing path: {token}")
    else:
        print("Path check: OK — every scripts/ and notebooks/ reference resolves.")

    # (b) Banner lint — warn or fail.
    banner_hits = check_banners(root, ignore_patterns)
    if banner_hits:
        print(f"Banner lint: {len(banner_hits)} banner-style comment(s):")
        for ref, lineno in banner_hits:
            print(f"  {ref}:{lineno}")
        print(
            "  Use triple-quoted \"\"\"__Section__\"\"\" docstrings, not # ----- banners."
        )
    else:
        print("Banner lint: OK — no banner-style comments found.")

    failed = bool(path_misses) or (args.banners == "fail" and bool(banner_hits))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
