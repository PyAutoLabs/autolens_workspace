#!/usr/bin/env bash
# Truncation guard: compare each CHANGED scripts/**/*.py against its size in a
# git base revision. Warns and exits non-zero when any changed script has
# shrunk >50% — a heuristic for accidental whole-file truncation by a bulk
# edit (see AGENTS.md "Bulk-edit safety"). Git itself is the baseline, so
# there is no snapshot file to refresh and nothing to go stale.
#
# Usage:
#   scripts/check_sizes.sh               # working tree vs HEAD
#   scripts/check_sizes.sh --base <ref>  # working tree vs <ref> (CI uses the
#                                        # PR merge-base)
#
# Override:
#   ALLOW_SHRINK=1 scripts/check_sizes.sh    # accept shrinkage, exit 0
#
# Exit codes: 0 = OK (or ALLOW_SHRINK), 1 = truncation suspected,
#             2 = base revision unresolvable (fail closed, never a silent pass).

set -e
WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS_ROOT"

BASE="HEAD"
if [ "${1:-}" = "--base" ]; then
  if [ -z "${2:-}" ]; then
    echo "ERROR: --base requires a git revision argument" >&2
    exit 2
  fi
  BASE="$2"
elif [ -n "${1:-}" ]; then
  echo "ERROR: unknown argument '$1' (usage: scripts/check_sizes.sh [--base <ref>])" >&2
  exit 2
fi

if ! BASE_SHA="$(git rev-parse --verify --quiet "${BASE}^{commit}")"; then
  echo "ERROR: cannot resolve base revision '$BASE' — refusing to report OK without a comparison." >&2
  exit 2
fi

BASE_SHA="$BASE_SHA" ALLOW_SHRINK="${ALLOW_SHRINK:-}" python3 - <<'PY'
import os
import subprocess
import sys

base = os.environ["BASE_SHA"]
changed = subprocess.run(
    ["git", "diff", "--name-only", base, "--", "scripts"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()

shrunk = []
for path in changed:
    if not path.endswith(".py"):
        continue
    if not os.path.isfile(path):
        continue  # deleted on disk — a deletion is explicit, not truncation
    proc = subprocess.run(
        ["git", "cat-file", "-s", f"{base}:{path}"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        continue  # absent in base — a new file has nothing to truncate
    prev = int(proc.stdout.strip())
    cur = os.path.getsize(path)
    if prev > 200 and cur < prev * 0.5:
        pct = 100 * (prev - cur) // prev
        shrunk.append((pct, path, prev, cur))

shrunk.sort(reverse=True)
if shrunk:
    print(f"WARNING: scripts shrunk by >50% vs base {base[:12]}:", file=sys.stderr)
    for pct, p, prev, cur in shrunk:
        print(f"  {pct:3d}%  {p}  {prev}b -> {cur}b", file=sys.stderr)
    print("", file=sys.stderr)
    print("This may indicate accidental whole-file truncation (see AGENTS.md", file=sys.stderr)
    print("Bulk-edit safety). If the shrinkage is intentional, re-run with", file=sys.stderr)
    print("ALLOW_SHRINK=1 to accept it.", file=sys.stderr)
    sys.exit(0 if os.environ.get("ALLOW_SHRINK") else 1)
print(f"OK: no changed script shrank >50% vs base {base[:12]}.")
PY
