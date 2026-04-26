#!/usr/bin/env bash
# Compare script sizes vs the workspace .script_sizes.json snapshot.
# Warns and exits non-zero when any script has shrunk >50% — a heuristic for
# accidental whole-file truncation by a bulk edit (see CLAUDE.md "Bulk-edit
# safety").
#
# Usage:
#   scripts/check_sizes.sh             # check working tree against snapshot
#   scripts/check_sizes.sh --update    # rewrite snapshot from current scripts/
#
# Override:
#   ALLOW_SHRINK=1 scripts/check_sizes.sh    # accept shrinkage, exit 0

set -e
WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SNAPSHOT="$WS_ROOT/.script_sizes.json"

if [ "${1:-}" = "--update" ]; then
  cd "$WS_ROOT"
  python3 - <<'PY' > "$SNAPSHOT"
import json, os
sizes = {}
for root, _, files in os.walk("scripts"):
    for f in sorted(files):
        if f.endswith(".py"):
            p = os.path.join(root, f)
            sizes[p] = os.path.getsize(p)
print(json.dumps(sizes, indent=2, sort_keys=True))
PY
  echo "Updated $SNAPSHOT"
  exit 0
fi

if [ ! -f "$SNAPSHOT" ]; then
  echo "No snapshot at $SNAPSHOT — generate one with: scripts/check_sizes.sh --update" >&2
  exit 1
fi

cd "$WS_ROOT"
ALLOW_SHRINK="${ALLOW_SHRINK:-}" python3 - <<'PY'
import json, os, sys
with open(".script_sizes.json") as f:
    snapshot = json.load(f)
shrunk = []
for path, prev in snapshot.items():
    if not os.path.isfile(path):
        continue
    cur = os.path.getsize(path)
    if prev > 200 and cur < prev * 0.5:
        pct = 100 * (prev - cur) // prev
        shrunk.append((pct, path, prev, cur))
shrunk.sort(reverse=True)
if shrunk:
    print("WARNING: scripts shrunk by >50% since the snapshot:", file=sys.stderr)
    for pct, p, prev, cur in shrunk:
        print(f"  {pct:3d}%  {p}  {prev}b -> {cur}b", file=sys.stderr)
    print("", file=sys.stderr)
    print("This may indicate accidental whole-file truncation (see CLAUDE.md", file=sys.stderr)
    print("Bulk-edit safety). If the shrinkage is intentional, re-run with", file=sys.stderr)
    print("ALLOW_SHRINK=1 and refresh via: scripts/check_sizes.sh --update", file=sys.stderr)
    sys.exit(0 if os.environ.get("ALLOW_SHRINK") else 1)
print("OK: all scripts within size tolerance.")
PY
