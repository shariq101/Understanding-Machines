#!/usr/bin/env bash
# Fetch the measured microscope JSON and image bank into the site.
#   bash kaggle/microscope/fetch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
tmp=$(mktemp -d)
kaggle kernels output shariq101/understanding-machine-microscope -p "$tmp"
src=$(find "$tmp" -type d -name microscope | head -1)
[ -n "$src" ] || { echo "no microscope output found" >&2; exit 1; }
rm -rf visualizations/microscope
cp -R "$src" visualizations/microscope
python3 - <<'PY'
import json
from pathlib import Path
p = Path("visualizations/microscope/manifest.json")
d = json.loads(p.read_text())
print("layers", [(x["id"], x["selected"], x["channels"], x["spatial"]) for x in d["layers"]])
print("images", sum(1 for _ in Path("visualizations/microscope/images").rglob("*.jpg")))
print("bytes", sum(f.stat().st_size for f in Path("visualizations/microscope").rglob("*") if f.is_file()))
PY
rm -rf "$tmp"
