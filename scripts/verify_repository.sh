#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

for script in scripts/*.sh; do
  bash -n "$script"
done

python3 -m json.tool config/provenance.json >/dev/null
python3 -m json.tool results/reference/run1_700s_v128/run.summary.json >/dev/null
python3 -m json.tool results/reference/run1_700s_v128/run.analysis.json >/dev/null
python3 -m json.tool results/reference/run1_700s_v128/planar_metrics >/dev/null

docker compose -f compose.run1-v128.yml config --quiet

if rg -n -F "$HOME/" \
  --glob '!config/provenance.json' --glob '!results/**' .; then
  echo "host-specific absolute path found" >&2
  exit 20
fi

if find . -type f -size +100M -print -quit | grep -q .; then
  echo "repository contains a file larger than 100 MiB" >&2
  exit 21
fi

if find src -type d \( -name .git -o -name build -o -name install \) -print -quit | grep -q .; then
  echo "nested repository or build directory found under src" >&2
  exit 22
fi

sha256sum -c config/critical-files.sha256

test_modules=()
for test_file in src/racer_integration/test_*.py; do
  # This monolithic legacy guard also asserts files from removed wheeled/UAV
  # stacks and behavior added after v128. It is intentionally not a release
  # test for this focused repository.
  if [ "$(basename "$test_file")" = "test_racer_no_progress_guard.py" ]; then
    continue
  fi
  test_modules+=("$(basename "$test_file" .py)")
done
PYTHONPATH="src/racer_integration" python3 -m unittest "${test_modules[@]}"
python3 -m unittest discover -s tests -p 'test_*.py'

echo "repository verification passed"
