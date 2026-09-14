#!/usr/bin/env bash
# Benchmarks the packaged champion and writes per-seed results.
#
# Usage:
#   ./TASK/run_baseline.sh                      full 1,100-seed practice set
#   ./TASK/run_baseline.sh quick_seeds.json     a subset, while iterating
#   ./TASK/run_baseline.sh <seeds> <output>     explicit output path
#
# Workers default to 4 and can be set with SWARM_WORKERS.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
SEEDS="${1:-$HERE/practice_seeds.json}"
OUT="${2:-$REPO/baseline_results.json}"
WORKERS="${SWARM_WORKERS:-4}"
MODEL="$HERE/champion/submission.zip"

if ! command -v swarm >/dev/null 2>&1; then
  echo "swarm is not on PATH. Activate the environment and re-run." >&2
  exit 1
fi

for f in "$SEEDS" "$MODEL"; do
  [ -f "$f" ] || { echo "Not found: $f" >&2; exit 1; }
done

COUNT="$(python3 -c 'import json,sys; print(sum(len(v) for v in json.load(open(sys.argv[1]))["type_seeds"].values()))' "$SEEDS")"

cat <<EOF
model    $MODEL
seeds    $SEEDS ($COUNT)
workers  $WORKERS
results  $OUT

EOF

swarm benchmark \
  --model "$MODEL" \
  --family-id cf_search_and_rescue \
  --seed-file "$SEEDS" \
  --workers "$WORKERS" \
  --relax-timeouts \
  --rpc-verbosity low \
  --summary-json-out "$OUT"

echo
echo "Per-seed results: $OUT"
echo
echo "Seeds that did not succeed:"
swarm visualize --summary-json "$OUT" --failed || true
