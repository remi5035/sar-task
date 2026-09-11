#!/usr/bin/env bash
# Benchmarks the packaged champion on the practice seed set and prints its score.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
SEEDS="$HERE/practice_seeds.json"
OUT="${1:-$REPO/baseline_results.json}"
WORKERS="${SWARM_WORKERS:-4}"

if ! command -v swarm >/dev/null 2>&1; then
  echo "The swarm CLI is not on PATH. Activate the environment first, then re-run." >&2
  exit 1
fi

cat <<EOF
Benchmarking the champion.

  seeds    $SEEDS  (1,100, full set)
  workers  $WORKERS
  results  $OUT

This is a multi-hour run. To iterate faster, trim the seed lists as shown in NOTES.md
and pass your smaller file as the seed file instead.

EOF

swarm benchmark \
  --model "$HERE/champion/submission.zip" \
  --family-id cf_search_and_rescue \
  --seed-file "$SEEDS" \
  --workers "$WORKERS" \
  --relax-timeouts \
  --rpc-verbosity low \
  --summary-json-out "$OUT"

echo
echo "Per-seed results written to $OUT"
echo "Seeds that did not succeed:"
swarm visualize --summary-json "$OUT" --failed || true
