#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="$ROOT/experiments/qwen35_27b_migration"
KEY_FILE="${1:-$ROOT/qwenkey.txt}"
BASE_URL="${BASE_URL:-https://maas.qianwenaiapi.com/compatible-mode/v1}"
MODEL="${MODEL:-qwen3.5-27b}"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "Missing API key file: $KEY_FILE" >&2
  echo "Create it with one DashScope-compatible key per line." >&2
  exit 2
fi

mkdir -p "$PKG/logs"
for protocol in single_score pairwise evidence; do
  out="$PKG/results/qwen3.5-27b_${protocol}"
  log="$PKG/logs/${protocol}.log"
  pidfile="$PKG/logs/${protocol}.pid"
  nohup python3 "$PKG/run_qwen35_api.py" \
    --protocol "$protocol" \
    --model "$MODEL" \
    --key-file "$KEY_FILE" \
    --base-url "$BASE_URL" \
    --output-dir "$out" >"$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "$protocol started: PID $(cat "$pidfile")"
done

echo "All three protocols are detached from the terminal."
