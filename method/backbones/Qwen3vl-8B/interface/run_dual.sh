#!/bin/bash
# ==========================================================================
# Run a single script on dual GPUs, wait for completion, then compute metrics.
# Usage: bash run_dual.sh <script.py> <result_dir> <result_json> [extra_args...]
# ==========================================================================
set -e

SCRIPT="$1"
RESULT_DIR_NAME="$2"
RESULT_JSON="$3"
shift 3
EXTRA_ARGS="$@"

INTERFACE_DIR="/mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/interface"
LOG_DIR="/mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/result"
RESULT_DIR="$LOG_DIR/$RESULT_DIR_NAME"
EXPERIMENT_DIR="/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main"

mkdir -p "$RESULT_DIR"

SCRIPT_BASE=$(basename "$SCRIPT" .py)
LOG0="$LOG_DIR/${SCRIPT_BASE}_gpu0.log"
LOG1="$LOG_DIR/${SCRIPT_BASE}_gpu1.log"

echo "=========================================="
echo "  $SCRIPT_BASE — Dual GPU"
echo "  Result: $RESULT_DIR/$RESULT_JSON"
echo "=========================================="

cd "$INTERFACE_DIR"

# Launch GPU 0
/usr/bin/python3 -u "$SCRIPT" --gpu_id 0 --num_gpus 2 $EXTRA_ARGS > "$LOG0" 2>&1 &
PID0=$!
echo "  GPU0 started (PID=$PID0) -> $LOG0"

# Launch GPU 1
/usr/bin/python3 -u "$SCRIPT" --gpu_id 1 --num_gpus 2 $EXTRA_ARGS > "$LOG1" 2>&1 &
PID1=$!
echo "  GPU1 started (PID=$PID1) -> $LOG1"

# Wait for both
wait $PID0
S0=$?
echo "  GPU0 finished (exit=$S0)"

wait $PID1
S1=$?
echo "  GPU1 finished (exit=$S1)"

# Sort results by group_id and compute metrics
echo "  Computing metrics..."
/usr/bin/python3 -u -c "
import sys, json
sys.path.insert(0, '$EXPERIMENT_DIR')
import calculate_matrix

result_file = '$RESULT_DIR/$RESULT_JSON'
with open(result_file) as f:
    data = json.load(f)

# Sort each dataset's results by group_id
for ds_name in data.get('datasets', {}):
    data['datasets'][ds_name].sort(key=lambda r: r['group_id'])
with open(result_file, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

metrics = {}
total_groups = 0
for ds_name, episodes in data.get('datasets', {}).items():
    success = [e for e in episodes if e.get('success')]
    total_groups += len(success)
    if success:
        metrics[ds_name] = calculate_matrix.calculate_all_metrics(success)

matrix_name = result_file.replace('.json', '_matrix.json')
with open(matrix_name, 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f'  Metrics saved ({total_groups} groups): {matrix_name}')
"

echo "  DONE: $SCRIPT_BASE"
echo ""
