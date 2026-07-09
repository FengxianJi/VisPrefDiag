#!/bin/bash
# Launch dual-GPU price prediction workers
# Each worker is a separate process with its own CUDA_VISIBLE_DEVICES
# Results are merged into a single JSON file via file locking

cd /mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/interface

LOG_DIR="/mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/result"
RESULT_DIR="$LOG_DIR/Single-agent_price"
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "Starting dual-GPU price prediction"
echo "=========================================="

# Launch GPU 0 worker
/usr/bin/python3 -u run_price.py --gpu_id 0 --num_gpus 2 --batch_size 6 \
    > "$LOG_DIR/run_price_gpu0.log" 2>&1 &
PID0=$!
echo "GPU 0 worker started (PID=$PID0)"

# Launch GPU 1 worker
/usr/bin/python3 -u run_price.py --gpu_id 1 --num_gpus 2 --batch_size 6 \
    > "$LOG_DIR/run_price_gpu1.log" 2>&1 &
PID1=$!
echo "GPU 1 worker started (PID=$PID1)"

echo "Waiting for both workers to finish..."
echo "  Monitor GPU0: tail -f $LOG_DIR/run_price_gpu0.log"
echo "  Monitor GPU1: tail -f $LOG_DIR/run_price_gpu1.log"

# Wait for both workers
wait $PID0
STATUS0=$?
echo "GPU 0 worker finished (exit=$STATUS0)"

wait $PID1
STATUS1=$?
echo "GPU 1 worker finished (exit=$STATUS1)"

# Calculate metrics
echo ""
echo "=========================================="
echo "Calculating metrics..."
echo "=========================================="
/usr/bin/python3 -u -c "
import sys, json
sys.path.insert(0, '/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main')
import calculate_matrix

result_file = '$RESULT_DIR/Single-agent_price.json'
with open(result_file) as f:
    data = json.load(f)

metrics = {}
for ds_name, episodes in data.get('datasets', {}).items():
    if episodes:
        print(f'  {ds_name}: {len(episodes)} episodes')
        metrics[ds_name] = calculate_matrix.calculate_all_metrics(episodes)

matrix_file = '$RESULT_DIR/Single-agent_price_matrix.json'
with open(matrix_file, 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f'Metrics saved: {matrix_file}')
"

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "=========================================="
