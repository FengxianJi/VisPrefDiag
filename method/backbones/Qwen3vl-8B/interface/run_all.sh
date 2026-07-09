#!/bin/bash
# ==========================================================================
# Sequential launcher: runs all 4 evaluation scripts one by one on dual GPUs.
# Each script uses 2 GPU workers. Results are saved with resume support.
#
# Usage:
#   tmux new-session -d -s run_all 'bash /mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/interface/run_all.sh'
#   tmux attach -t run_all      # to monitor
# ==========================================================================

set -e

DIR="/mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/interface"
LOG="/mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/result/run_all.log"

echo "==========================================================" | tee "$LOG"
echo "  Qwen3-VL-8B Full Evaluation Pipeline"                      | tee -a "$LOG"
echo "  Started: $(date)"                                           | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"

# ---- 1. Single-agent_pair (ranking) ----
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] === 1/4 run_pair.py ===" | tee -a "$LOG"
bash "$DIR/run_dual.sh" run_pair.py Single-agent_pair Single-agent_pair.json 2>&1 | tee -a "$LOG"

# ---- 2. Single-agent-Budgeted (two-phase pricing) ----
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] === 2/4 run_budgeted.py ===" | tee -a "$LOG"
bash "$DIR/run_dual.sh" run_budgeted.py Single-agent-Budgeted Single-agent-Budgeted.json 2>&1 | tee -a "$LOG"

# ---- 3. Valuator_judge (multi-agent scoring) ----
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] === 3/4 run_judge.py ===" | tee -a "$LOG"
bash "$DIR/run_dual.sh" run_judge.py Valuator_judge Valuator_judge.json 2>&1 | tee -a "$LOG"

# ---- 4. Auction-arena (multi-agent auction) ----
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] === 4/4 run_auction.py ===" | tee -a "$LOG"
bash "$DIR/run_dual.sh" run_auction.py Auction-arena Auction-arena.json 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"
echo "  ALL DONE! $(date)"                                         | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"
