#!/bin/bash
# Wrapper script to run run_price.py with clean CUDA environment
cd /mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/interface

# Ensure clean CUDA environment
unset LD_PRELOAD
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HOME="/home/fengxian.ji"

exec /usr/bin/python3 -u run_price.py --batch_size 6 2>&1 | tee /mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/result/run_price.log
