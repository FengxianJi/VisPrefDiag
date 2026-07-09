#!/bin/bash
# 使用增强重试策略运行评估

cd /mnt/data/user/fengxian.ji/AUCTION/method/Single-agent-Budgeted

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running this script}"

echo "========================================"
echo "启动评估（增强重试策略）"
echo "- 模型: google/gemini-3-flash-preview"
echo "- 最大重试: 10次"
echo "- 重试延迟: 15/30/45秒..."
echo "========================================"
echo ""

# 运行所有数据集
/usr/bin/python3 -u run_evaluation.py \
    --model_name "google/gemini-3-flash-preview" \
    --temperature 0.7 \
    --budget_ratio 1.2 \
    --num_processes 5 \
    --datasets Low-price Mid-price High-price Overall \
    --output_dir "/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main/mini Main/test@1" \
    2>&1 | tee /tmp/evaluation_with_retry.log

echo ""
echo "========================================"
echo "评估完成"
echo "========================================"
