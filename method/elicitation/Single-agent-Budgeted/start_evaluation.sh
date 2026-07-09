#!/bin/bash
# 启动脚本：运行 Single-agent-Budgeted 评估

# 设置环境变量
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running this script}"

# 进入脚本目录
cd /mnt/data/user/fengxian.ji/AUCTION/method/Single-agent-Budgeted

# 设置日志文件
LOG_DIR="/mnt/data/user/fengxian.ji/AUCTION/method/Single-agent-Budgeted/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"

# 运行评估脚本
echo "======================================"
echo "Single-agent-Budgeted 评估开始"
echo "开始时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "======================================"
echo ""

/usr/bin/python3 run_evaluation.py \
    --model_name "google/gemini-3-flash-preview" \
    --temperature 0.7 \
    --budget_ratio 1.2 \
    --num_processes 20 \
    --datasets Low-price Mid-price High-price Overall \
    --output_dir "/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main/mini Main/test@1" \
    2>&1 | tee "$LOG_FILE"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "评估成功完成!"
    echo "结束时间: $(date)"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "评估出现错误，请查看日志: $LOG_FILE"
    echo "结束时间: $(date)"
    echo "======================================"
    exit 1
fi
