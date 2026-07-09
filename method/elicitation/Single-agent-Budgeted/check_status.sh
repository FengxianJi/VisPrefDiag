#!/bin/bash
# 快速检查评估状态

LOG_FILE="/tmp/gemini3_eval_20260126_170855.log"

clear
echo "=========================================="
echo "评估状态检查"
echo "=========================================="
echo ""

# 进程状态
if ps aux | grep 9612 | grep -v grep > /dev/null; then
    echo "✓ 评估正在运行 (PID: 9612)"
    WORKERS=$(ps aux | grep "run_evaluation.py" | grep -v grep | wc -l)
    echo "✓ 工作进程: $WORKERS 个"
else
    echo "✗ 评估未运行"
fi

echo ""

# 完成进度
if [ -f "$LOG_FILE" ]; then
    COMPLETED=$(grep -c "Completed scene" "$LOG_FILE")
    echo "已完成 scenes: $COMPLETED"
    
    # 错误统计
    ERRORS=$(grep -ic error "$LOG_FILE")
    echo "错误数量: $ERRORS"
    
    echo ""
    echo "最新日志 (最后15行):"
    echo "------------------------------------------"
    tail -15 "$LOG_FILE"
else
    echo "日志文件不存在"
fi

echo ""
echo "=========================================="
echo "更新时间: $(date)"
echo "=========================================="
