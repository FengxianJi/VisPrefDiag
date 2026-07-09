#!/bin/bash
# 监控评估进度的脚本

echo "==============================================="
echo "Single-agent-Budgeted 评估进度监控"
echo "==============================================="
echo ""

# 检查进程状态
echo "1. 进程状态:"
echo "-------------------------------------------"
MAIN_PID=$(ps aux | grep "start_evaluation.sh" | grep -v grep | awk '{print $2}' | head -1)
PYTHON_PID=$(ps aux | grep "run_evaluation.py" | grep -v grep | head -1 | awk '{print $2}')

if [ -n "$MAIN_PID" ]; then
    echo "✓ 主进程运行中 (PID: $MAIN_PID)"
else
    echo "✗ 主进程未运行"
fi

if [ -n "$PYTHON_PID" ]; then
    echo "✓ Python评估进程运行中 (PID: $PYTHON_PID)"
    CHILD_COUNT=$(ps aux | grep "run_evaluation.py" | grep -v grep | wc -l)
    echo "  子进程数: $CHILD_COUNT"
else
    echo "✗ Python评估进程未运行"
fi

echo ""

# 检查日志文件
echo "2. 最新日志（最后20行）:"
echo "-------------------------------------------"
if [ -f "/tmp/full_evaluation.log" ]; then
    tail -20 /tmp/full_evaluation.log
else
    echo "日志文件不存在"
fi

echo ""

# 检查输出文件
echo "3. 输出文件状态:"
echo "-------------------------------------------"
OUTPUT_DIR="/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main/mini Main/test@1"
if [ -d "$OUTPUT_DIR" ]; then
    echo "输出目录: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR" 2>/dev/null || echo "输出目录为空"
else
    echo "输出目录尚未创建"
fi

echo ""

# 检查日志目录
echo "4. 详细日志文件:"
echo "-------------------------------------------"
LOG_DIR="/mnt/data/user/fengxian.ji/AUCTION/method/Single-agent-Budgeted/logs"
if [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "最新日志: $LATEST_LOG"
        echo "文件大小: $(du -h "$LATEST_LOG" | cut -f1)"
        echo ""
        echo "最后30行:"
        tail -30 "$LATEST_LOG"
    else
        echo "没有日志文件"
    fi
else
    echo "日志目录不存在"
fi

echo ""
echo "==============================================="
echo "监控完成 - $(date)"
echo "==============================================="
