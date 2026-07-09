#!/bin/bash
# 监控评估进度

clear
echo "=========================================="
echo "Google Gemini 3 Flash Preview - 评估监控"
echo "=========================================="
echo ""

# 检查进程状态
echo "1. 进程状态:"
echo "----------------------------------------"
PID=$(ps aux | grep "run_with_retry.sh" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    echo "✓ 评估正在运行 (PID: $PID)"
    PYTHON_PIDS=$(ps aux | grep "run_evaluation.py" | grep "gemini" | grep -v grep | wc -l)
    echo "✓ 活动进程数: $PYTHON_PIDS"
else
    echo "✗ 评估未运行"
fi
echo ""

# 显示已完成的scenes
echo "2. 完成进度:"
echo "----------------------------------------"
COMPLETED=$(grep -c "Completed scene" /tmp/gemini3_evaluation.log 2>/dev/null || echo "0")
echo "已完成 scenes: $COMPLETED"
echo ""

# 显示最新日志
echo "3. 最新日志 (最后20行):"
echo "----------------------------------------"
tail -20 /tmp/gemini3_evaluation.log 2>/dev/null || echo "日志文件不存在"
echo ""

# 检查错误
echo "4. 错误统计:"
echo "----------------------------------------"
ERRORS=$(grep -c "ERROR\|Error\|error" /tmp/gemini3_evaluation.log 2>/dev/null || echo "0")
echo "错误数量: $ERRORS"
if [ "$ERRORS" -gt 0 ]; then
    echo "最新错误:"
    grep "ERROR\|Error" /tmp/gemini3_evaluation.log 2>/dev/null | tail -3
fi
echo ""

echo "=========================================="
echo "监控时间: $(date)"
echo "日志文件: /tmp/gemini3_evaluation.log"
echo "=========================================="
