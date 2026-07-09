#!/bin/bash
# 实时监控评估进度

echo "========================================"
echo "Single-agent-Budgeted 评估进度监控"
echo "========================================"
echo ""

# 查看最新输出
echo "最新进度（最后30行）:"
echo "----------------------------------------"
tail -30 /tmp/claude/-mnt-data-user-fengxian-ji-AUCTION/tasks/b9dccca.output
echo ""

# 统计进度
echo "----------------------------------------"
echo "统计信息:"
COMPLETED_ITEMS=$(grep -c "实际:" /tmp/claude/-mnt-data-user-fengxian-ji-AUCTION/tasks/b9dccca.output 2>/dev/null || echo 0)
API_RETRIES=$(grep -c "API负载饱和" /tmp/claude/-mnt-data-user-fengxian-ji-AUCTION/tasks/b9dccca.output 2>/dev/null || echo 0)

echo "  已完成预测: $COMPLETED_ITEMS 个"
echo "  API重试次数: $API_RETRIES 次"
echo ""

# 检查进程
echo "----------------------------------------"
PROCESS_COUNT=$(ps aux | grep "run_simple.py" | grep -v grep | wc -l)
if [ $PROCESS_COUNT -gt 0 ]; then
    echo "✓ 评估进程运行中"
    ps aux | grep "run_simple.py" | grep -v grep | awk '{print "  PID: "$2", 运行时间: "$10}'
else
    echo "✗ 评估进程未运行"
fi
echo ""

# 检查结果文件
echo "----------------------------------------"
if [ -f "/tmp/simple_evaluation_result.json" ]; then
    echo "✓ 结果文件已生成: /tmp/simple_evaluation_result.json"
    PRED_COUNT=$(cat /tmp/simple_evaluation_result.json | grep -o "folder_id" | wc -l)
    echo "  包含 $PRED_COUNT 个预测"
else
    echo "⏳ 结果文件尚未生成"
fi

echo ""
echo "========================================"
echo "监控时间: $(date)"
echo "========================================"
