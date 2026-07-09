# 归因链分析说明

这个目录实现了你现在最优先需要补的那条“归因链”，目标是回答：

- Hard 样本失败，到底是因为 `feature gap` 消失；
- 还是因为 reward model / evaluator 根本“看不懂”。

## 目录里的核心脚本

- `analyze_attribution_chain.py`：主分析脚本。

运行方式：

```bash
python show/analysis/test/analyze_attribution_chain.py
```

## 这个脚本做了什么

### 实验一：全局 Δs 分位 vs R@1

- 输入：`show/table/table13/classic/*_detail.json`
- 主图默认模型：`ImageReward`
- 定义：
  - `oracle gap = s(human_top1) - s(human_top2)`
- 做法：
  - 对 466 个样本按 `oracle gap` 从小到大排序；
  - 划分成 5 个等频分位段；
  - 统计每个分位段的 `R@1`；
  - 同时统计每个分位段中 Hard / Mid / Easy 的占比。

### 实验二：Hard bin 内部再分层

- 只保留 Hard 的 54 个样本；
- 仍然使用 `oracle gap`；
- 对每个 reward model 单独排序，并切成 `Hard-Low / Hard-Mid / Hard-High` 三组；
- 最后对多个 reward model 的结果取平均。

这一步是主论证的核心，因为它直接检验：

- 在同样都属于 Hard 的样本内部；
- 只要可分离信号稍微恢复一点；
- 准确率是否就会上升。

### 实验三：Δs 作为 confidence proxy 的 AUROC

- 这里使用推断时可见的：
  - `top margin = s(pred_top1) - s(pred_top2)`
- 分别在 Hard / Mid / Easy 三个 bin 内，计算 correct / incorrect 的 AUROC。

这样做的目的不是替代实验二，而是补充说明：

- 在 Hard bin 内，模型虽然不是完全失明；
- 但它手里可用的排序幅度已经很弱，因此对正确/错误样本的区分能力下降。

## 输出文件

- `fig1_global_delta_quantiles.png`：实验一主图
- `fig2_hard_subbins.png`：实验二主图
- `fig3_confidence_auroc.png`：实验三辅助图
- `global_delta_quantiles.csv`：实验一数据表
- `hard_subbins.csv`：实验二数据表
- `confidence_auroc.csv`：实验三数据表
- `analysis_summary.md`：中文结果摘要
- `attribution_chain_results.json`：完整结构化结果

## 当前实现里的一个重要处理

脚本会自动排除 gap 完全退化的模型。

例如当前 `LLaVA-Reward` 在这个导出文件里，`top margin` / Hard 内 `oracle gap` 都退化成常数，因此不适合作为 gap 分析的平均对象。脚本会在结果里明确记录这个排除原因，而不会静默忽略。

## 建议你在论文里怎么放

- 主文先放 `fig2_hard_subbins.png`
- 再放 `fig1_global_delta_quantiles.png`
- `fig3_confidence_auroc.png` 放附录或补充材料

如果你下一步要继续往论文正文里落，我建议优先补两块：

1. 把实验二写成一段非常强的归因论证；
2. 把实验一写成“difficulty bin 具有外部效度”的支撑分析。
