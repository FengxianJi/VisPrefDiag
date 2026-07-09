# 归因链分析结果

## 1. 分析目标

- 目标命题：Hard 样本失败是否主要因为 feature gap 消失，而不是模型完全看不懂。
- 数据来源：直接读取 `show/table/table13/classic/*_detail.json` 与 `benchmark/ImageReward/test.json`。
- 人类 difficulty bin：Hard = 54，Mid = 229，Easy = 183，总计 466 组。

## 2. 本次实现里使用的两个 gap 定义

- `oracle gap = s(human_top1) - s(human_top2)`：用于实验一和实验二，直接衡量 reward model 是否还能把人类最优图和最接近的竞争者拉开。
- `top margin = s(pred_top1) - s(pred_top2)`：用于实验三，作为推断阶段可见的 confidence proxy，检验在不同 difficulty bin 内还能否排序正确/错误样本。

## 3. 参考 table13 的总体结果

- ImageReward：Overall R@1 = 34.12%
- HPSv2：Overall R@1 = 34.55%
- PickScore：Overall R@1 = 30.90%
- LLaVA-Reward：Overall R@1 = 41.85%

## 4. 实验一：全局 Δs 分位与准确率

- 主图默认使用 `ImageReward`。
- R@1 是否单调不下降：是。
- Q1：n=94，R@1=0.00%，Hard/Mid/Easy=10.6%/48.9%/40.4% ，Δs范围=[-2.654, -0.463]
- Q2：n=93，R@1=0.00%，Hard/Mid/Easy=15.1%/53.8%/31.2% ，Δs范围=[-0.461, -0.065]
- Q3：n=93，R@1=44.09%，Hard/Mid/Easy=15.1%/43.0%/41.9% ，Δs范围=[-0.058, 0.202]
- Q4：n=93，R@1=54.84%，Hard/Mid/Easy=10.8%/55.9%/33.3% ，Δs范围=[0.219, 0.776]
- Q5：n=93，R@1=72.04%，Hard/Mid/Easy=6.5%/44.1%/49.5% ，Δs范围=[0.777, 2.781]

## 5. 实验二：Hard bin 内部再分层

- 用于平均的有效模型：ImageReward, HPSv2, PickScore。
- 排除 LLaVA-Reward：Hard 子集上的 oracle gap 全部相同，无法做有效分层。
- 平均 R@1 是否单调不下降：是。
- Hard-Low：平均R@1=0.00%，模型间标准差=0.00%，平均Δs范围≈[-0.980, -0.163]
- Hard-Mid：平均R@1=24.07%，模型间标准差=11.42%，平均Δs范围≈[-0.151, 0.143]
- Hard-High：平均R@1=70.37%，模型间标准差=6.93%，平均Δs范围≈[0.182, 0.943]

## 6. 实验三：confidence AUROC

- 用于平均的有效模型：ImageReward, HPSv2, PickScore。
- 排除 LLaVA-Reward：top margin 全部相同，无法形成有效的 confidence ranking。
- Hard：平均AUROC=56.36%，模型间标准差=1.94%
- Mid：平均AUROC=57.74%，模型间标准差=4.45%
- Easy：平均AUROC=58.56%，模型间标准差=1.90%

## 7. 建议如何写进论文

- 主论证先放实验二：只看 Hard bin，若 Hard-High 明显高于 Hard-Low，就能直接反驳“模型完全看不懂”的解释。
- 实验一紧接着给全局外部效度：随着 oracle gap 增大，R@1 整体上升，说明 difficulty bin 背后确实对应可测的信号强弱。
- 实验三作为辅助解释：在 Hard bin 内，推断时可见的 top margin 对 correct/incorrect 的排序能力最弱，说明不是方向错了，而是可用幅度不够。
- 如果正文篇幅有限，可把实验一放主文图，实验二放主文小表，实验三放附录图。
