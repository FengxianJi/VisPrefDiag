# A. 方法与可复现性：竞标者策略/政策定义

> 代码依据：`method/Valuator/src/bidder_base.py`、`src/prompt_base.py`、`src/auctioneer_base.py`、`auction_workflow.py`、`method/Qwen3vl-4B/interface/run_auction.py`

---

## 1. 竞标者如何从输入 (c, S) 得到估值/偏好？

### 1.1 输入定义

| 符号 | 对应代码字段 | 内容 |
|------|-------------|------|
| $c$ | `item.context` | 评估语境文本（如"Evaluate this artwork for a modern office space"），每件作品独立 |
| $S$ | `items: List[Item]` | 当前 episode 的全部拍卖作品集合，每件作品包含：图像路径（`image_path`）、缩略图路径（`thumbnail_path`）、起拍价（`price`）、语境（`context`） |

**注意**：图像评估模式下 `true_value=None`，真实价格 `real_final_price` 对所有竞标者完全不可见，仅用于预算总量计算。

### 1.2 感知偏好的三阶段流程

拍卖流程按顺序分为三个阶段（`auction_workflow.py`）：

```
Learn Round（可选）→ Plan Round → Phase 0: 预拍卖全局观察 → Auction Round（逐物品竞价）
```

#### 阶段一：Phase 0 — 预拍卖全局观察（`auctioneer_base.py: pre_auction_observation`）

在拍卖正式开始前，每个竞标者接收一张 **Contact Sheet**（所有作品缩略图的网格拼接图），独立建立全局偏好排序。

**Prompt 内容**：
```
You will participate in an auction with {N} artworks.
Above is a contact sheet showing all {N} artworks you can bid on.

Please carefully observe and analyze these artworks, then rank them from most
to least preferred based on artistic quality, style, composition, and potential value.

Return your response in the following JSON format:
{
    "preference_ranking": [2, 0, 4, 1, 3, 5],
    "reasoning": "Your reasoning for this ranking"
}
IMPORTANT: Return only valid JSON without any additional text or markdown formatting.
```

**输出映射**：LLM 返回 `preference_ranking`（作品索引从最喜欢到最不喜欢的排列），存入 `bidder.global_preference: list`。

**Contact Sheet 生成**（`auctioneer_base.py: generate_contact_sheet`）：
按 $\lceil\sqrt{N}\rceil \times \lceil N/\lceil\sqrt{N}\rceil\rceil$ 的网格拼接所有作品缩略图，尺寸由 `cfg.THUMBNAIL_SIZE = (200, 200)` 控制。
若单件作品含多图则先合并为单张（`run_auction.py: _create_contact_sheet`，默认每格 300×300）。

#### 阶段二：Plan Round — 拍卖前整体规划（`prompt_base.py: INSTRUCT_PLAN_TEMPLATE`）

竞标者获取所有作品的文本描述（图像模式下仅显示名称和起拍价，不显示估值），在预算约束下完成主观评估。

**Prompt 关键内容（精简）**：
```
As {bidder_name}, you have a total budget of ${budget}. This auction has {N} items:
{items_info}

**Important Evaluation Guidelines:**
- Base your judgment ONLY on the aesthetic qualities, visual appeal, and context-fit.
- DO NOT infer value from starting prices — they are arbitrary.

**CRITICAL — Extreme Differentiation & Penalty Rule:**
- Your BEST item: allocate 90-95% of your total budget.
- Your WORST item: allocate only 2-5% of your total budget.

Output priorities (1-3) for each item as JSON, then output budget allocations as JSON.
  * 1 - Nearly worthless. Allocate $2-$5.
  * 2 - Some value but inferior. Allocate a small budget.
  * 3 - Outstanding. Allocate almost your entire budget.
```

**LLM 输出格式示例**：
```json
{"Artwork #001": 3, "Artwork #002": 1, "Artwork #003": 2}
{"Artwork #001": 950, "Artwork #002": 5, "Artwork #003": 45}
```

竞标者最终输出的文本由 `extract_jsons_from_text` 解析，后一个 JSON 作为预算分配 $w_i$，存入 `bidder.budget_allocation: dict`（item_id → 分配金额）。计划文本存入 `bidder.cur_plan`。

#### 阶段三：Bid Round — 逐物品出价（`prompt_base.py: INSTRUCT_BID_TEMPLATE`）

每件物品展示时，竞标者在对话历史（system + plan + 当前 bid history）基础上接收出价指令。

**Prompt 关键内容**：
```
Now, the auctioneer says: "{auctioneer_msg}"
This is round {k}/2 for this item.

Your current budget status: ${remaining_budget} remaining.
**YOUR ALLOCATED BUDGET FOR THIS ITEM: ${allocated}**
(You MUST bid exactly ${allocated}. Not more, not less.)

[Global Context] Based on your initial observation of all artworks,
you ranked this artwork #{rank} out of {N} in your preference.

MANDATORY BIDDING RULES:
1. You MUST bid exactly your pre-allocated amount. NON-NEGOTIABLE.
2. HARD CEILING: FORBIDDEN from bidding more than allocation.
3. HARD FLOOR: MUST bid at least allocation.
```

出价格式要求：`"I bid $xxx!"` 或 `"I'm out!"`（Round 1 禁止退出）。

---

### 1.3 解码设置（Decoding Settings）

| 参数 | Qwen3-VL-4B 本地 | API 模型（GPT/Gemini/Claude） |
|------|-----------------|------------------------------|
| `temperature` | **0.0**（`DEFAULT_TEMPERATURE = 0.0`，`run_auction.py:37`） | 配置化，默认 **0.7**（`Bidder.temperature`，`bidder_base.py:77`） |
| `top-p` | 模型默认（未显式设置） | 模型默认（未显式设置） |
| `max_tokens` | `max(4000 − input_tokens, 192)`（`run_auction.py:64`） | GPT-4: `max(8000 − input, 192)`；GPT-3.5: `max(3900 − input, 192)`；Claude: `2048`；Gemini/DeepSeek: `max(8000 − input, 192)` |
| 重试机制 | 异常时无重试 | 最多 6 次指数退避重试（`2^i` 秒） |

---

## 2. LLM/VLM 输出如何映射到预算分配和出价

### 2.1 拍卖前"预算分配计划" $w_i$（Eq.3）

$w_i$ 的生成有两条路径，最终都写入 `bidder.budget_allocation[item_id]`：

**路径 A — Plan Round LLM 直接输出**（主路径，`bidder_base.py: init_plan`）：
竞标者在 Plan Round 中直接输出每件作品的具体金额（如上节示例）。代码解析 `cur_plan` 中最后一个 JSON 对象，键为作品名、值为分配金额（整数 $）。

**路径 B — Phase 0 排序驱动的指数权重分配**（`bidder_base.py: allocate_budget_by_preference`）：
根据 Phase 0 全局偏好排序 $r \in \{0, 1, \ldots, N-1\}$，计算指数递减权重：

$$w_r = \frac{10^{-r}}{\sum_{k=0}^{N-1} 10^{-k}} \cdot B_{\text{total}}$$

特殊情况：$N=2$ 时硬编码为 $[0.95, 0.05]$；$N=1$ 时全分配。
这使得最优作品获得预算的绝对主导（例如 $N=3$：约 90.9%、9.1%、0.9%）。

**出价时预算约束注入**（`bidder_base.py: get_bid_instruct`）：
查找 `self.budget_allocation[cur_item.id]`，以 hard ceiling + hard floor 的形式强制写入 Bid prompt，要求竞标者出价严格等于预分配金额。

### 2.2 每轮出价（bid）

出价解析流程（`auction_workflow.py`）：

1. LLM 输出含 `"I bid $xxx!"` 或 `"I'm out!"` 的自由文本；
2. `parse_bid_price` 调用 `auctioneer.parse_bid(msg)`，使用 `PARSE_BID_INSTRUCTION` prompt 从文本中抽取整数价格（退出返回 -1）；
3. **强制 Round 1 出价**：若 Round 1 竞标者想退出，最多重试 3 次，仍拒绝则强制设为起拍价 `cur_item.price`；
4. **Sanity check**（最多 3 次）：出价不得超过剩余预算，且不得低于当前最高出价 + 最小加价幅度（`min_markup_pct = 0.1 × item.price`）；若 3 次仍不合规则强制退出。

最终每轮拍卖只进行 **2 轮竞价**（`MAX_BID_ROUNDS = 2`），价高者得。

---

## 3. 代理优化的目标/效用、同质/异质、信息共享

### 3.1 优化目标与效用

所有竞标者使用统一的 `desire = 'aesthetic_alignment'`（`run_auction.py:154`），系统消息中注入：

> "Your primary objective is to acquire items that **best align with your aesthetic preferences and taste**, while managing your budget strategically."

效用函数**未被显式数学化**，而是编码为 LLM 的自然语言目标——让竞标者根据主观审美形成偏好，再将偏好映射为预算分配。与原始 AuctionArena 中的 `maximize_profit`（$\max \sum(v_i - b_i)$）或 `maximize_items`（$\max |\text{won items}|$）不同，此处无确定性的效用函数，估值完全依赖 VLM 的视觉-审美判断。

此外：
- `overestimate_percent = 0`：不对真实价值施加过估偏差（图像模式下 `true_value = None`，此字段无效）；
- `correct_belief = True`：竞标者维护对其他人剩余预算/已赢物品的正确信念。

### 3.2 同质 vs. 异质

| 维度 | 设置 | 细节 |
|------|------|------|
| 目标 | **同质** | 3 个竞标者均为 `aesthetic_alignment` |
| 模型 | **同质**（Qwen3-VL-4B 本地）| 均使用同一 checkpoint，temperature=0.0 |
| 偏好 | **可异质** | Phase 0 各自独立观察 contact sheet，可形成不同排序 |
| 预算 | **同质** | 统一预算 $B = \sum_i p_i^{\text{real}} \times r$（$r=1.2$，`DEFAULT_BUDGET_RATIO`）|
| 规划策略 | **同质** | 均为 `plan_strategy='adaptive'` |

尽管三个 bidder 目标和模型相同，由于 LLM 非确定性输出（即便 temperature=0.0，由于图像理解的细微差异）以及独立的 Phase 0 观察，不同 bidder 仍可对同一组作品形成不同的优先级排序，产生竞争。

### 3.3 信息共享

| 信息类型 | 共享范围 | 机制 |
|----------|---------|------|
| Contact sheet（全体作品缩略图） | 全体竞标者可见 | Phase 0 统一广播 |
| 各自 Plan（优先级 + 预算分配） | **私有**，不共享 | 仅保存在 `bidder.cur_plan` |
| Phase 0 全局排序 | **私有**，不共享 | 仅保存在 `bidder.global_preference` |
| 每轮出价金额 | **公开**（所有竞标者可见）| 拍卖师广播到对话历史 |
| 成交价与赢家 | **公开** | `auctioneer.hammer_fall()` 后所有人更新 |
| 其他竞标者的剩余预算/盈利 | **可追踪**（`correct_belief=True`）| `status_quo.winning_bids` 和 `remaining_budget` |
| 跨 episode 学习 | **禁用**（Monte Carlo 模式）| `enable_learning=False`，每次独立重置 |

**关键结论**：竞标者之间不共享偏好/计划，但出价行为完全公开，且在 `correct_belief=True` 条件下可以从观察到的出价中推断他人的剩余预算和目标，体现了不完全信息博弈的结构。

---

## 附：完整调用链路图

```
Data: EmergingArt group (N artworks, each with images + context)
            │
            ▼
[Contact Sheet 生成]  ← Auctioneer.generate_contact_sheet()
  N × thumbnail → ceil(√N) × ceil(√N) 网格图
            │
            ▼
[Phase 0: 全局偏好观察]  ← Auctioneer.pre_auction_observation()
  Prompt: contact_sheet_image + ranking_request
  Output: global_preference = [rank_0, rank_1, ..., rank_{N-1}]
  → allocate_budget_by_preference() → budget_allocation[item_id] = w_i
            │
            ▼
[Plan Round]  ← Bidder.init_plan(INSTRUCT_PLAN_TEMPLATE)
  Input: system_msg + all_items_info (text only)
  Output: priority_json + budget_allocation_json
  → budget_allocation[item_id] = w_i (覆盖 Phase 0 分配)
            │
            ▼
  ┌─── For each item i in sequence ───┐
  │                                   │
  │  [Bid Round k=1,2]                │
  │  Prompt: system + plan + bid_hist │
  │    + auctioneer_msg               │
  │    + remaining_budget             │
  │    + "MUST bid $w_i"             │
  │    + "[Global Context] rank #r"   │
  │  Output: "I bid $b_i!"           │
  │  → parse → sanity_check → record  │
  │                                   │
  └───────────────────────────────────┘
            │
            ▼
[Monte Carlo 聚合]  ← run_monte_carlo()
  num_runs=1（默认），每次独立重置
  pred_score = mean clearing price of item i
```

---

---

# B. 拍卖流程可复现算法

> 代码依据：`auction_workflow.py`、`src/auctioneer_base.py`、`src/bidder_base.py`、`config.py`、`method/Qwen3vl-4B/interface/run_auction.py`

---

## 2.1 全局超参数与初始化

| 参数 | 符号 | 值 | 来源 |
|------|------|----|------|
| 每 episode 竞标者数 | $M$ | 3 | `run_auction.py:170` |
| 每件物品最大出价轮数 | $K$ | 2 | `auction_workflow.py: MAX_BID_ROUNDS=2` |
| 起拍价 | $p_0$ | $1（固定） | `run_auction.py:135 price=1` |
| 最小加价约束（round 2）| — | 无固定步长 $\Delta$，仅须严格 $> b^*$ | `bidder_base.py: bid_sanity_check` |
| 统一预算 | $B$ | $\sum_i p_i^{\text{real}} \times r_B$ | `run_auction.py:172–175` |
| 预算比例 | $r_B$ | **1.2**（Qwen3-VL-4B）; 0.4（config默认）| `run_auction.py: DEFAULT_BUDGET_RATIO` |
| 打折降价 | — | 禁用（`enable_discount=False`）| `run_auction.py:181` |
| Monte Carlo 运行次数 | $R$ | 1（Qwen3-VL-4B默认）; 推荐30–50 | `run_auction.py: DEFAULT_NUM_RUNS`; `config.py: NUM_RUNS=30` |
| 随机种子 | — | 42（固定）| `run_auction.py: --seed 42` |
| 物品顺序 shuffle | — | 禁用（`shuffle=False`）| `config.py: SHUFFLE_ITEMS=False` |

**预算计算**：

$$B = \left(\sum_{i=1}^{N} p_i^{\text{real}}\right) \times r_B, \quad r_B = 1.2$$

其中 $p_i^{\text{real}}$ 为物品 $i$ 的真实成交价（`item.real_final_price`，对 bidder 不可见）。所有竞标者共享相同的 $B$（`UNIFIED_BUDGET=True`）。

---

## 2.2 单次拍卖流程（Algorithm 1）

```
Algorithm 1: Single Auction for One Episode

Input:  items = [item_1, ..., item_N]  (N artworks, fixed order)
        bidders = [b_1, b_2, b_3]  (M=3, all fresh instances)
        B = unified_budget

── Phase 0: Pre-auction Global Observation ──────────────────────────────
1. Auctioneer generates contact_sheet from all N thumbnails
   (ceil(√N) × ceil(N/ceil(√N)) grid, each cell THUMBNAIL_SIZE=200×200)
2. For each bidder b_m (m=1..M):
   a. Call VLM: (system_msg, contact_sheet_img, ranking_prompt) → preference_ranking_m
      Output: global_preference_m = [idx(1), idx(2), ..., idx(N)]  (most→least preferred)
      Fallback if parse fails: global_preference_m = [0, 1, ..., N-1]
   b. Compute Phase-0 budget allocation:
        w_i^(0) = B × 10^{-rank(i,m)} / Σ_j 10^{-rank(j,m)}
        Special case N=2: [0.95·B, 0.05·B]

── Plan Round ────────────────────────────────────────────────────────────
3. For each bidder b_m (parallel threads):
   a. Call VLM: (system_msg, INSTRUCT_PLAN_TEMPLATE) → plan_text
   b. Parse last JSON in plan_text → {item_name: dollar_amount}
      → budget_allocation_m[item.id] = w_i  (overwrites Phase-0 allocation)
   c. Store cur_plan_m = plan_text

── Auction Round: For each item i = 1..N (sequential) ───────────────────
4. auctioneer.present_item(i) → cur_item = item_i
5. highest_bidder ← None; highest_bid ← -1; prev_round_max_bid ← -1

6. For k = 0, 1  (bid_round, 0-indexed):

   a. Build auctioneer_msg:
        k=0: announce item_i, remaining items list, starting price p_0=$1
        k=1: announce round info, show round-0 bids, current highest bid b*

   b. active_bidders ← {b_m | b_m ≠ highest_bidder AND NOT b_m.withdraw}

   c. For each b_m ∈ active_bidders (parallel):
      i.  Inject bid instruction:
            · auctioneer_msg
            · remaining_budget_m
            · "MUST bid exactly $w_i"  (from budget_allocation_m[item_i.id])
            · "[Global Context] you ranked this artwork #rank out of N"
            Dialogue context: [system_msg] + [plan_instruct, plan_text] + bid_history
      ii. Call VLM → raw_text → parse "I bid $xxx!" or "I'm out!" → bid_price

      iii. Round-1 mandatory-bid enforcement (k=0, bid_price < 0):
             Retry ≤ 3×; if still refusing → force bid_price = p_0 = $1

      iv. Sanity check (≤ 3 retries; exceed → force withdraw bid_price = -1):
            · bid_price ≤ remaining_budget_m            (budget constraint)
            · if k=0: bid_price ≥ p_0                  (≥ starting price)
            · if k≥1: bid_price > prev_round_max_bid    (strictly > current max; no fixed Δ)

   d. Record bids; update highest_bidder / highest_bid:
        For each valid bid (bid_price > 0):
          if bid_price > highest_bid:
            highest_bid ← bid_price; highest_bidder ← b_m
          elif bid_price == highest_bid:       ← TIE-BREAKING
            highest_bidder ← random.choice([highest_bidder, b_m])  (50/50)

   e. prev_round_max_bid ← highest_bid

   f. check_hammer(k):
        k=0: sold TRUE  if (only 1 bidder AND prev_round_max_bid < 0)
                            OR (0 effective bids this round AND highest_bidder is None)
             sold FALSE otherwise (continue to k=1)
        k=1: sold TRUE always (hard stop after 2 rounds)
        → break loop if sold

7. Determine winner and clearing price:
     if highest_bidder is None:
       item unsold (fail_to_sell=True); clearing_price_i = None (excluded from stats)
     else:
       winner_i = highest_bidder; clearing_price_i = highest_bid
       winner_i.budget -= clearing_price_i

8. Update all bidders' status_quo (remaining_budget, winning_bids of all bidders)

9. Adaptive replan (plan_strategy='adaptive'):
   Each bidder calls replan with (updated status_quo + remaining items)
   → updates cur_plan and budget_allocation for subsequent items

Output per item: (winner_i, clearing_price_i)
```

---

## 2.3 停止条件汇总

| 层级 | 条件 | 处理 |
|------|------|------|
| 单件物品 | `bid_round >= 1`（已完成2轮）| 强制落锤（返回 True）|
| 单件物品 | round 0 仅 1 人出价且无历史最高价 | 直接成交 |
| 单件物品 | 所有人退出且 highest_bidder=None | 流拍，`fail_to_sell=True`，无清算价 |
| 单次出价 | sanity check 3 次仍不合规 | 强制退出（`bid_price=-1`）|
| 整轮拍卖 | `items_queue` 为空 | `auctioneer.end_auction()` 返回 True |
| Monte Carlo | 已完成 $R$ 次独立拍卖 | 聚合统计后退出 |

---

## 2.4 Tie-breaking（平局处理）

同一轮内多人出价相同（`auctioneer_base.py: record_bid`）：

```python
elif self.highest_bid == hist['bid']:
    self.highest_bidder = random.choice([self.highest_bidder, hist['bidder']])
```

等概率随机抽签（1/2）决定 provisional winner。由于 `random.seed(42)` 在 `run_monte_carlo` 入口处设置，相同 seed 下结果可复现。

三路平局时：`record_bid` 按出价到达顺序依次更新，等效于顺序两两随机抽签。

---

## 2.5 Monte Carlo 随机性控制与聚合（Algorithm 2）

```
Algorithm 2: Monte Carlo Aggregation

Input:  episode (N items), R = num_runs, seed = 42

1. random.seed(seed); np.random.seed(seed)

2. clearing_prices[item_name] ← []

3. For run r = 1..R:
   a. Create fresh bidder instances (no shared state; enable_learning=False)
   b. Create fresh Auctioneer; init_items(items)  (shuffle=False, order fixed)
   c. Run Algorithm 1 → {item_name: clearing_price_r}
   d. Append clearing_price_r to clearing_prices[item_name]
   e. Incremental save after each group (fault tolerance)

4. For each item i with prices = [P_i^(1), ..., P_i^(R)]:
   stats_i = {
     mean   : μ_i = mean(prices),          ← pred_score
     std    : σ_i = std(prices),
     median : median(prices),
     min/max: min/max(prices),
     count  : |prices|,
     winner_distribution: {bidder_name: win_count}
   }

5. pred_score_i ← μ_i  (output 'pred_score' field)
```

**聚合策略对比**：

| 方式 | 实现状态 | 适用场景 |
|------|---------|---------|
| Mean $\mu_i$ | 默认（`pred_score = stats['mean']`）| 无偏期望估计，对称分布最优 |
| Median | 已计算未用作 pred_score | 分布偏斜时更鲁棒 |
| Voting（多数赢）| 未实现（winner_distribution 已记录）| top-1 预测场景 |

$R=1$（Qwen3-VL-4B 默认）时无聚合效果；正式评估建议 $R \geq 10$。

**随机性来源**：

| 来源 | seed 控制 | 说明 |
|------|-----------|------|
| Tie-breaking `random.choice` | 受控 | seed=42 全局设置 |
| Item shuffle | 受控（已禁用）| `SHUFFLE_ITEMS=False` |
| LLM 采样 | 部分受控 | Qwen3-VL-4B `temperature=0.0` 近似确定；API 模型 `temperature=0.7` 有随机性 |
| 多进程出价顺序 | 不受控 | `Pool.imap` 跨进程随机状态不同步 |

---

---

# C. 清算价 → 最终分数/排序向量 $\mathbf{s}$ 的映射

> 代码依据：`method/Qwen3vl-4B/interface/run_auction.py`、`show/table/table1.py`

---

## 3.1 第一步：清算价 → pred_score（跨 run 聚合）

$$s_i = \frac{1}{R} \sum_{r=1}^{R} P_i^{(r)}$$

代码：`run_auction.py:207`，`'pred_score': stats['mean']`。$R=1$ 时 $s_i = P_i^{(1)}$。

---

## 3.2 第二步：Episode 内 Min-Max 归一化 → $\hat{s}_i$

$$\hat{s}_i = \frac{s_i - \min_{j \in \mathcal{E}} s_j}{\max_{j \in \mathcal{E}} s_j - \min_{j \in \mathcal{E}} s_j + \varepsilon}, \quad \varepsilon = 10^{-8}$$

$\mathcal{E}$ 为同一 episode 的全部 $N$ 件物品，**无跨 episode 归一化**。代码：`table1.py: normalize_scores`。

$\hat{s}_i \in [0,1]$；最优物品 $=1$，最差物品 $\approx 0$。

---

## 3.3 衍生统计量

**$\sigma_S$（总体标准差，episode 内离散度）**：

$$\sigma_S = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\hat{s}_i - \bar{\hat{s}}\right)^2}, \quad \bar{\hat{s}} = \frac{1}{N}\sum_{i=1}^{N} \hat{s}_i$$

除以 $N$（总体标准差，非样本标准差 $N-1$）。

**$\Delta_{\text{top}}$（top-2 分数差）**：

$$\Delta_{\text{top}} = \hat{s}_{(1)} - \hat{s}_{(2)}, \quad \hat{s}_{(1)} \geq \hat{s}_{(2)} \geq \cdots$$

---

## 3.4 Episode-Normalized Preference-Strength Score：公式与示例

**完整映射链**：

$$P_i^{(r)} \;\xrightarrow[\text{(mean across R runs)}]{}\; s_i \;\xrightarrow[\text{(min-max within episode)}]{}\; \hat{s}_i \;\xrightarrow[\text{(argsort desc)}]{}\; \hat{\pi}$$

**数值示例**（$N=3$，$R=5$，$r_B=1.2$）：

| 物品 | $p_i^{\text{real}}$ | 清算价 $\{P_i^{(r)}\}_{r=1}^{5}$ | $s_i$ | $\hat{s}_i$ |
|------|---------------------|----------------------------------|-------|-------------|
| A | \$800 | {480, 500, 460, 490, 470} | **480** | **1.000** |
| B | \$350 | {210, 220, 200, 215, 205} | **210** | **0.356** |
| C | \$120 | {60, 65, 58, 62, 60} | **61** | **0.000** |

$$B = (800+350+120)\times 1.2 = \$1{,}548$$

$$\text{denom} = 480 - 61 + 10^{-8} = 419$$

$$\hat{s}_A = \frac{419}{419} = 1.000,\quad \hat{s}_B = \frac{149}{419} \approx 0.356,\quad \hat{s}_C = 0.000$$

$$\bar{\hat{s}} = 0.452,\quad \sigma_S = \sqrt{\frac{0.300+0.009+0.204}{3}} \approx 0.414,\quad \Delta_{\text{top}} = 0.644$$

$$g_{\text{price}} = \frac{800-350}{800} = 0.5625,\quad \text{GapErr} = |0.644 - 0.563| = 0.081$$

预测排名 $\hat{\pi}$：$A \succ B \succ C$；真实排名 $\pi^*$：$A \succ B \succ C$ ✓

Top1Hit = 1，Regret@1 = 0，Spearman-$\rho$ = 1.0（本例完全一致）。

---

## 3.5 跨 Episode 处理

**无跨 episode 归一化**，所有指标均 per-episode 计算后取算术均值：

$$\text{Metric}_{\text{overall}} = \frac{1}{|\mathcal{D}|} \sum_{e \in \mathcal{D}} \text{Metric}_e$$

**阈值的跨 episode 全局计算**（FailTop / FailDisp / OSR / USR）：

```python
# 全部 episodes 的 Δ_top 和 σ_S 列表
all_delta_tops, all_sigma_ss = compute_delta_top_and_sigma(episodes)
delta_threshold_10 = percentile(all_delta_tops, 10)   # global 10th pct → FailTop/USR 阈值
delta_threshold_90 = percentile(all_delta_tops, 90)   # global 90th pct → OSR 阈值
sigma_threshold_10 = percentile(all_sigma_ss,   10)   # global 10th pct → FailDisp 阈值
```

阈值全局计算确保 Near-tie 子集的判断标准在所有方法间具有可比性。
