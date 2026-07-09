# Valuator 方案1：预拍卖全局观察阶段 - 修改说明

## 修改日期
2026-02-12

## 修改目标
实施方案1：预拍卖全局观察阶段，解决 Valuator 中间排名混乱的问题。

## 核心思想
在拍卖开始前，让所有 bidder 先观察所有作品的 contact sheet，建立全局偏好排序。
在后续拍卖出价时，bidder 可以结合全局偏好信息进行更合理的预算分配。

## 修改内容

### 1. `src/bidder_base.py`

#### 修改位置：第 126-130 行
添加了全局偏好字段：

```python
# Phase 0: Pre-auction global observation
global_preference: list = []  # Global preference ranking from pre-auction observation
```

#### 修改位置：第 470-507 行
修改 `get_bid_instruct` 方法，添加全局偏好信息：

```python
def get_bid_instruct(self, auctioneer_msg: str, bid_round: int):
    auctioneer_msg = auctioneer_msg.replace(self.name, f'You ({self.name})')

    # Add global preference information if available
    preference_info = ""
    if hasattr(self, 'global_preference') and self.global_preference:
        cur_item = self._get_cur_item()
        try:
            # Find current item's rank in global preference
            if cur_item.id in self.global_preference:
                rank = self.global_preference.index(cur_item.id) + 1
                total_items = len(self.global_preference)
                preference_info = f"\n[Global Context] Based on your initial observation of all artworks, you ranked this artwork #{rank} out of {total_items} in your preference."
        except (ValueError, AttributeError):
            pass

    bid_instruct = INSTRUCT_BID_TEMPLATE.format(...)

    # Append preference info to bid instruction
    if preference_info:
        bid_instruct = bid_instruct + preference_info + "\nConsider your global ranking when deciding your bid amount and budget allocation strategy."
    
    ...
```

**效果**：bidder 在出价时会看到类似这样的提示：
```
[Global Context] Based on your initial observation of all artworks, you ranked this artwork #2 out of 6 in your preference.
Consider your global ranking when deciding your bid amount and budget allocation strategy.
```

---

### 2. `src/auctioneer_base.py`

#### 修改位置：第 169 行之后
添加 `pre_auction_observation` 方法：

```python
def pre_auction_observation(self, bidders: List):
    """
    Phase 0: Pre-auction global observation
    让所有bidder先观察所有作品，建立全局偏好排序

    Args:
        bidders: bidder列表
    """
    import ujson as json

    print("\n" + "="*60)
    print("Phase 0: Pre-auction Global Observation")
    print("="*60)

    # 生成contact sheet（如果还没有）
    if self.contact_sheet_path is None:
        self.generate_contact_sheet()

    if self.contact_sheet_path is None:
        print("Warning: No contact sheet available, skipping pre-auction observation")
        return

    # 每个bidder观察并建立全局偏好
    for bidder in bidders:
        if bidder.model_name == 'rule':
            # Rule bidder 使用默认排序
            bidder.global_preference = list(range(len(self.items)))
            continue

        # 构建观察prompt
        prompt = f"""You will participate in an auction with {len(self.items)} artworks.
Above is a contact sheet showing all {len(self.items)} artworks you can bid on.

Please carefully observe and analyze these artworks, then rank them from most to least preferred based on artistic quality, style, composition, and potential value.

Return your response in the following JSON format:
{{
    "preference_ranking": [2, 0, 4, 1, 3, 5],  // List of artwork indices (0 to {len(self.items)-1}) from most to least preferred
    "reasoning": "Your reasoning for this ranking"
}}

IMPORTANT: Return only valid JSON without any additional text or markdown formatting."""

        try:
            # 调用LLM获取全局偏好
            messages = [
                SystemMessage(content=bidder.system_message),
                HumanMessage(content=prompt)
            ]

            result = bidder._run_llm_standalone(messages)

            # 解析响应
            response = result.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            parsed = json.loads(response)

            if 'preference_ranking' in parsed:
                bidder.global_preference = parsed['preference_ranking']
                print(f"  {bidder.name}: preference_ranking = {bidder.global_preference}")
            else:
                # 默认排序
                bidder.global_preference = list(range(len(self.items)))
                print(f"  {bidder.name}: No preference_ranking found, using default")

        except Exception as e:
            print(f"  {bidder.name}: Error in pre-auction observation: {e}")
            # 默认排序
            bidder.global_preference = list(range(len(self.items)))

    print("="*60 + "\n")
```

**效果**：在拍卖开始前，会输出类似：
```
============================================================
Phase 0: Pre-auction Global Observation
============================================================
  Bidder 1: preference_ranking = [2, 0, 4, 1, 3, 5]
  Bidder 2: preference_ranking = [0, 2, 1, 4, 5, 3]
  Bidder 3: preference_ranking = [1, 0, 3, 2, 5, 4]
============================================================
```

---

### 3. `auction_workflow.py`

#### 修改位置：第 100-108 行
在 Plan Round 之后、拍卖开始之前调用预拍卖观察：

```python
plan_instructs = [bidder.get_plan_instruct(auctioneer.items) for bidder in bidder_list]

bidding_multithread(bidder_list, plan_instructs, func_type='plan', thread_num=thread_num)

if yield_for_demo:
    chatbot_list = bidders_to_chatbots(bidder_list)
    yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr, disable_gr] + disable_all_box(bidder_list)

# ***************** Phase 0: Pre-auction Global Observation *****************
auctioneer.pre_auction_observation(bidder_list)

if yield_for_demo:
    chatbot_list = bidders_to_chatbots(bidder_list)
    yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr, disable_gr] + disable_all_box(bidder_list)

bar = tqdm(total=len(auctioneer.items_queue), desc='Auction Progress')
```

**效果**：拍卖流程变为：
1. Learn Round（如果启用）
2. Plan Round
3. **Phase 0: Pre-auction Global Observation**（新增）
4. Auction Round（逐个拍卖）

---

## 工作原理

### Phase 0: 预拍卖观察
1. Auctioneer 生成所有作品的 contact sheet
2. 每个 bidder 看到 contact sheet，调用 LLM 建立全局偏好排序
3. 偏好排序存储在 `bidder.global_preference` 中（例如 `[2, 0, 4, 1, 3, 5]` 表示最喜欢作品2，其次是作品0...）

### Phase 1: 拍卖阶段
1. 逐个拍卖作品
2. 当 bidder 需要出价时，`get_bid_instruct` 会自动添加全局偏好信息
3. Bidder 看到提示："你在初始观察时将这个作品排名为 #2"
4. Bidder 可以根据全局排名决定出价策略（高排名→高出价，低排名→低出价或退出）

---

## 预期效果

| 指标 | 当前 | 预期 | 提升 |
|---|---|---|---|
| Top1Hit | 67.3% | 65-68% | 保持 |
| Rank-3 | 33.8% | 45-50% | +12-16% |
| Rank-4 | 36.1% | 48-55% | +12-19% |
| Acc-All | 24.1% | 30-32% | +6-8% |
| Acc-Low | 25.2% | 35-38% | +10-13% |

---

## 优势

1. ✅ **解决信息不对称**：bidder 知道后面还有什么作品
2. ✅ **减少预算路径依赖**：可以根据全局偏好预先规划预算
3. ✅ **保留博弈特性**：仍然是多智能体竞价，保留 Top1Hit 优势
4. ✅ **实施简单**：只需在拍卖前加一个阶段，不改变核心拍卖逻辑

---

## 兼容性

- ✅ 向后兼容：如果 `global_preference` 为空，代码会正常运行（不添加偏好信息）
- ✅ Rule bidder 兼容：Rule bidder 使用默认排序 `[0,1,2,3,4,5]`
- ✅ 图像模式兼容：依赖 contact sheet 生成，如果不支持图像则跳过

---

## 测试建议

1. 运行小规模测试（1-2 groups）验证功能正常
2. 检查日志输出，确认 Phase 0 正常执行
3. 检查 bidder 的出价 prompt，确认包含全局偏好信息
4. 对比修改前后的准确率指标

---

## 后续改进方向

如果方案1效果良好，可以考虑：
- **方案5：后处理校准**（α=0.8）：用共识排序校正拍卖价格
- **方案2：预算分配策略**：根据全局偏好预分配预算（排名#1分配40%，#2分配25%...）

---

## 回滚方法

如果需要回滚修改：

1. 删除 `bidder_base.py` 中的 `global_preference: list = []`
2. 删除 `auctioneer_base.py` 中的 `pre_auction_observation` 方法
3. 删除 `auction_workflow.py` 中的 `auctioneer.pre_auction_observation(bidder_list)` 调用
4. 恢复 `get_bid_instruct` 方法到原始版本

或者使用 git：
```bash
cd /mnt/data/user/fengxian.ji/AUCTION/method/Valuator
git checkout src/bidder_base.py src/auctioneer_base.py auction_workflow.py
```
