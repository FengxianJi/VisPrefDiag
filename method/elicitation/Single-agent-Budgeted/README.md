# Single LLM Auction Agent

这个目录包含了基于LLM的拍卖agent实现，用于对艺术品进行偏好排序和价格预测。

## 文件说明

- `dataset.py`: 数据加载和处理模块
- `sigelLLM.py`: LLM拍卖agent实现
- `README.md`: 本文档

## 功能特性

### Dataset模块 (dataset.py)

1. **数据加载**: 从JSON文件加载测试集数据
2. **随机化处理**:
   - 按group_id分组items
   - 随机打乱group顺序
   - 随机打乱每个item内部的图片顺序
   - 保证价格公平性
3. **预算计算**: 自动计算budget（默认为所有图片价格之和的1.2倍）
4. **Episode创建**: 为LLM决策创建完整的episode数据

### SigelLLM模块 (sigelLLM.py)

1. **Contact Sheet**: 将所有候选图片拼接成一张缩略图展示
2. **偏好排序**: 在episode开始时获取LLM对所有艺术品的偏好排序
3. **价格预测**: 逐个展示艺术品图片，获取LLM的出价预测
4. **决策历史**: 记录每次决策的历史和剩余预算
5. **偏好跟踪**: 维护LLM的偏好序列，并在后续决策中提供参考

## 安装依赖

```bash
pip install pillow openai
```

## 使用方法

### 1. 基本使用

```python
from dataset import load_dataset
from sigelLLM import SigelLLMAuction, run_auction_episode

# 加载数据集
dataset = load_dataset(
    json_path='../../../benchmark/EmergingArt_test100/data/Low-price.json',
    budget_ratio=1.2,
    seed=42
)

# 创建episode
episode = dataset.create_episode(
    shuffle_items=True,
    shuffle_images=True
)

# 创建LLM agent
agent = SigelLLMAuction(
    api_key='your-openai-api-key',
    model_name='gpt-4o',
    temperature=0.7
)

# 运行完整的拍卖episode
result = run_auction_episode(
    agent=agent,
    episode=episode,
    output_dir='./output'
)
```

### 2. 数据集使用示例

```python
from dataset import load_dataset

# 加载数据集
dataset = load_dataset(
    json_path='path/to/Overall.json',
    budget_ratio=1.2,
    seed=42
)

# 获取统计信息
stats = dataset.get_statistics()
print(f"总items数: {stats['total_items']}")
print(f"Groups数: {stats['num_groups']}")
print(f"默认Budget: €{stats['default_budget']:.2f}")

# 创建单个episode
episode = dataset.create_episode()
print(f"Episode Group ID: {episode['group_id']}")
print(f"Budget: €{episode['budget']:.2f}")
print(f"Items: {episode['num_items']}")

# 创建所有episodes
episodes = dataset.create_all_episodes()
for ep in episodes:
    print(f"Group {ep['group_id']}: {ep['num_items']} items, Budget: €{ep['budget']:.2f}")
```

### 3. 自定义预算比例

```python
# 使用不同的预算比例
dataset_low = load_dataset('Low-price.json', budget_ratio=1.0, seed=42)  # 刚好够
dataset_high = load_dataset('High-price.json', budget_ratio=1.5, seed=42)  # 更宽裕
```

### 4. LLM Agent使用示例

```python
from sigelLLM import SigelLLMAuction

# 创建agent
agent = SigelLLMAuction(
    api_key='your-api-key',
    api_base='https://api.openai.com/v1',
    model_name='gpt-4o',
    temperature=0.7
)

# 手动控制流程
# 1. 创建contact sheet
contact_sheet = agent.create_contact_sheet(
    image_paths=['path/to/img1.jpg', 'path/to/img2.jpg'],
    thumbnail_size=(300, 300),
    output_path='contact_sheet.jpg'
)

# 2. 获取偏好排序
preference = agent.get_preference_ranking(
    contact_sheet=contact_sheet,
    budget=1000.0,
    num_items=6
)
print(f"偏好排序: {preference['preference_ranking']}")

# 3. 逐个预测价格
for i, item_images in enumerate(all_items_images, 1):
    prediction = agent.predict_price(
        item_images=item_images,
        item_index=i,
        budget=1000.0,
        remaining_budget=remaining,
        num_remaining_items=6-i+1
    )
    print(f"Item {i} 出价: €{prediction['bid_price']:.2f}")
    remaining -= prediction['bid_price']

# 4. 获取决策摘要
summary = agent.get_decision_summary()
print(f"总花费: €{summary['total_spent']:.2f}")
print(f"正向出价次数: {summary['num_positive_bids']}")
```

## 工作流程

1. **Episode开始**:
   - 加载数据并创建episode
   - 随机打乱items和图片顺序
   - 计算budget

2. **展示Contact Sheet**:
   - 将所有候选艺术品的第一张图片拼接成一张大图
   - 展示给LLM

3. **获取偏好排序**:
   - LLM根据contact sheet给出偏好排序
   - 记录偏好序列

4. **逐个决策**:
   - 按顺序展示每个艺术品的所有图片
   - 提供：总预算、剩余预算、剩余items数、历史决策、初始偏好
   - LLM给出出价预测
   - 记录决策历史

5. **结果保存**:
   - Contact sheet图片
   - 完整的决策历史
   - 偏好排序
   - 预测价格 vs 实际价格

## 输出格式

### Episode结构
```json
{
  "group_id": 0,
  "items": [
    {
      "folder_id": "96",
      "price": 110.0,
      "folder_path": "/path/to/folder",
      "images": ["/path/to/image1.jpg"],
      "num_images": 1
    }
  ],
  "budget": 832.80,
  "total_price": 694.00,
  "num_items": 6
}
```

### 决策结果结构
```json
{
  "episode_info": {
    "group_id": 0,
    "budget": 832.80,
    "total_price": 694.00,
    "num_items": 6
  },
  "preference_result": {
    "preference_ranking": [3, 1, 5, 2, 4, 6],
    "reasoning": "Ranking based on artistic quality..."
  },
  "predictions": [
    {
      "item_index": 1,
      "folder_id": "88",
      "actual_price": 110.0,
      "bid_price": 120.0,
      "prediction": {
        "bid_price": 120.0,
        "reasoning": "This artwork shows..."
      }
    }
  ],
  "summary": {
    "total_decisions": 6,
    "total_spent": 650.0,
    "num_positive_bids": 5,
    "decision_history": [...],
    "preference_sequence": [3, 1, 5, 2, 4, 6]
  }
}
```

## 配置说明

### 数据集配置

- `json_path`: 测试集JSON文件路径（Overall.json, Low-price.json, Mid-price.json, High-price.json）
- `budget_ratio`: 预算比例，默认1.2（实际总价的1.2倍）
- `seed`: 随机种子，用于可复现性

### LLM Agent配置

- `api_key`: OpenAI API密钥
- `api_base`: API base URL（默认：https://api.openai.com/v1）
- `model_name`: 模型名称（推荐：gpt-4o, gpt-4-vision-preview）
- `temperature`: 温度参数（0-1，控制随机性）

## 注意事项

1. **API Key**: 使用前需要设置OpenAI API key
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

2. **模型选择**: 必须使用支持vision的模型（如gpt-4o）

3. **预算管理**: agent会自动确保出价不超过剩余预算

4. **图片格式**: 支持jpg, jpeg, png格式

5. **随机性**: 使用seed参数可以确保实验可复现

## 测试数据集

位于 `/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt_test100/data/`:

- `Overall.json`: 完整测试集（36个items，6个groups）
- `Low-price.json`: 低价区间（12个items，2个groups）
- `Mid-price.json`: 中价区间（12个items，2个groups）
- `High-price.json`: 高价区间（12个items，2个groups）

## 扩展开发

### 添加新的决策策略

在`sigelLLM.py`中可以扩展新的决策策略：

```python
def predict_price_with_strategy(self, strategy='balanced'):
    if strategy == 'aggressive':
        # 更激进的出价策略
        pass
    elif strategy == 'conservative':
        # 更保守的出价策略
        pass
    elif strategy == 'balanced':
        # 平衡策略（默认）
        pass
```

### 使用其他LLM API

修改`_call_llm_api`方法以支持其他API：

```python
def _call_llm_api(self, prompt: str, image_base64: str) -> str:
    # 修改此方法以适配其他API
    # 如: Claude, Gemini, 自定义API等
    pass
```

## 许可证

MIT License
