"""
Auction Arena - Unified Configuration Center
统一超参数配置中心

用于图像评估的拍卖机制配置
所有实验的超参数在此统一管理，确保可复现性和一致性
"""

import os

# ============================================================
# 0. API 配置
# ============================================================

# UniAPI 配置（用于 DeepSeek 等模型）
UNIAPI_API_KEY = os.environ.get("UNIAPI_API_KEY", "")
UNIAPI_BASE_URL = "https://api.uniapi.io/v1"

# 将配置设置到环境变量（如果未设置）
if UNIAPI_API_KEY and not os.environ.get('UNIAPI_API_KEY'):
    os.environ['UNIAPI_API_KEY'] = UNIAPI_API_KEY
if not os.environ.get('UNIAPI_BASE_URL'):
    os.environ['UNIAPI_BASE_URL'] = UNIAPI_BASE_URL


class AuctionConfig:
    """拍卖系统统一配置类"""

    # ============================================================
    # 1. 预算机制 (Budget Mechanism)
    # ============================================================

    # 预算比例：统一预算 = 所有物品起拍价之和 × BUDGET_RATIO
    # 推荐值：0.4 (中等稀缺)
    # - 0.25-0.35: 激烈竞争，高稀缺性
    # - 0.40: 平衡的中等稀缺 (推荐)
    # - 0.45-0.60: 较为宽松，低竞争强度
    # 注意：当起拍价设为 1 时，此比例会产生极低预算，建议使用 FIXED_BUDGET
    BUDGET_RATIO = 0.4

    # 是否为所有 bidder 设置统一预算
    # True: 所有 agent 预算相同（消除预算差异的影响）
    # False: 从 bidders.jsonl 中读取各自预算
    UNIFIED_BUDGET = True

    # 固定预算模式（用于起拍价很小或为0的场景）
    # None: 使用 BUDGET_RATIO 基于真实成交价计算预算
    # 数值（如 500）: 所有 bidder 使用固定预算 $500，不依赖成交价
    # 推荐：设为 None，让预算根据真实拍卖的价值规模自动调整
    FIXED_BUDGET = None

    # ============================================================
    # 2. Monte Carlo 实验控制
    # ============================================================

    # Monte Carlo 运行次数
    # 用于估计期望成交价和价格方差
    # 推荐值：30-50 次（平衡统计显著性和成本）
    NUM_RUNS = 30

    # 随机种子（用于控制拍卖顺序是否 shuffle）
    # None: 每次运行使用不同随机种子
    # 固定值（如 42）: 所有运行使用相同随机种子，结果可复现
    RANDOM_SEED = 42

    # 是否在每次运行前 shuffle 物品顺序
    # False: 固定顺序（推荐，方差只来自 Agent 决策）
    # True: 每次随机顺序
    SHUFFLE_ITEMS = False

    # ============================================================
    # 3. 并发与性能控制
    # ============================================================

    # 最大并发线程数
    # 建议值：2-3（避免 API rate limit）
    # 图像输入会增加 token 消耗，需要控制并发
    MAX_THREADS = 2

    # API 请求间隔（秒）
    API_REQUEST_INTERVAL = 0.5

    # ============================================================
    # 4. Agent 默认配置
    # ============================================================

    # 默认 LLM 模型
    DEFAULT_MODEL = 'gpt-4-0613'

    # 采样温度（控制输出随机性）
    # 0.0: 确定性输出
    # 0.7: 平衡的随机性（推荐）
    # 1.0+: 更高随机性
    DEFAULT_TEMPERATURE = 0.7

    # 规划策略
    # - 'none': 不做规划，逐个竞拍
    # - 'static': 初始规划一次，不再更新
    # - 'adaptive': 动态再规划（推荐）
    DEFAULT_PLAN_STRATEGY = 'adaptive'

    # 是否纠正 agent 的错误信念
    # True: 当 agent 记错自己的预算/物品时，系统自动纠正
    # False: 不纠正（更真实但可能导致逻辑错误）
    CORRECT_BELIEF = True

    # 是否启用跨轮学习
    # True: agent 从上一轮拍卖中学习经验
    # False: 每轮独立运行
    ENABLE_LEARNING = False

    # 过度估值百分比（用于模拟不同风险偏好）
    # 0: 估值等于感知价值
    # 10: 愿意出价比感知价值高 10%
    # -10: 保守出价，低于感知价值 10%
    DEFAULT_OVERESTIMATE_PERCENT = 0

    # ============================================================
    # 5. 图像相关配置 (Image-specific)
    # ============================================================

    # 图像数据目录
    IMAGE_DIR = 'data/images'

    # 缩略图尺寸（用于 contact sheet）
    THUMBNAIL_SIZE = (200, 200)

    # 是否使用 contact sheet（全局概览图）
    # True: 规划阶段展示所有候选图的拼图（使用 thumbnail_path）
    # False: 仅展示文本描述
    USE_CONTACT_SHEET = True

    # 支持的图像格式
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.webp']

    # 图像信息使用场景：
    # - 规划阶段：使用 thumbnail_path 生成 contact sheet
    # - 竞拍阶段：显示 image_path（主图）+ context（文本描述）

    # ============================================================
    # 6. 评估目标与 Prompt 配置
    # ============================================================

    # 评估目标类型
    # - 'aesthetic_alignment': 最大化美学对齐度（图像评估）
    # - 'maximize_profit': 最大化利润（原版 AuctionArena）
    # - 'maximize_items': 最大化获得物品数量
    OBJECTIVE = 'aesthetic_alignment'

    # 评估上下文类型
    # - 'office_space': 办公空间适配度
    # - 'home_decor': 家居装饰适配度
    # - 'gallery_exhibition': 画廊展览价值
    # - 'general': 通用美学评估
    EVALUATION_CONTEXT = 'general'

    # ============================================================
    # 7. 数据路径配置
    # ============================================================

    # 数据根目录
    DATA_DIR = 'data/example'

    # 物品配置文件
    ITEMS_FILE = 'items_demo.jsonl'

    # Bidder 配置文件
    BIDDERS_FILE = 'bidders_demo.jsonl'

    # 日志输出目录
    LOG_DIR = 'logs'

    # 临时文件目录（用于 contact sheet 等）
    TEMP_DIR = 'temp'

    # ============================================================
    # 8. 输出与日志配置
    # ============================================================

    # 输出格式
    # - 'csv': CSV 表格
    # - 'json': JSON 格式
    # - 'txt': 纯文本
    OUTPUT_FORMAT = 'csv'

    # 是否保存完整对话日志
    SAVE_FULL_LOG = True

    # 是否保存 agent 的中间推理过程
    SAVE_REASONING = True

    # 日志详细级别
    # - 'minimal': 仅保存最终结果
    # - 'standard': 保存关键步骤
    # - 'verbose': 保存所有细节
    LOG_LEVEL = 'standard'

    # ============================================================
    # 9. 敏感性分析配置
    # ============================================================

    # 预算比例敏感性分析的测试值
    SENSITIVITY_BUDGET_RATIOS = [0.3, 0.4, 0.5]

    # 温度参数敏感性分析的测试值
    SENSITIVITY_TEMPERATURES = [0.3, 0.7, 1.0]

    # ============================================================
    # 10. 实验控制开关
    # ============================================================

    # 是否运行敏感性分析
    RUN_SENSITIVITY_ANALYSIS = False

    # 是否生成可视化图表
    GENERATE_PLOTS = True

    # 是否计算分歧度指标（disagreement metrics）
    CALCULATE_DISAGREEMENT = True

    # 是否记录 BIP (Bidding Increment Percentage)
    TRACK_BIP = True

    # ============================================================
    # 辅助方法
    # ============================================================

    @classmethod
    def get_unified_budget(cls, items):
        """
        根据物品列表计算统一预算

        Args:
            items: 物品列表，每个物品需包含 'real_final_price' 或 'price' 字段

        Returns:
            统一预算金额（float）
        """
        # 优先使用固定预算模式
        if cls.FIXED_BUDGET is not None:
            return cls.FIXED_BUDGET

        # 否则基于真实成交价计算
        if isinstance(items, list) and len(items) > 0:
            # 处理 dict 格式的 items
            if isinstance(items[0], dict):
                # 优先使用 real_final_price（真实成交价）
                if 'real_final_price' in items[0]:
                    total_final_price = sum([item['real_final_price'] for item in items])
                else:
                    # fallback 到 price
                    total_final_price = sum([item['price'] for item in items])
            # 处理 Item 对象
            else:
                # 优先使用 real_final_price 属性
                if hasattr(items[0], 'real_final_price'):
                    total_final_price = sum([item.real_final_price for item in items])
                else:
                    # fallback 到 price
                    total_final_price = sum([item.price for item in items])
        else:
            total_final_price = 0

        return total_final_price * cls.BUDGET_RATIO

    @classmethod
    def to_dict(cls):
        """
        导出所有配置为字典（用于日志记录）

        Returns:
            配置字典
        """
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and k.isupper()
        }

    @classmethod
    def update(cls, **kwargs):
        """
        批量更新配置参数

        Args:
            **kwargs: 要更新的参数

        Example:
            AuctionConfig.update(BUDGET_RATIO=0.5, MAX_THREADS=4)
        """
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    @classmethod
    def print_config(cls):
        """打印当前配置（用于调试）"""
        print("=" * 60)
        print("Current Auction Configuration")
        print("=" * 60)

        sections = {
            "Budget Mechanism": ["BUDGET_RATIO", "UNIFIED_BUDGET"],
            "Concurrency": ["MAX_THREADS", "API_REQUEST_INTERVAL"],
            "Agent Defaults": ["DEFAULT_MODEL", "DEFAULT_TEMPERATURE", "DEFAULT_PLAN_STRATEGY"],
            "Image Config": ["USE_CONTACT_SHEET", "THUMBNAIL_SIZE"],
            "Evaluation": ["OBJECTIVE", "EVALUATION_CONTEXT"],
            "Output": ["OUTPUT_FORMAT", "LOG_LEVEL"],
        }

        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                if hasattr(cls, key):
                    print(f"  {key}: {getattr(cls, key)}")

        print("=" * 60)

    @classmethod
    def validate(cls):
        """
        验证配置参数的合理性

        Raises:
            ValueError: 如果配置不合理
        """
        # 检查预算比例
        if not 0.1 <= cls.BUDGET_RATIO <= 1.0:
            raise ValueError(f"BUDGET_RATIO must be in [0.1, 1.0], got {cls.BUDGET_RATIO}")

        # 检查温度参数
        if not 0.0 <= cls.DEFAULT_TEMPERATURE <= 2.0:
            raise ValueError(f"DEFAULT_TEMPERATURE must be in [0.0, 2.0], got {cls.DEFAULT_TEMPERATURE}")

        # 检查规划策略
        if cls.DEFAULT_PLAN_STRATEGY not in ['none', 'static', 'adaptive']:
            raise ValueError(f"Invalid DEFAULT_PLAN_STRATEGY: {cls.DEFAULT_PLAN_STRATEGY}")

        # 检查评估目标
        if cls.OBJECTIVE not in ['aesthetic_alignment', 'maximize_profit', 'maximize_items']:
            raise ValueError(f"Invalid OBJECTIVE: {cls.OBJECTIVE}")

        print("✓ Configuration validation passed")
        return True


# ============================================================
# 快捷配置预设 (Presets)
# ============================================================

class PresetConfigs:
    """预定义的配置预设"""

    @staticmethod
    def minimal_test():
        """最小测试配置（快速验证）"""
        AuctionConfig.update(
            MAX_THREADS=1,
            SAVE_FULL_LOG=False
        )

    @staticmethod
    def standard_experiment():
        """标准实验配置（论文使用）"""
        AuctionConfig.update(
            BUDGET_RATIO=0.4,
            MAX_THREADS=2,
            DEFAULT_PLAN_STRATEGY='adaptive'
        )

    @staticmethod
    def high_competition():
        """高竞争配置（低预算）"""
        AuctionConfig.update(
            BUDGET_RATIO=0.3
        )

    @staticmethod
    def low_competition():
        """低竞争配置（高预算）"""
        AuctionConfig.update(
            BUDGET_RATIO=0.6
        )


# ============================================================
# 使用示例
# ============================================================

if __name__ == '__main__':
    # 打印当前配置
    AuctionConfig.print_config()

    # 验证配置
    AuctionConfig.validate()

    # 示例：更新配置
    # AuctionConfig.update(BUDGET_RATIO=0.5, MAX_THREADS=4)

    # 示例：使用预设
    # PresetConfigs.minimal_test()
    # AuctionConfig.print_config()

    # 示例：计算统一预算
    example_items = [
        {'name': 'Item 1', 'price': 1000},
        {'name': 'Item 2', 'price': 2000},
        {'name': 'Item 3', 'price': 1500}
    ]
    budget = AuctionConfig.get_unified_budget(example_items)
    print(f"\nUnified budget for example items: ${budget}")
    print(f"(Total starting price: ${sum(item['price'] for item in example_items)}, ratio: {AuctionConfig.BUDGET_RATIO})")
