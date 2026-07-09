import sys
import os
sys.path.append('..')
from utils import LoadJsonL


class Item():
    """
    拍卖物品类

    支持两种模式：
    1. 原版模式：基于 true_value 的虚拟商品拍卖
    2. 图像评估模式：基于图像的美学评估（无 true_value）
    """

    def __init__(self,
                 id: int,
                 name: str,
                 price: int,
                 desc: str = '',
                 true_value: int = None,
                 # 图像相关字段
                 image_path: str = None,
                 thumbnail_path: str = None,
                 # 评估上下文字段
                 context: str = None,
                 style_tags: list = None,
                 artist_info: str = None,
                 # 真实成交价（用于预算计算）
                 real_final_price: float = None,
                 # LLM评分字段（0-100分）
                 llm_score: float = None,
                 # 保留兼容性（但不使用）
                 detail_image_path: str = None):
        """
        初始化拍卖物品

        Args:
            id: 物品唯一标识符
            name: 物品名称
            price: 起拍价（用于拍卖起始）
            desc: 物品描述（原版模式使用）
            true_value: 真实价值（仅用于原版模式，图像评估模式下为 None）

            # 图像相关参数（图像评估模式）
            image_path: 主图路径（竞拍时展示的完整图片）
            thumbnail_path: 缩略图路径（规划阶段 contact sheet 使用）

            # 评估上下文参数（图像评估模式）
            context: 评估语境/文本描述（如"评估该作品在现代办公空间的适配度"）
            style_tags: 风格标签列表（可选，如 ["abstract", "warm"]）
            artist_info: 艺术家信息（可选）

            # 预算计算参数
            real_final_price: 真实成交价（用于计算统一预算，不展示给 bidders）

            # LLM评分参数（图像评估模式）
            llm_score: LLM评分（0-100分，用于替代 true_value 计算利润）

        实际使用的信息结构：
            规划阶段：显示所有 thumbnail_path 拼接的 contact sheet
            竞拍阶段：显示 image_path（主图） + context（文本描述）
        """
        self.id = id
        self.name = name
        self.price = price
        self.desc = desc
        self.true_value = true_value  # 图像评估模式下为 None
        self._original_price = price

        # 真实成交价（用于预算计算，不展示给 bidders）
        self.real_final_price = real_final_price if real_final_price is not None else price

        # LLM评分（0-100分，图像评估模式下用于替代 true_value）
        self.llm_score = llm_score

        # 图像相关属性（实际使用）
        self.image_path = image_path              # 主图（竞拍时展示）
        self.thumbnail_path = thumbnail_path      # 缩略图（contact sheet）

        # 评估上下文属性
        self.context = context                    # 文本描述
        self.style_tags = style_tags if style_tags else []
        self.artist_info = artist_info

        # 保留字段（向后兼容，但实际不使用）
        self.detail_image_path = detail_image_path

        # 判断是否为图像评估模式
        self.is_image_mode = image_path is not None

    def get_desc(self):
        """
        生成拍卖描述文本

        Returns:
            拍卖师使用的描述字符串
        """
        base_desc = f"{self.name}, starting at ${int(self.price)}."

        # 如果有评估上下文，添加到描述中
        if self.context:
            base_desc += f"\n[Context: {self.context}]"

        return base_desc

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"{self.name}"

    def info(self):
        """
        返回物品的完整信息

        Returns:
            信息字符串（根据模式不同，格式不同）
        """
        if self.is_image_mode:
            # 图像评估模式：不显示 true_value
            info_str = f"{self.name}: starting at ${int(self.price)}"
            if self.style_tags:
                info_str += f" [Styles: {', '.join(self.style_tags)}]"
            return info_str
        else:
            # 原版模式：显示价格区间
            if self.true_value is not None:
                return f"{self.name}: ${int(self.price)} to ${self.true_value}."
            else:
                return f"{self.name}: starting at ${int(self.price)}."

    def has_image(self):
        """
        检查是否有主图

        Returns:
            bool: 是否存在主图路径且文件存在
        """
        if self.image_path is None:
            return False
        return os.path.exists(self.image_path)

    def has_thumbnail(self):
        """
        检查是否有缩略图

        Returns:
            bool: 是否存在缩略图路径且文件存在
        """
        if self.thumbnail_path is None:
            return False
        return os.path.exists(self.thumbnail_path)

    def get_image_info(self):
        """
        获取图像信息摘要

        Returns:
            dict: 图像相关信息
        """
        return {
            'has_main_image': self.has_image(),
            'has_thumbnail': self.has_thumbnail(),
            'image_path': self.image_path,
            'thumbnail_path': self.thumbnail_path
        }

    def lower_price(self, percentage: float = 0.2):
        """
        降低起拍价（流拍降价机制）

        Args:
            percentage: 降价百分比（默认 20%）
        """
        # lower starting price by percentage
        self.price = int(self.price * (1 - percentage))

    def reset_price(self):
        """
        重置为原始起拍价
        """
        self.price = self._original_price

    def to_dict(self):
        """
        导出为字典（用于日志记录）

        Returns:
            dict: 物品的所有属性
        """
        return {
            'id': self.id,
            'name': self.name,
            'price': self.price,
            'desc': self.desc,
            'true_value': self.true_value,
            'image_path': self.image_path,
            'thumbnail_path': self.thumbnail_path,
            'context': self.context,
            'style_tags': self.style_tags,
            'artist_info': self.artist_info,
            'is_image_mode': self.is_image_mode,
            'llm_score': self.llm_score
        }


def create_items(item_info_jsl):
    """
    从 JSONL 文件加载物品列表

    支持两种格式：
    1. 原版格式：包含 true_value 的虚拟商品
    2. 图像格式：包含 image_path 等字段的图像评估物品

    Args:
        item_info_jsl: JSONL 文件路径

    Returns:
        list[Item]: Item 对象列表

    Example JSONL (原版格式):
        {"id": 1, "name": "Widget A", "price": 1000, "desc": "A widget", "true_value": 2000}

    Example JSONL (图像格式):
        {"id": 1, "name": "Artwork #001", "price": 500,
         "image_path": "data/images/001.jpg",
         "thumbnail_path": "data/images/001_thumb.jpg",
         "context": "Evaluate this artwork for office decoration",
         "style_tags": ["abstract", "warm"]}

    信息使用场景：
        - 规划阶段：使用 thumbnail_path 生成 contact sheet
        - 竞拍阶段：显示 image_path（主图）+ context（文本描述）
    """
    item_info_jsl = LoadJsonL(item_info_jsl)
    item_list = []
    for info in item_info_jsl:
        item_list.append(Item(**info))
    return item_list


def item_list_equal(items_1: list, items_2: list):
    """
    比较两个物品列表是否包含相同的物品（不考虑顺序）

    Args:
        items_1: 物品列表 1（可以是 Item 对象或字符串）
        items_2: 物品列表 2（可以是 Item 对象或字符串）

    Returns:
        bool: 两个列表是否包含相同的物品
    """
    # could be a list of strings (names) or a list of Items
    item_1_names = [item.name if isinstance(item, Item) else item for item in items_1]
    item_2_names = [item.name if isinstance(item, Item) else item for item in items_2]
    return set(item_1_names) == set(item_2_names)


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("测试 Item 类")
    print("=" * 60)

    # 测试 1: 原版模式（带 true_value）
    print("\n【测试 1】原版模式（虚拟商品）")
    item1 = Item(
        id=1,
        name="Widget A",
        price=1000,
        desc="A widget for all your needs",
        true_value=2000
    )
    print(f"  创建成功: {item1}")
    print(f"  get_desc(): {item1.get_desc()}")
    print(f"  info(): {item1.info()}")
    print(f"  is_image_mode: {item1.is_image_mode}")

    # 测试 2: 图像评估模式（无 true_value）
    print("\n【测试 2】图像评估模式")
    item2 = Item(
        id=2,
        name="Artwork #001",
        price=500,
        desc="Abstract painting",
        image_path="data/images/001.jpg",
        thumbnail_path="data/images/001_thumb.jpg",
        context="评估该作品在现代办公空间的适配度",
        style_tags=["abstract", "warm", "minimalist"],
        artist_info="新生代画家 John Doe"
    )
    print(f"  创建成功: {item2}")
    print(f"  get_desc(): {item2.get_desc()}")
    print(f"  info(): {item2.info()}")
    print(f"  is_image_mode: {item2.is_image_mode}")
    print(f"  style_tags: {item2.style_tags}")
    print(f"  context: {item2.context}")

    # 测试 3: 图像文件检查
    print("\n【测试 3】图像文件检查")
    print(f"  has_image(): {item2.has_image()}")
    print(f"  has_thumbnail(): {item2.has_thumbnail()}")
    print(f"  get_image_info(): {item2.get_image_info()}")

    # 测试 4: 价格操作
    print("\n【测试 4】价格操作")
    print(f"  原始价格: ${item1.price}")
    item1.lower_price(0.2)
    print(f"  降价 20% 后: ${item1.price}")
    item1.reset_price()
    print(f"  重置后: ${item1.price}")

    # 测试 5: 导出为字典
    print("\n【测试 5】导出为字典")
    print(f"  item1.to_dict():")
    for key, value in item1.to_dict().items():
        print(f"    {key}: {value}")

    # 测试 6: 列表比较
    print("\n【测试 6】列表比较")
    items_a = [item1, item2]
    items_b = [item2, item1]
    print(f"  item_list_equal([item1, item2], [item2, item1]): {item_list_equal(items_a, items_b)}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
