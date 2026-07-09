import json
import os
import random
from typing import List, Dict
import glob


class AuctionDataset:
    """
    拍卖数据集加载器

    功能：
    1. 加载测试集JSON数据
    2. 按group_id分组items
    3. 随机打乱group顺序和item内部图片顺序
    4. 计算budget（默认为所有图片价格之和的1.2倍）
    5. 返回episode数据供LLM进行决策
    """

    def __init__(self,
                 json_path: str,
                 budget_ratio: float = 1.2,
                 seed: int = None):
        """
        初始化数据集

        Args:
            json_path: 测试集JSON文件路径（如 Overall.json, Low-price.json等）
            budget_ratio: 预算比例，默认1.2倍
            seed: 随机种子，用于可复现性
        """
        self.json_path = json_path
        self.budget_ratio = budget_ratio
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        # 加载数据
        self.data = self._load_data()
        self.groups = self._group_by_id()

    def _load_data(self) -> List[Dict]:
        """加载JSON数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _group_by_id(self) -> Dict[int, List[Dict]]:
        """按group_id分组items"""
        groups = {}
        for item in self.data:
            group_id = item['group_id']
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(item)
        return groups

    def _load_item_images(self, item: Dict) -> List[str]:
        """
        加载单个item的所有图片路径

        Args:
            item: 包含folder_path的item字典

        Returns:
            图片路径列表
        """
        figure_dir = os.path.join(item['folder_path'], 'figure')

        if not os.path.exists(figure_dir):
            return []

        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(figure_dir, ext)))

        # 排序以保证初始顺序一致
        image_files.sort()

        return image_files

    def create_episode(self,
                      group_id: int = None,
                      shuffle_groups: bool = True,
                      shuffle_items: bool = True,
                      shuffle_images: bool = True) -> Dict:
        """
        创建一个episode（一轮拍卖）

        Args:
            group_id: 指定group_id，如果为None则随机选择
            shuffle_groups: 是否随机打乱group顺序
            shuffle_items: 是否随机打乱item顺序
            shuffle_images: 是否随机打乱每个item内部的图片顺序

        Returns:
            episode字典，包含：
            - items: item列表，每个item包含images, price, folder_id等
            - budget: 预算
            - total_price: 所有item价格之和
            - group_id: 该episode的group_id
        """
        # 选择group
        if group_id is None:
            group_ids = list(self.groups.keys())
            if shuffle_groups:
                random.shuffle(group_ids)
            group_id = group_ids[0]

        items = self.groups[group_id].copy()

        # 随机打乱items顺序
        if shuffle_items:
            random.shuffle(items)

        # 处理每个item
        episode_items = []
        total_price = 0.0

        for item in items:
            # 加载图片
            images = self._load_item_images(item)

            # 随机打乱图片顺序
            if shuffle_images:
                random.shuffle(images)

            episode_item = {
                'folder_id': item['folder_id'],
                'price': item['price'],
                'folder_path': item['folder_path'],
                'images': images,
                'num_images': len(images)
            }

            episode_items.append(episode_item)
            total_price += item['price']

        # 计算budget
        budget = total_price * self.budget_ratio

        episode = {
            'group_id': group_id,
            'items': episode_items,
            'budget': budget,
            'total_price': total_price,
            'num_items': len(episode_items)
        }

        return episode

    def create_all_episodes(self,
                           shuffle_groups: bool = True,
                           shuffle_items: bool = True,
                           shuffle_images: bool = True) -> List[Dict]:
        """
        创建所有groups的episodes

        Returns:
            episode列表
        """
        episodes = []
        group_ids = list(self.groups.keys())

        if shuffle_groups:
            random.shuffle(group_ids)

        for group_id in group_ids:
            episode = self.create_episode(
                group_id=group_id,
                shuffle_groups=False,
                shuffle_items=shuffle_items,
                shuffle_images=shuffle_images
            )
            episodes.append(episode)

        return episodes

    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        total_items = len(self.data)
        num_groups = len(self.groups)

        prices = [item['price'] for item in self.data]

        stats = {
            'total_items': total_items,
            'num_groups': num_groups,
            'min_price': min(prices),
            'max_price': max(prices),
            'avg_price': sum(prices) / len(prices),
            'total_price': sum(prices),
            'default_budget': sum(prices) * self.budget_ratio
        }

        return stats


def load_dataset(json_path: str,
                budget_ratio: float = 1.2,
                seed: int = None) -> AuctionDataset:
    """
    便捷函数：加载数据集

    Args:
        json_path: JSON文件路径
        budget_ratio: 预算比例
        seed: 随机种子

    Returns:
        AuctionDataset实例

    Example:
        >>> dataset = load_dataset('data/Low-price.json', budget_ratio=1.2, seed=42)
        >>> episode = dataset.create_episode()
        >>> print(f"Budget: €{episode['budget']:.2f}")
        >>> print(f"Items: {episode['num_items']}")
    """
    return AuctionDataset(json_path, budget_ratio, seed)


if __name__ == '__main__':
    # 测试代码
    json_path = '/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data/Low-price.json'

    print("=" * 60)
    print("测试 AuctionDataset")
    print("=" * 60)

    # 创建数据集
    dataset = load_dataset(json_path, budget_ratio=1.2, seed=42)

    # 获取统计信息
    stats = dataset.get_statistics()
    print("\n数据集统计:")
    print(f"  总items数: {stats['total_items']}")
    print(f"  Groups数: {stats['num_groups']}")
    print(f"  价格范围: €{stats['min_price']:.2f} - €{stats['max_price']:.2f}")
    print(f"  平均价格: €{stats['avg_price']:.2f}")
    print(f"  总价格: €{stats['total_price']:.2f}")
    print(f"  默认Budget: €{stats['default_budget']:.2f}")

    # 创建一个episode
    print("\n" + "=" * 60)
    print("创建Episode示例")
    print("=" * 60)

    episode = dataset.create_episode(shuffle_items=True, shuffle_images=True)

    print(f"\nEpisode Group ID: {episode['group_id']}")
    print(f"Budget: €{episode['budget']:.2f}")
    print(f"Total Price: €{episode['total_price']:.2f}")
    print(f"Num Items: {episode['num_items']}")
    print(f"Remaining Budget: €{episode['budget'] - episode['total_price']:.2f}")

    print("\nItems详情:")
    for i, item in enumerate(episode['items']):
        print(f"  Item {i+1}:")
        print(f"    Folder ID: {item['folder_id']}")
        print(f"    Price: €{item['price']:.2f}")
        print(f"    Num Images: {item['num_images']}")
        if item['num_images'] > 0:
            print(f"    First Image: {os.path.basename(item['images'][0])}")
