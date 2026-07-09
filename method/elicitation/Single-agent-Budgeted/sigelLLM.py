import os
import json
import base64
from typing import List, Dict, Tuple
from PIL import Image
import io
import math


class SigelLLMAuction:
    """
    单个LLM拍卖agent

    功能：
    1. 创建contact sheet（将所有候选图片拼接成一张图）
    2. 调用LLM API获取偏好排序
    3. 逐个展示图片并获取价格预测
    4. 记录决策历史和偏好序列
    """

    def __init__(self,
                 api_key: str = None,
                 api_base: str = None,
                 model_name: str = "gpt-4o",
                 temperature: float = 0.7):
        """
        初始化LLM agent

        Args:
            api_key: LLM API密钥
            api_base: API base URL
            model_name: 模型名称
            temperature: 温度参数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model_name = model_name
        self.temperature = temperature

        # 决策历史
        self.decision_history = []
        self.preference_sequence = []

    def create_contact_sheet(self,
                            image_paths: List[str],
                            thumbnail_size: Tuple[int, int] = (300, 300),
                            grid_cols: int = None,
                            output_path: str = None) -> Image.Image:
        """
        创建contact sheet（拼图）

        Args:
            image_paths: 图片路径列表
            thumbnail_size: 缩略图大小
            grid_cols: 网格列数，如果为None则自动计算
            output_path: 输出路径，如果提供则保存图片

        Returns:
            PIL Image对象
        """
        num_images = len(image_paths)

        # 自动计算网格大小
        if grid_cols is None:
            grid_cols = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_cols)

        # 创建空白画布
        canvas_width = grid_cols * thumbnail_size[0]
        canvas_height = grid_rows * thumbnail_size[1]
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        # 粘贴每张图片
        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                continue

            # 计算位置
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * thumbnail_size[0]
            y = row * thumbnail_size[1]

            # 加载并调整图片大小
            img = Image.open(img_path)
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

            # 粘贴到画布
            canvas.paste(img, (x, y))

        # 保存（可选）
        if output_path:
            canvas.save(output_path)

        return canvas

    def image_to_base64(self, image: Image.Image) -> str:
        """将PIL Image转换为base64字符串"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def get_preference_ranking(self,
                              contact_sheet: Image.Image,
                              budget: float,
                              num_items: int) -> Dict:
        """
        第一步：展示contact sheet，让LLM给出偏好排序

        Args:
            contact_sheet: 拼接好的contact sheet图片
            budget: 预算
            num_items: item数量

        Returns:
            包含偏好排序和理由的字典
        """
        # 将图片转换为base64
        images_base64 = [self.image_to_base64(contact_sheet)]

        # 构建prompt
        prompt = f"""You are participating in an art auction. You have been given a budget of €{budget:.2f} to bid on {num_items} artworks.

Above is a contact sheet showing all {num_items} candidate artworks in this auction round.

Please analyze these artworks and provide your preference ranking from most to least preferred. Consider artistic quality, style, composition, and potential value.

Return your response in the following JSON format:
{{
    "preference_ranking": [1, 2, 3, ...],  // List of artwork numbers (1 to {num_items}) from most to least preferred
    "reasoning": "Your reasoning for this ranking"
}}

IMPORTANT: Return only valid JSON without any additional text or markdown formatting."""

        # 调用LLM API
        try:
            response = self._call_llm_api(prompt, images_base64)

            # 清理响应（移除可能的markdown代码块）
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            # 解析响应
            result = json.loads(response)

            # 检查是否有错误
            if 'error' in result:
                raise Exception(f"API返回错误: {result['error']}")

            # 检查是否有preference_ranking
            if 'preference_ranking' not in result:
                print(f"⚠️ API响应中没有preference_ranking字段，使用默认排序")
                result['preference_ranking'] = list(range(1, num_items + 1))

            # 保存偏好序列
            self.preference_sequence = result.get('preference_ranking', [])

            return result

        except Exception as e:
            print(f"Error getting preference ranking: {e}")
            print(f"Response was: {response if 'response' in locals() else 'No response'}")
            # 返回默认排序
            return {
                'preference_ranking': list(range(1, num_items + 1)),
                'reasoning': f'Error occurred: {str(e)}'
            }

    def predict_price(self,
                     item_images: List[str],
                     item_index: int,
                     budget: float,
                     remaining_budget: float,
                     num_remaining_items: int) -> Dict:
        """
        第二步：逐个展示item的图片，让LLM预测价格

        Args:
            item_images: 当前item的所有图片路径
            item_index: 当前item的索引（从1开始）
            budget: 总预算
            remaining_budget: 剩余预算
            num_remaining_items: 剩余item数量

        Returns:
            包含预测价格和理由的字典
        """
        # 创建当前item的图片展示
        images_base64 = [self.image_to_base64(Image.open(img)) for img in item_images]

        # 获取历史决策信息
        history_text = self._format_decision_history()

        # 获取偏好信息
        preference_info = ""
        if self.preference_sequence and item_index <= len(self.preference_sequence):
            rank = self.preference_sequence.index(item_index) + 1 if item_index in self.preference_sequence else "Unknown"
            preference_info = f"\nBased on your initial preference ranking, this artwork was ranked #{rank}."

        # 构建prompt
        prompt = f"""You are participating in an art auction. Here is your current situation:

Total Budget: €{budget:.2f}
Remaining Budget: €{remaining_budget:.2f}
Current Item: #{item_index}
Remaining Items to bid on: {num_remaining_items} (including this one)
{preference_info}

{history_text}

Above shows the artwork you need to bid on now. You can see all images for this artwork.

Please provide your bid price for this artwork. Consider:
1. Your initial preference for this artwork
2. Your remaining budget
3. Number of remaining artworks
4. Your previous decisions
5. Balancing preference with budget constraints

You can bid any amount from €0 to your remaining budget. You can also choose not to bid (€0) to save budget for later items.

Return your response in the following JSON format:
{{
    "bid_price": 150.00,  // Your bid amount in euros
    "reasoning": "Your reasoning for this bid"
}}

IMPORTANT: Return only valid JSON without any additional text or markdown formatting."""

        # 调用LLM API
        try:
            response = self._call_llm_api(prompt, images_base64)

            # 清理响应（移除可能的markdown代码块）
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            # 解析响应
            result = json.loads(response)

            # 检查是否有错误
            if 'error' in result:
                raise Exception(f"API返回错误: {result['error']}")

            # 检查是否有bid_price
            if 'bid_price' not in result:
                raise Exception(f"API响应中没有bid_price字段。响应内容: {result}")

            bid_price = float(result.get('bid_price', 0))

            # 确保出价不超过剩余预算
            bid_price = min(bid_price, remaining_budget)
            bid_price = max(bid_price, 0)

            result['bid_price'] = bid_price

            # 记录决策历史
            self.decision_history.append({
                'item_index': item_index,
                'bid_price': bid_price,
                'reasoning': result.get('reasoning', ''),
                'remaining_budget_before': remaining_budget,
                'remaining_budget_after': remaining_budget - bid_price
            })

            return result

        except Exception as e:
            print(f"Error predicting price for item {item_index}: {e}")
            print(f"Response was: {response if 'response' in locals() else 'No response'}")
            return {
                'bid_price': 0.0,
                'reasoning': f'Error occurred: {str(e)}'
            }

    def _format_decision_history(self) -> str:
        """格式化决策历史为文本"""
        if not self.decision_history:
            return "This is your first bid in this auction round."

        history_lines = ["Your previous decisions in this auction:"]
        for decision in self.decision_history:
            history_lines.append(
                f"  Item #{decision['item_index']}: Bid €{decision['bid_price']:.2f} "
                f"(Remaining: €{decision['remaining_budget_after']:.2f})"
            )

        return "\n".join(history_lines)

    def _call_llm_api(self, prompt: str, images_base64: list, max_retries: int = 10, retry_delay: int = 15) -> str:
        """
        调用LLM API（带重试机制）- 使用OpenRouter

        Args:
            prompt: 文本prompt
            images_base64: base64编码的图片列表
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）

        Returns:
            LLM响应文本
        """
        import time
        from openai import OpenAI

        for attempt in range(max_retries):
            try:
                print(f"    [API调用] 尝试 {attempt + 1}/{max_retries}...")

                # 创建OpenAI客户端（使用OpenRouter）
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                )

                # 构建内容列表：所有图片 + 文本
                content_list = []
                for img_b64 in images_base64:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })
                content_list.append({
                    "type": "text",
                    "text": prompt
                })

                # 构建消息
                messages = [
                    {
                        "role": "system",
                        "content": "You are an art expert participating in art auctions. Analyze artworks and provide price predictions based on artistic quality, composition, and market value."
                    },
                    {
                        "role": "user",
                        "content": content_list
                    }
                ]

                # 调用API
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://github.com/auction-research",
                        "X-Title": "Art Auction Research",
                    },
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=16000
                )

                # 获取响应内容
                content = completion.choices[0].message.content

                # 检查content是否为空
                if not content or content.strip() == "":
                    print(f"[WARNING] API返回空内容，finish_reason: {completion.choices[0].finish_reason}")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    return '{"error": "API返回空内容"}'

                return content

            except Exception as e:
                error_msg = str(e)

                # 检查是否是速率限制或负载错误
                if 'rate' in error_msg.lower() or 'limit' in error_msg.lower() or '负载' in error_msg or 'overloaded' in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"[API速率限制] 等待 {wait_time} 秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return f'{{"error": "API速率限制，已达最大重试次数"}}'

                # 其他错误也重试
                if attempt < max_retries - 1:
                    print(f"[ERROR] API调用异常: {e}，等待 {retry_delay} 秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                else:
                    import traceback
                    error_detail = traceback.format_exc()
                    print(f"API调用错误: {error_detail}")
                    return f'{{"error": "API调用失败: {str(e)}"}}'

        return '{"error": "已达最大重试次数"}'

    def reset(self):
        """重置agent状态，开始新的episode"""
        self.decision_history = []
        self.preference_sequence = []

    def get_decision_summary(self) -> Dict:
        """获取决策摘要"""
        if not self.decision_history:
            return {}

        total_spent = sum(d['bid_price'] for d in self.decision_history)
        num_bids = len([d for d in self.decision_history if d['bid_price'] > 0])

        return {
            'total_decisions': len(self.decision_history),
            'total_spent': total_spent,
            'num_positive_bids': num_bids,
            'decision_history': self.decision_history,
            'preference_sequence': self.preference_sequence
        }


def run_auction_episode(agent: SigelLLMAuction,
                       episode: Dict,
                       output_dir: str = None) -> Dict:
    """
    运行一个完整的拍卖episode

    Args:
        agent: SigelLLMAuction实例
        episode: 从dataset.create_episode()返回的episode字典
        output_dir: 输出目录，用于保存contact sheet和结果

    Returns:
        包含所有决策和结果的字典
    """
    # 重置agent
    agent.reset()

    budget = episode['budget']
    items = episode['items']
    num_items = len(items)

    print(f"\n{'='*60}")
    print(f"开始Episode {episode['group_id']}")
    print(f"Budget: €{budget:.2f}, Items: {num_items}")
    print(f"{'='*60}\n")

    # 步骤1: 创建contact sheet
    print("步骤1: 创建contact sheet...")
    all_first_images = [img for item in items for img in item['images']]

    contact_sheet_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        contact_sheet_path = os.path.join(output_dir, f"contact_sheet_group{episode['group_id']}.jpg")

    contact_sheet = agent.create_contact_sheet(
        all_first_images,
        thumbnail_size=(300, 300),
        output_path=contact_sheet_path
    )

    if contact_sheet_path:
        print(f"  Contact sheet已保存: {contact_sheet_path}")

    # 步骤2: 获取偏好排序
    print("\n步骤2: 获取偏好排序...")
    preference_result = agent.get_preference_ranking(contact_sheet, budget, num_items)
    print(f"  偏好排序: {preference_result['preference_ranking']}")
    print(f"  理由: {preference_result['reasoning'][:100]}...")

    # 步骤3: 逐个item获取价格预测
    print("\n步骤3: 逐个item获取价格预测...\n")

    remaining_budget = budget
    predictions = []

    for idx, item in enumerate(items, 1):
        print(f"Item {idx}/{num_items} (Folder ID: {item['folder_id']}):")

        # 获取价格预测
        prediction = agent.predict_price(
            item_images=item['images'],
            item_index=idx,
            budget=budget,
            remaining_budget=remaining_budget,
            num_remaining_items=num_items - idx + 1
        )

        bid_price = prediction['bid_price']
        actual_price = item['price']

        print(f"  实际价格: €{actual_price:.2f}")
        print(f"  预测出价: €{bid_price:.2f}")
        print(f"  剩余预算: €{remaining_budget:.2f} -> €{remaining_budget - bid_price:.2f}")
        print(f"  理由: {prediction['reasoning'][:80]}...")

        # 更新剩余预算
        remaining_budget -= bid_price

        predictions.append({
            'item_index': idx,
            'folder_id': item['folder_id'],
            'actual_price': actual_price,
            'bid_price': bid_price,
            'prediction': prediction
        })

        print()

    # 总结
    print(f"{'='*60}")
    print("Episode总结:")
    print(f"  总预算: €{budget:.2f}")
    print(f"  总花费: €{budget - remaining_budget:.2f}")
    print(f"  剩余预算: €{remaining_budget:.2f}")
    print(f"  实际总价: €{episode['total_price']:.2f}")
    print(f"{'='*60}\n")

    # 保存结果
    result = {
        'episode_info': {
            'group_id': episode['group_id'],
            'budget': budget,
            'total_price': episode['total_price'],
            'num_items': num_items
        },
        'preference_result': preference_result,
        'predictions': predictions,
        'summary': agent.get_decision_summary()
    }

    if output_dir:
        result_path = os.path.join(output_dir, f"result_group{episode['group_id']}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {result_path}")

    return result


if __name__ == '__main__':
    # 测试代码
    from dataset import load_dataset

    print("=" * 60)
    print("测试 SigelLLMAuction")
    print("=" * 60)

    # 注意：实际使用时需要设置API key
    print("\n注意: 本测试需要设置OPENAI_API_KEY环境变量")
    print("示例: export OPENAI_API_KEY='your-api-key'")

    # 创建数据集
    json_path = '/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt_test100/data/Low-price.json'
    dataset = load_dataset(json_path, budget_ratio=1.2, seed=42)

    # 创建episode
    episode = dataset.create_episode()

    # 创建agent（如果没有API key，会返回错误信息）
    agent = SigelLLMAuction(
        model_name="gpt-4o",
        temperature=0.7
    )

    # 运行episode
    output_dir = '/mnt/data/user/fengxian.ji/AUCTION/method/sigelLLM/output'
    result = run_auction_episode(agent, episode, output_dir)

    print("\n测试完成！")
