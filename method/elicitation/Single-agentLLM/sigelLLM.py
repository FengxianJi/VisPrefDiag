import os
import json
import base64
from typing import List, Dict


class SigelLLMAuction:
    """
    单个LLM拍卖agent

    功能：
    1. 调用LLM API直接预测价格
    2. 记录决策历史
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

    def image_to_base64(self, image_path: str) -> str:
        """将图片文件转换为base64字符串"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    def predict_price(self,
                     item_images: List[str],
                     item_index: int) -> Dict:
        """
        展示item的图片，让LLM直接预测价格

        Args:
            item_images: 当前item的所有图片路径
            item_index: 当前item的索引（从1开始）

        Returns:
            包含预测价格和理由的字典
        """
        # 使用第一张图片
        if not item_images:
            return {
                'bid_price': 0.0,
                'reasoning': 'No images available'
            }

        # 将所有图片转换为base64
        images_base64 = [self.image_to_base64(img) for img in item_images]

        # 构建简化的prompt
        prompt = f"""You are an art expert evaluating artwork #{item_index} in an auction.

Above shows the artwork you need to evaluate.

Please predict the fair market price for this artwork based on:
1. Artistic quality and technique
2. Composition and aesthetics
3. Style and originality
4. Potential market value

Return your response in the following JSON format:
{{
    "bid_price": 150.00,  // Your predicted price in euros
    "reasoning": "Your reasoning for this price prediction"
}}

IMPORTANT: Return only valid JSON without any additional text or markdown formatting."""

        # 调用LLM API
        try:
            response = self._call_llm_api(prompt, images_base64)

            # 解析响应
            result = json.loads(response)
            bid_price = float(result.get('bid_price', 0))
            bid_price = max(bid_price, 0)  # 确保非负

            result['bid_price'] = bid_price

            # 简化的决策历史记录
            self.decision_history.append({
                'item_index': item_index,
                'bid_price': bid_price,
                'reasoning': result.get('reasoning', '')
            })

            return result

        except Exception as e:
            print(f"Error predicting price for item {item_index}: {e}")
            return {
                'bid_price': 0.0,
                'reasoning': f'Error occurred: {str(e)}'
            }

    def _call_llm_api(self, prompt: str, images_base64: list, max_retries: int = 5) -> str:
        """
        调用LLM API（带重试机制）- 使用OpenRouter

        Args:
            prompt: 文本prompt
            images_base64: base64编码的图片列表
            max_retries: 最大重试次数

        Returns:
            LLM响应文本
        """
        import time
        from openai import OpenAI

        for attempt in range(max_retries):
            try:
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
                        wait_time = 2 ** (attempt + 1)
                        print(f"[Retry {attempt + 1}/{max_retries}] 等待 {wait_time}秒后重试...")
                        time.sleep(wait_time)
                        continue
                    return '{"error": "API returned empty content"}'

                # 成功获取内容，返回
                return content

            except Exception as e:
                error_msg = str(e)

                # 检查是否是速率限制或负载错误，需要重试
                if 'rate' in error_msg.lower() or 'limit' in error_msg.lower() or '负载' in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** (attempt + 1)
                        print(f"[Retry {attempt + 1}/{max_retries}] API速率限制，等待 {wait_time}秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return f'{{"error": "API rate limit after {max_retries} retries: {error_msg}"}}'

                # 其他错误也重试
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[Retry {attempt + 1}/{max_retries}] API调用异常: {e}，等待 {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    import traceback
                    error_detail = traceback.format_exc()
                    print(f"API调用错误 (最终失败): {error_detail}")
                    return f'{{"error": "API call failed after {max_retries} retries: {str(e)}"}}'

        # 所有重试都失败
        return '{"error": "All retries failed"}'

    def reset(self):
        """重置agent状态，开始新的episode"""
        self.decision_history = []

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
            'decision_history': self.decision_history
        }


def run_auction_episode(agent: SigelLLMAuction,
                       episode: Dict,
                       output_dir: str = None) -> Dict:
    """
    运行一个完整的拍卖episode

    Args:
        agent: SigelLLMAuction实例
        episode: 从dataset.create_episode()返回的episode字典
        output_dir: 输出目录，用于保存结果

    Returns:
        包含所有决策和结果的字典
    """
    # 重置agent
    agent.reset()

    items = episode['items']
    num_items = len(items)

    print(f"\n{'='*60}")
    print(f"开始Episode {episode['group_id']}")
    print(f"Items: {num_items}")
    print(f"{'='*60}\n")

    # 逐个item获取价格预测
    print("开始逐个预测价格...\n")

    predictions = []

    for idx, item in enumerate(items, 1):
        print(f"Item {idx}/{num_items} (Folder ID: {item['folder_id']}):")

        # 获取价格预测
        prediction = agent.predict_price(
            item_images=item['images'],
            item_index=idx
        )

        bid_price = prediction['bid_price']
        actual_price = item['price']

        print(f"  实际价格: €{actual_price:.2f}")
        print(f"  预测价格: €{bid_price:.2f}")
        print(f"  理由: {prediction['reasoning'][:80]}...")

        predictions.append({
            'item_index': idx,
            'folder_id': item['folder_id'],
            'actual_price': actual_price,
            'bid_price': bid_price,
            'prediction': prediction
        })

        print()

    # 总结
    total_predicted = sum(p['bid_price'] for p in predictions)
    print(f"{'='*60}")
    print("Episode总结:")
    print(f"  总预测价格: €{total_predicted:.2f}")
    print(f"  实际总价: €{episode['total_price']:.2f}")
    print(f"{'='*60}\n")

    # 保存结果
    result = {
        'episode_info': {
            'group_id': episode['group_id'],
            'total_price': episode['total_price'],
            'num_items': num_items
        },
        'predictions': predictions,
        'summary': agent.get_decision_summary()
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
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
