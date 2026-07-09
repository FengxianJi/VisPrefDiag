import os
import json
import base64
from typing import List, Dict


class SigelLLMAuction:
    """
    单个LLM拍卖agent（排序模式）

    功能：
    1. 将所有item的图片一起展示给LLM
    2. LLM根据美学价值对图片进行排序
    3. 记录排序结果
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

        # 排序历史
        self.ranking_history = []

    def image_to_base64(self, image_path: str) -> str:
        """将图片文件转换为base64字符串"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    def rank_items(self, all_items: List[Dict]) -> Dict:
        """
        将所有item的图片一起展示，让LLM进行排序

        Args:
            all_items: 所有item的列表，每个item包含 {'images': [...], 'folder_id': ...}

        Returns:
            包含排序结果和理由的字典:
            {
                'ranking': [2, 5, 1, 3, 4],  # 排序结果（1-indexed）
                'reasoning': '...',           # 排序理由
                'scores': [85, 92, 78, ...]   # 可选：每个item的评分
            }
        """
        if not all_items:
            return {
                'ranking': [],
                'reasoning': 'No items to rank',
                'scores': []
            }

        num_items = len(all_items)

        # 为每个item收集所有图片，转换为base64（按item分组）
        item_images_base64 = []
        item_info = []

        for idx, item in enumerate(all_items, 1):
            if item['images']:
                imgs_b64 = [self.image_to_base64(img) for img in item['images']]
                item_images_base64.append(imgs_b64)
                item_info.append({
                    'index': idx,
                    'folder_id': item.get('folder_id', f'item_{idx}'),
                    'num_images': len(item['images'])
                })
            else:
                item_images_base64.append([])
                item_info.append({
                    'index': idx,
                    'folder_id': item.get('folder_id', f'item_{idx}'),
                    'num_images': 0
                })

        # 构建排序prompt
        prompt = f"""You are an art expert evaluating {num_items} artworks in an auction.

Below are {num_items} artworks (numbered 1 to {num_items}). Each artwork is shown with all its images.

Please rank these artworks from BEST to WORST based on:
1. Artistic quality and technique
2. Composition and aesthetics
3. Creativity and originality
4. Overall visual appeal
5. Potential market value

Return your response in the following JSON format:
{{
    "ranking": [2, 5, 1, 3, 4],  // Array of artwork numbers from best to worst
    "reasoning": "Brief explanation of your ranking criteria and key observations",
    "scores": [85, 92, 78, 88, 80]  // Optional: scores (0-100) for each artwork in original order (1, 2, 3, 4, 5)
}}

IMPORTANT:
- The ranking array must contain all numbers from 1 to {num_items}, each appearing exactly once
- The first number in ranking is the BEST artwork, the last is the WORST
- Return only valid JSON without any additional text or markdown formatting."""

        # 调用LLM API
        try:
            response = self._call_llm_api_multi_images(prompt, item_images_base64)

            # 解析响应
            result = json.loads(response)
            ranking = result.get('ranking', [])

            # 验证ranking的有效性
            if len(ranking) != num_items or set(ranking) != set(range(1, num_items + 1)):
                print(f"[WARNING] Invalid ranking: {ranking}, using default order")
                ranking = list(range(1, num_items + 1))

            result['ranking'] = ranking

            # 记录到历史
            self.ranking_history.append({
                'num_items': num_items,
                'ranking': ranking,
                'reasoning': result.get('reasoning', ''),
                'scores': result.get('scores', [])
            })

            return result

        except Exception as e:
            print(f"Error ranking items: {e}")
            import traceback
            traceback.print_exc()

            # 返回默认顺序
            return {
                'ranking': list(range(1, num_items + 1)),
                'reasoning': f'Error occurred: {str(e)}',
                'scores': [50] * num_items
            }

    def _call_llm_api_multi_images(self, prompt: str, item_images_base64: List[List[str]], max_retries: int = 5) -> str:
        """
        调用LLM API（多张图片，按item分组）- 使用OpenRouter

        Args:
            prompt: 文本prompt
            item_images_base64: base64编码的图片列表的列表（按item分组）
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

                # 构建内容列表：先添加文本，再按item分组添加图片+标签
                content_list = [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]

                # 按item分组添加所有图片
                for idx, imgs_b64 in enumerate(item_images_base64, 1):
                    for img_b64 in imgs_b64:
                        content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "low"
                            }
                        })
                    # 添加图片标签
                    content_list.append({
                        "type": "text",
                        "text": f"[Artwork {idx}]"
                    })

                # 构建消息
                messages = [
                    {
                        "role": "system",
                        "content": "You are an art expert specializing in auction evaluation and ranking. Analyze artworks objectively based on artistic merit, technique, composition, and market potential."
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
                        "X-Title": "Art Auction Ranking Research",
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

    def _call_llm_api(self, prompt: str, image_base64: str, max_retries: int = 5) -> str:
        """
        调用LLM API（带重试机制）- 使用OpenRouter

        Args:
            prompt: 文本prompt
            image_base64: base64编码的图片
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

                # 构建消息
                messages = [
                    {
                        "role": "system",
                        "content": "You are an art expert participating in art auctions. Analyze artworks and provide price predictions based on artistic quality, composition, and market value."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
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
        self.ranking_history = []

    def get_ranking_summary(self) -> Dict:
        """获取排序摘要"""
        if not self.ranking_history:
            return {}

        return {
            'total_rankings': len(self.ranking_history),
            'ranking_history': self.ranking_history
        }


def run_auction_episode(agent: SigelLLMAuction,
                       episode: Dict,
                       output_dir: str = None) -> Dict:
    """
    运行一个完整的拍卖episode（排序模式）

    Args:
        agent: SigelLLMAuction实例
        episode: 从dataset.create_episode()返回的episode字典
        output_dir: 输出目录，用于保存结果

    Returns:
        包含排序结果的字典
    """
    # 重置agent
    agent.reset()

    items = episode['items']
    num_items = len(items)

    print(f"\n{'='*60}")
    print(f"开始Episode {episode['group_id']} (排序模式)")
    print(f"Items: {num_items}")
    print(f"{'='*60}\n")

    # 准备所有items给LLM排序
    print("将所有artwork一起展示给LLM进行排序...\n")

    # 调用排序方法
    ranking_result = agent.rank_items(items)

    ranking = ranking_result['ranking']
    reasoning = ranking_result['reasoning']
    scores = ranking_result.get('scores', [])

    print(f"排序完成!")
    print(f"\n排序结果 (从最佳到最差):")
    print(f"  {ranking}")
    print(f"\n排序理由:")
    print(f"  {reasoning}")

    if scores:
        print(f"\n每个artwork的评分 (0-100):")
        for idx, score in enumerate(scores, 1):
            print(f"  Artwork {idx}: {score}")

    # 分析排序结果与实际价格的关系
    print(f"\n{'='*60}")
    print("排序 vs 实际价格分析:")
    print(f"{'='*60}\n")

    # 创建排序到价格的映射
    ranking_analysis = []
    for rank_position, item_idx in enumerate(ranking, 1):
        item = items[item_idx - 1]  # item_idx是1-indexed
        actual_price = item['price']
        score = scores[item_idx - 1] if scores else 0

        ranking_analysis.append({
            'rank': rank_position,
            'item_index': item_idx,
            'folder_id': item['folder_id'],
            'actual_price': actual_price,
            'score': score
        })

        print(f"排名 #{rank_position}: Artwork {item_idx} (Folder: {item['folder_id']})")
        print(f"  实际价格: €{actual_price:.2f}")
        if score:
            print(f"  LLM评分: {score}/100")
        print()

    # 计算相关性指标
    # 理想情况：排名越前，价格越高
    # 计算Spearman秩相关系数
    try:
        from scipy.stats import spearmanr
        ranks = [r['rank'] for r in ranking_analysis]
        prices = [r['actual_price'] for r in ranking_analysis]
        correlation, p_value = spearmanr(ranks, prices)

        print(f"{'='*60}")
        print(f"统计分析:")
        print(f"  Spearman秩相关系数: {correlation:.3f}")
        print(f"  P值: {p_value:.4f}")
        if correlation < 0:
            print(f"  解释: 排名越靠前，价格越高 (负相关是预期的)")
        print(f"{'='*60}\n")
    except ImportError:
        print("提示: 安装scipy可以查看相关性分析 (pip install scipy)")
        correlation = None
        p_value = None

    # 保存结果
    result = {
        'episode_info': {
            'group_id': episode['group_id'],
            'total_price': episode['total_price'],
            'num_items': num_items
        },
        'ranking_result': {
            'ranking': ranking,
            'reasoning': reasoning,
            'scores': scores
        },
        'ranking_analysis': ranking_analysis,
        'statistics': {
            'correlation': correlation,
            'p_value': p_value
        } if correlation is not None else {},
        'summary': agent.get_ranking_summary()
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, f"ranking_result_group{episode['group_id']}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {result_path}")

    return result


if __name__ == '__main__':
    # 测试代码
    from dataset import load_dataset

    print("=" * 60)
    print("测试 SigelLLMAuction (排序模式)")
    print("=" * 60)

    # 注意：实际使用时需要设置API key
    print("\n注意: 本测试需要设置OPENAI_API_KEY环境变量")
    print("示例: export OPENAI_API_KEY='your-api-key'")

    # 创建数据集
    json_path = '/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data/Low-price.json'
    dataset = load_dataset(json_path, budget_ratio=1.2, seed=42)

    # 创建episode
    episode = dataset.create_episode()

    print(f"\n将对 {episode['num_items']} 个artworks进行排序")

    # 创建agent（如果没有API key，会返回错误信息）
    agent = SigelLLMAuction(
        model_name="gpt-4o",
        temperature=0.7
    )

    # 运行episode
    output_dir = '/mnt/data/user/fengxian.ji/AUCTION/method/Single-agent_pair/output'
    result = run_auction_episode(agent, episode, output_dir)

    print("\n测试完成！")
