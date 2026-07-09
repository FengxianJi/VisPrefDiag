import os
import json
import base64
from typing import List, Dict


class SigelLLMAuction:
    """
    单个LLM拍卖agent（评分模式）

    功能：
    1. 调用LLM API逐个预测图片评分（0-100分）
    2. 记录评分历史
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

        # 评分历史
        self.score_history = []

    def image_to_base64(self, image_path: str) -> str:
        """将图片文件转换为base64字符串"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    def predict_score(self,
                     item_images: List[str],
                     item_index: int) -> Dict:
        """
        展示item的图片，让LLM预测美学评分（0-100分）

        Args:
            item_images: 当前item的所有图片路径
            item_index: 当前item的索引（从1开始）

        Returns:
            包含预测评分和理由的字典
        """
        # 使用第一张图片
        if not item_images:
            return {
                'score': 50.0,
                'reasoning': 'No images available'
            }

        # 将所有图片转换为base64
        images_base64 = [self.image_to_base64(img) for img in item_images]

        # 构建评分prompt
        prompt = f"""You are an art expert evaluating artwork #{item_index}.

Above shows the artwork you need to evaluate.

Please rate this artwork on a scale of 0-100 based on:
1. Artistic quality and technique
2. Composition and aesthetics
3. Creativity and originality
4. Overall visual appeal
5. Emotional impact

Scoring guidelines:
- 90-100: Exceptional, museum-quality artwork
- 80-89: Excellent, professional-level work
- 70-79: Good quality with notable strengths
- 60-69: Above average, competent execution
- 50-59: Average, meets basic standards
- 40-49: Below average, noticeable issues
- 0-39: Poor quality or significant flaws

Return your response in the following JSON format:
{{
    "score": 75.5,  // Your score (0-100, can include decimals)
    "reasoning": "Brief explanation of your score"
}}

IMPORTANT: Return only valid JSON without any additional text or markdown formatting."""

        # 调用LLM API
        try:
            response = self._call_llm_api(prompt, images_base64)

            # 解析响应
            result = json.loads(response)
            score = float(result.get('score', 50))

            # 确保评分在0-100范围内
            score = max(0, min(100, score))

            result['score'] = score

            # 记录评分历史
            self.score_history.append({
                'item_index': item_index,
                'score': score,
                'reasoning': result.get('reasoning', '')
            })

            return result

        except Exception as e:
            print(f"Error predicting score for item {item_index}: {e}")
            return {
                'score': 50.0,
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
                        "content": "You are an art expert specializing in aesthetic evaluation. Analyze artworks objectively and provide scores (0-100) based on artistic quality, composition, technique, creativity, and overall appeal."
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
        self.score_history = []

    def get_score_summary(self) -> Dict:
        """获取评分摘要"""
        if not self.score_history:
            return {}

        scores = [s['score'] for s in self.score_history]
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            'total_evaluations': len(self.score_history),
            'average_score': round(avg_score, 2),
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'score_history': self.score_history
        }


def run_auction_episode(agent: SigelLLMAuction,
                       episode: Dict,
                       output_dir: str = None) -> Dict:
    """
    运行一个完整的拍卖episode（评分模式）

    Args:
        agent: SigelLLMAuction实例
        episode: 从dataset.create_episode()返回的episode字典
        output_dir: 输出目录，用于保存结果

    Returns:
        包含所有评分和结果的字典
    """
    # 重置agent
    agent.reset()

    items = episode['items']
    num_items = len(items)

    print(f"\n{'='*60}")
    print(f"开始Episode {episode['group_id']} (评分模式)")
    print(f"Items: {num_items}")
    print(f"{'='*60}\n")

    # 逐个item获取评分
    print("开始逐个评分artwork...\n")

    evaluations = []

    for idx, item in enumerate(items, 1):
        print(f"Item {idx}/{num_items} (Folder ID: {item['folder_id']}):")

        # 获取评分
        evaluation = agent.predict_score(
            item_images=item['images'],
            item_index=idx
        )

        score = evaluation['score']
        actual_price = item['price']

        print(f"  实际价格: €{actual_price:.2f}")
        print(f"  LLM评分: {score:.1f}/100")
        print(f"  理由: {evaluation['reasoning'][:80]}...")

        evaluations.append({
            'item_index': idx,
            'folder_id': item['folder_id'],
            'actual_price': actual_price,
            'score': score,
            'evaluation': evaluation
        })

        print()

    # 计算统计信息
    scores = [e['score'] for e in evaluations]
    prices = [e['actual_price'] for e in evaluations]

    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"{'='*60}")
    print("Episode总结:")
    print(f"  平均评分: {avg_score:.2f}/100")
    print(f"  最高评分: {max(scores):.1f}/100")
    print(f"  最低评分: {min(scores):.1f}/100")
    print(f"  实际总价: €{episode['total_price']:.2f}")
    print(f"{'='*60}\n")

    # 计算评分与价格的相关性
    try:
        from scipy.stats import pearsonr, spearmanr

        pearson_corr, pearson_p = pearsonr(scores, prices)
        spearman_corr, spearman_p = spearmanr(scores, prices)

        print(f"评分与价格相关性分析:")
        print(f"  Pearson相关系数: {pearson_corr:.3f} (p={pearson_p:.4f})")
        print(f"  Spearman秩相关系数: {spearman_corr:.3f} (p={spearman_p:.4f})")

        if abs(pearson_corr) > 0.5:
            print(f"  解释: 评分与价格有{'正' if pearson_corr > 0 else '负'}相关关系")
        else:
            print(f"  解释: 评分与价格相关性较弱")

        print(f"{'='*60}\n")

        correlation_stats = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p
        }
    except ImportError:
        print("提示: 安装scipy可以查看相关性分析 (pip install scipy)")
        correlation_stats = {}
    except Exception as e:
        print(f"相关性计算出错: {e}")
        correlation_stats = {}

    # 保存结果
    result = {
        'episode_info': {
            'group_id': episode['group_id'],
            'total_price': episode['total_price'],
            'num_items': num_items
        },
        'evaluations': evaluations,
        'statistics': {
            'average_score': avg_score,
            'min_score': min(scores),
            'max_score': max(scores),
            'score_std': (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0,
            **correlation_stats
        },
        'summary': agent.get_score_summary()
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, f"score_result_group{episode['group_id']}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {result_path}")

    return result


if __name__ == '__main__':
    # 测试代码
    from dataset import load_dataset

    print("=" * 60)
    print("测试 SigelLLMAuction (评分模式)")
    print("=" * 60)

    # 注意：实际使用时需要设置API key
    print("\n注意: 本测试需要设置OPENAI_API_KEY环境变量")
    print("示例: export OPENAI_API_KEY='your-api-key'")

    # 创建数据集
    json_path = '/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data/Low-price.json'
    dataset = load_dataset(json_path, budget_ratio=1.2, seed=42)

    # 创建episode
    episode = dataset.create_episode()

    print(f"\n将对 {episode['num_items']} 个artworks进行评分（0-100分）")

    # 创建agent（如果没有API key，会返回错误信息）
    agent = SigelLLMAuction(
        model_name="gpt-4o",
        temperature=0.7
    )

    # 运行episode
    output_dir = '/mnt/data/user/fengxian.ji/AUCTION/method/Single-agentLLM_score/output'
    result = run_auction_episode(agent, episode, output_dir)

    print("\n测试完成！")
