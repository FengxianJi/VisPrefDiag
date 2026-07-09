import re
import os
import sys
from typing import List, Dict
from langchain.prompts import PromptTemplate
# Import ChatOpenAI individually to avoid broken modules
try:
    from langchain.chat_models.openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models.openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field
from collections import defaultdict
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import random
import inflect

# 引入配置
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from config import AuctionConfig as cfg
except ImportError:
    # 如果无法导入配置，使用默认值
    class cfg:
        ENABLE_DISCOUNT = False
        MIN_MARKUP_PCT = 0.1
        USE_CONTACT_SHEET = True
        THUMBNAIL_SIZE = (200, 200)

from .bidder_base import Bidder
from .human_bidder import HumanBidder
from .item_base import Item
from .prompt_base import PARSE_BID_INSTRUCTION

p = inflect.engine()


class Auctioneer(BaseModel):
    """
    拍卖师类

    负责管理拍卖流程、物品展示、竞价规则
    支持原版模式和图像评估模式
    """

    # 拍卖机制参数（直接设置默认值）
    enable_discount: bool = False  # 流拍降价（图像评估模式不使用）
    min_markup_pct: float = 0.1     # 最小加价幅度（默认10%）

    # 拍卖状态
    items: List[Item] = []
    cur_item: Item = None
    highest_bidder: Bidder = None
    highest_bid: int = -1
    bidding_history: Dict = Field(default_factory=lambda: defaultdict(list))
    items_queue: List[Item] = []
    auction_logs: Dict = Field(default_factory=lambda: defaultdict(list))

    # 价格与预算
    openai_cost: float = 0
    prev_round_max_bid: int = -1
    min_bid: int = 0
    fail_to_sell: bool = False
    unified_budget: float = 0.0  # 统一预算（图像评估模式）

    # 图像相关
    contact_sheet_path: str = None  # Contact sheet 路径

    class Config:
        arbitrary_types_allowed = True
    
    def init_items(self, items: List[Item]):
        """
        初始化拍卖物品

        Args:
            items: 物品列表
        """
        for item in items:
            # reset discounted price
            item.reset_price()
        self.items = items
        self.items_queue = items.copy()

        # 计算统一预算（默认启用）
        self.unified_budget = self.calculate_unified_budget(items)

    def calculate_unified_budget(self, items: List[Item]) -> float:
        """
        计算统一预算

        Args:
            items: 物品列表

        Returns:
            统一预算金额 = 所有物品起拍价之和 × BUDGET_RATIO
        """
        return cfg.get_unified_budget(items)

    def generate_contact_sheet(self, items: List[Item] = None, output_path: str = None) -> str:
        """
        生成 Contact Sheet（所有物品缩略图的拼接图）

        Args:
            items: 物品列表（默认使用 self.items）
            output_path: 输出路径（默认使用临时目录）

        Returns:
            Contact sheet 图片路径
        """
        if items is None:
            items = self.items

        # 如果不启用 contact sheet，返回 None
        if not cfg.USE_CONTACT_SHEET:
            return None

        try:
            from PIL import Image
            import math
        except ImportError:
            print("Warning: PIL not installed. Cannot generate contact sheet.")
            return None

        # 创建临时目录
        if output_path is None:
            temp_dir = cfg.TEMP_DIR
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, 'contact_sheet.jpg')

        # 过滤出有缩略图的物品
        items_with_thumb = [item for item in items if item.has_thumbnail()]

        if len(items_with_thumb) == 0:
            print("Warning: No items with thumbnails found.")
            return None

        # 计算网格尺寸
        n = len(items_with_thumb)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        # 缩略图尺寸
        thumb_width, thumb_height = cfg.THUMBNAIL_SIZE

        # 创建空白画布
        sheet = Image.new('RGB', (cols * thumb_width, rows * thumb_height), 'white')

        # 拼接缩略图
        for idx, item in enumerate(items_with_thumb):
            try:
                img = Image.open(item.thumbnail_path).resize((thumb_width, thumb_height))
                x = (idx % cols) * thumb_width
                y = (idx // cols) * thumb_height
                sheet.paste(img, (x, y))
            except Exception as e:
                print(f"Warning: Failed to load thumbnail for {item.name}: {e}")
                continue

        # 保存
        sheet.save(output_path)
        self.contact_sheet_path = output_path
        return output_path

    def pre_auction_observation(self, bidders: List):
        """
        Phase 0: Pre-auction global observation
        让所有judge先观察所有作品，建立全局偏好排序
        """
        import ujson as json

        print("\n" + "="*60)
        print("Phase 0: Pre-evaluation Global Observation")
        print("="*60)

        # 生成contact sheet（如果还没有）
        if self.contact_sheet_path is None:
            self.generate_contact_sheet()

        if self.contact_sheet_path is None:
            print("Warning: No contact sheet available, skipping pre-evaluation observation")
            return

        # 每个judge观察并建立全局偏好
        for bidder in bidders:
            if bidder.model_name == 'rule':
                bidder.global_preference = list(range(len(self.items)))
                continue

            prompt = f"""You will participate in an art evaluation with {len(self.items)} artworks.
Above is a contact sheet showing all {len(self.items)} artworks you will evaluate.

Please carefully observe and analyze these artworks, then rank them from most to least preferred based on artistic quality, style, composition, and potential value.

Return your response in the following JSON format:
{{
    "preference_ranking": [2, 0, 4, 1, 3, 5],  // List of artwork indices (0 to {len(self.items)-1}) from most to least preferred
    "reasoning": "Your reasoning for this ranking"
}}

IMPORTANT: Return only valid JSON without any additional text or markdown formatting."""

            try:
                messages = [
                    SystemMessage(content=bidder.system_message),
                    HumanMessage(content=prompt)
                ]
                result = bidder._run_llm_standalone(messages)

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
                    bidder.global_preference = list(range(len(self.items)))
                    print(f"  {bidder.name}: No preference_ranking found, using default")

            except Exception as e:
                print(f"  {bidder.name}: Error in pre-evaluation observation: {e}")
                bidder.global_preference = list(range(len(self.items)))

        print("="*60 + "\n")

    def summarize_items_info(self):
        desc = ''
        for item in self.items:
            desc += f"- {item.get_desc()}\n"
        return desc.strip()
    
    def present_item(self):
        cur_item = self.items_queue.pop(0)
        self.cur_item = cur_item
        return cur_item
    
    def shuffle_items(self):
        random.shuffle(self.items)
        self.items_queue = self.items.copy()
    
    def record_bid(self, bid_info: dict, bid_round: int):
        '''
        Save the bidding history for each round, log the highest bidder and highest bidding
        '''
        # bid_info: {'bidder': xxx, 'bid': xxx, 'raw_msg': xxx}
        self.bidding_history[bid_round].append(bid_info)
        for hist in self.bidding_history[bid_round]:
            if hist['bid'] > 0:
                if self.highest_bid < hist['bid']:
                    self.highest_bid = hist['bid']
                    self.highest_bidder = hist['bidder']
                elif self.highest_bid == hist['bid']:
                    # random if there's a tie
                    self.highest_bidder = random.choice([self.highest_bidder, hist['bidder']])
        self.auction_logs[f"{self.cur_item.get_desc()}"].append(
            {'bidder': bid_info['bidder'], 
             'bid': bid_info['bid'], 
             'bid_round': bid_round})

    def _biddings_to_string(self, bid_round: int):
        '''
        Return a string that summarizes the bidding history in a round
        '''
        # bid_hist_text = '' if bid_round == 0 else f'- {self.highest_bidder}: ${self.highest_bid}\n'
        bid_hist_text = ''
        for js in self.bidding_history[bid_round]:
            if js['bid'] < 0:
                bid_hist_text += f"- {js['bidder']} abstained\n"
            else:
                bid_hist_text += f"- {js['bidder']}: score {js['bid']}\n"
        return bid_hist_text.strip()
    
    def all_bidding_history_to_string(self):
        bid_hist_text = ''
        for bid_round in self.bidding_history:
            bid_hist_text += f"Round {bid_round}:\n{self._biddings_to_string(bid_round)}\n\n"
        return bid_hist_text.strip()

    def ask_for_bid(self, bid_round: int):
        '''
        Ask for score, return the message to be sent to judges
        '''
        round_info = f" (Round {bid_round + 1}/1)"
        if self.highest_bidder is None:
            if bid_round > 0:
                msg = f"No scores received yet for {self.cur_item.name}. Please provide your score (0-100).{round_info}"
            else:
                remaining_items = [self.cur_item.name] + [item.name for item in self.items_queue]
                msg = f"Attention, judges! {len(remaining_items)} item(s) left, they are: {', '.join(remaining_items)}.\n\nNow, please score {self.cur_item}. Evaluate this artwork on a scale of 0-100 based on your aesthetic judgment.{round_info}"
        else:
            bidding_history = self._biddings_to_string(bid_round - 1)
            msg = f"Thank you! This is the {p.ordinal(bid_round)} round of scoring for this item:\n{bidding_history}\n\nNow we have a score of {self.highest_bid} from {self.highest_bidder.name} for {self.cur_item.name}. Does anyone want to give a higher score?{round_info}"
        return msg
    
    def ask_for_rebid(self, fail_msg: str, bid_price: int):
        return f"Your bid of ${bid_price} failed, because {fail_msg}: You must reconsider your bid."

    def get_hammer_msg(self):
        """
        生成评分结果消息
        """
        if self.highest_bidder is None:
            return f"Since no one scored {self.cur_item.name}, we'll move on to the next item."
        else:
            if self.cur_item.is_image_mode:
                return f"Scored! {self.cur_item} — highest score {self.highest_bid} by {self.highest_bidder}!"
            else:
                return f"Scored! {self.cur_item} — highest score {self.highest_bid} by {self.highest_bidder}! The true value for {self.cur_item} is ${self.cur_item.true_value}."

    def check_hammer(self, bid_round: int):
        # check if the item is sold
        self.fail_to_sell = False
        num_bid = self._num_bids_in_round(bid_round)

        # Hard limit: force hammer after 2 rounds (bid_round is 0-indexed, so >= 1 means 2nd round done)
        if bid_round >= 1:
            if self.highest_bidder is None:
                self.fail_to_sell = True
            return True

        # highest_bidder has already been updated in record_bid().
        # so when num_bid == 0 & highest_bidder is None, it means no one bid on this item
        if self.highest_bidder is None:
            if num_bid == 0:
                # failed to sell, as there is no highest bidder
                self.fail_to_sell = True
                if self.enable_discount and bid_round < 3:
                    # lower the starting price by 50%. discoutn only applies to the first 3 rounds
                    self.cur_item.lower_price(0.5)
                    is_sold = False
                else:
                    is_sold = True
            else:
                # won't happen
                raise ValueError(f"highest_bidder is None but num_bid is {num_bid}")
        else:
            if self.prev_round_max_bid < 0 and num_bid == 1:
                # only one bidder in the first round 
                is_sold = True
            else:
                self.prev_round_max_bid = self.highest_bid
                is_sold = self._num_bids_in_round(bid_round) == 0
        return is_sold
    
    def _num_bids_in_round(self, bid_round: int):
        # check if there is no bid in the current round
        cnt = 0
        for hist in self.bidding_history[bid_round]:
            if hist['bid'] > 0:
                cnt += 1
        return cnt

    def hammer_fall(self):
        """
        落锤：确认成交并记录日志
        """
        # 打印成交信息
        if self.cur_item.is_image_mode:
            # 图像评估模式：不显示 true_value
            print(f'* Sold! {self.cur_item} goes to {self.highest_bidder} at ${self.highest_bid}.')
            bid_info = f"{self.highest_bid}"
        else:
            # 原版模式：显示 true_value
            print(f'* Sold! {self.cur_item} (${self.cur_item.true_value}) goes to {self.highest_bidder} at ${self.highest_bid}.')
            bid_info = f"{self.highest_bid} (${self.cur_item.true_value})"

        # 记录到拍卖日志
        self.auction_logs[f"{self.cur_item.get_desc()}"].append({
            'bidder': self.highest_bidder,
            'bid': bid_info,
            'bid_round': 'Hammer price' if self.cur_item.is_image_mode else 'Hammer price (true value)'
        })

        # 重置状态
        self.cur_item = None
        self.highest_bidder = None
        self.highest_bid = -1
        self.bidding_history = defaultdict(list)
        self.prev_round_max_bid = -1
        self.fail_to_sell = False

    def end_auction(self):
        return len(self.items_queue) == 0
    
    def gather_all_status(self, bidders: List[Bidder]):
        status = {}
        for bidder in bidders:
            status[bidder.name] = {
                'profit': bidder.profit, 
                'items_won': bidder.items_won
            }
        return status

    def parse_bid(self, text: str):
        # For local models, use direct regex parsing to avoid API calls
        # Priority: Extract any number from the response

        # Check for withdrawal/abstain first
        if '-1' in text or 'withdraw' in text.lower() or 'abstain' in text.lower():
            return -1

        # Try multiple score extraction patterns (most specific to least specific)
        score_patterns = [
            r'(?:I score|score:?|scoring)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # "I score 75", "score: 75"
            r'(\d+)\s*(?:/\s*100|out of 100)',  # "75/100", "75 out of 100"
            r'\b(\d+(?:\.\d+)?)\b',  # Any standalone number
        ]

        for pattern in score_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Filter to 0-100 range (likely scores, not dollar amounts)
                for match in matches:
                    try:
                        score = float(match)
                        if 0 <= score <= 100:
                            return int(score)
                    except ValueError:
                        continue

        # Fallback: extract any number (for $ amounts if this is still bid mode)
        bid_numbers = re.findall(r'\$?\d+', text.replace(',', ''))
        if bid_numbers:
            return int(bid_numbers[-1].replace('$', ''))

        # Last resort: print warning and return None (will trigger rebid)
        print(f'[parse_bid] WARNING: Could not parse response: {text[:100]}...')
        return None

    def log(self, bidder_personal_reports: list = [], show_model_name=True):
        ''' example
        Apparatus H, starting at $1000.

        1st bid:
        Bidder 1 (gpt-3.5-turbo-16k-0613): $1200
        Bidder 2 (gpt-3.5-turbo-16k-0613): $1100
        Bidder 3 (gpt-3.5-turbo-16k-0613): Withdrawn
        Bidder 4 (gpt-3.5-turbo-16k-0613): $1200
        
        2nd bid:
        Bidder 1 (gpt-3.5-turbo-16k-0613): Withdrawn
        Bidder 2 (gpt-3.5-turbo-16k-0613): Withdrawn
        
        Hammer price:
        Bidder 4 (gpt-3.5-turbo-16k-0613): $1200
        '''
        markdown_output = "## Auction Log\n\n"
        for i, (item, bids) in enumerate(self.auction_logs.items()):
            markdown_output += f"### {i+1}. {item}\n\n"
            cur_bid_round = -1
            for i, bid in enumerate(bids):
                if bid['bid_round'] != cur_bid_round:
                    cur_bid_round = bid['bid_round']
                    if isinstance(bid['bid_round'], int):
                        markdown_output += f"\n#### {p.ordinal(bid['bid_round']+1)} bid:\n\n"
                    else:
                        markdown_output += f"\n#### {bid['bid_round']}:\n\n"
                bid_price = f"${bid['bid']}" if bid['bid'] != -1 else 'Withdrew'
                if isinstance(bid['bidder'], Bidder) or isinstance(bid['bidder'], HumanBidder):
                    if show_model_name:
                        markdown_output += f"* {bid['bidder']} ({bid['bidder'].model_name}): {bid_price}\n"
                    else:
                        markdown_output += f"* {bid['bidder']}: {bid_price}\n"
                else:
                    markdown_output += f"* None bid\n"
            markdown_output += "\n"
        
        if len(bidder_personal_reports) != 0:
            markdown_output += f"\n## Personal Report"
            for report in bidder_personal_reports:
                markdown_output += f"\n\n{report}"
        return markdown_output.strip()
    
    def finish_auction(self):
        self.auction_logs = defaultdict(list)
        self.cur_item = None
        self.highest_bidder = None
        self.highest_bid = -1
        self.bidding_history = defaultdict(list)
        self.items_queue = []
        self.items = []
        self.prev_round_max_bid = -1
        self.fail_to_sell = False
        self.min_bid = 0
        