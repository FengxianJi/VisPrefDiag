"""
Drop-in replacement for SigelLLMAuction that uses local Qwen3-VL-4B-Thinking.

Compatible with:
  - run_Single-agentLLM_score  (predict_score)
  - run_Single-agent           (predict_price)
  - run_Single-agent-Budgeted  (create_contact_sheet, get_preference_ranking, predict_price)
  - run_Single-agent_pair      (rank_items)
"""

import json
import math
import os
import re
import io
import base64
from typing import Dict, List, Tuple

from PIL import Image

from model_loader import get_qwen_model


class SigelLLMAuction:
    """Local Qwen3-VL-4B inference agent (API-compatible with original SigelLLMAuction)."""

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        model_name: str = "qwen3-vl-4b-thinking",
        temperature: float = 0.7,
    ):
        # api_key / api_base are ignored (kept for interface compatibility)
        self.model_name = model_name
        self.temperature = temperature
        self._model = get_qwen_model()

        # histories
        self.score_history: list = []
        self.decision_history: list = []
        self.ranking_history: list = []
        self.preference_sequence: list = []

    # ── public methods ───────────────────────────────────────────────────

    def predict_score(self, item_images: List[str], item_index: int) -> Dict:
        """Rate artwork 0-100.  Compatible with Single-agentLLM_score."""
        if not item_images:
            return {"score": 50.0, "reasoning": "No images available"}

        prompt = (
            f"You are an art expert evaluating artwork #{item_index}.\n\n"
            "Please rate this artwork on a scale of 0-100 based on:\n"
            "1. Artistic quality and technique\n"
            "2. Composition and aesthetics\n"
            "3. Creativity and originality\n"
            "4. Overall visual appeal\n"
            "5. Emotional impact\n\n"
            "Scoring guidelines:\n"
            "- 90-100: Exceptional, museum-quality artwork\n"
            "- 80-89: Excellent, professional-level work\n"
            "- 70-79: Good quality with notable strengths\n"
            "- 60-69: Above average, competent execution\n"
            "- 50-59: Average, meets basic standards\n"
            "- 40-49: Below average, noticeable issues\n"
            "- 0-39: Poor quality or significant flaws\n\n"
            "Return your response in the following JSON format:\n"
            '{"score": <number>, "reasoning": "<explanation>"}\n\n'
            "IMPORTANT: Return only valid JSON without any additional text or markdown formatting."
        )

        try:
            raw = self._call_local(prompt, images=item_images)
            result = _parse_json(raw)
            score = float(result.get("score", 50))
            score = max(0.0, min(100.0, score))
            result["score"] = score
            self.score_history.append(
                {"item_index": item_index, "score": score, "reasoning": result.get("reasoning", "")}
            )
            return result
        except Exception as e:
            print(f"Error predicting score for item {item_index}: {e}")
            return {"score": 50.0, "reasoning": f"Error: {e}"}

    def predict_price(
        self,
        item_images: List[str],
        item_index: int,
        budget: float = None,
        remaining_budget: float = None,
        num_remaining_items: int = None,
    ) -> Dict:
        """Predict price.  Compatible with Single-agent and Single-agent-Budgeted."""
        if not item_images:
            return {"bid_price": 0.0, "reasoning": "No images available"}

        # build prompt
        if budget is not None and remaining_budget is not None:
            # budgeted mode
            preference_info = ""
            if self.preference_sequence and item_index <= len(self.preference_sequence):
                rank = (
                    self.preference_sequence.index(item_index) + 1
                    if item_index in self.preference_sequence
                    else "Unknown"
                )
                preference_info = f"\nBased on your initial preference ranking, this artwork was ranked #{rank}."

            history_text = self._format_decision_history()

            prompt = (
                f"You are participating in an art auction. Here is your current situation:\n\n"
                f"Total Budget: \u20ac{budget:.2f}\n"
                f"Remaining Budget: \u20ac{remaining_budget:.2f}\n"
                f"Current Item: #{item_index}\n"
                f"Remaining Items to bid on: {num_remaining_items} (including this one)\n"
                f"{preference_info}\n\n"
                f"{history_text}\n\n"
                "Above shows the artwork you need to bid on now.\n\n"
                "Please provide your bid price for this artwork. Consider:\n"
                "1. Your initial preference for this artwork\n"
                "2. Your remaining budget\n"
                "3. Number of remaining artworks\n"
                "4. Your previous decisions\n"
                "5. Balancing preference with budget constraints\n\n"
                "You can bid any amount from \u20ac0 to your remaining budget.\n\n"
                "Return your response in the following JSON format:\n"
                '{"bid_price": <number>, "reasoning": "<your reasoning>"}\n\n'
                "IMPORTANT: Return only valid JSON without any additional text or markdown formatting."
            )
        else:
            # simple price prediction
            prompt = (
                f"You are an art expert evaluating artwork #{item_index} in an auction.\n\n"
                "Please predict the fair market price for this artwork based on:\n"
                "1. Artistic quality and technique\n"
                "2. Composition and aesthetics\n"
                "3. Style and originality\n"
                "4. Potential market value\n\n"
                "Return your response in the following JSON format:\n"
                '{"bid_price": <number>, "reasoning": "<your reasoning>"}\n\n'
                "IMPORTANT: Return only valid JSON without any additional text or markdown formatting."
            )

        try:
            raw = self._call_local(prompt, images=item_images)
            result = _parse_json(raw)
            bid_price = float(result.get("bid_price", 0))
            bid_price = max(bid_price, 0)
            if remaining_budget is not None:
                bid_price = min(bid_price, remaining_budget)
            result["bid_price"] = bid_price

            self.decision_history.append(
                {
                    "item_index": item_index,
                    "bid_price": bid_price,
                    "reasoning": result.get("reasoning", ""),
                    "remaining_budget_before": remaining_budget,
                    "remaining_budget_after": (remaining_budget - bid_price) if remaining_budget is not None else None,
                }
            )
            return result
        except Exception as e:
            print(f"Error predicting price for item {item_index}: {e}")
            return {"bid_price": 0.0, "reasoning": f"Error: {e}"}

    def get_preference_ranking(self, contact_sheet: Image.Image, budget: float, num_items: int) -> Dict:
        """Get preference ranking from contact sheet.  Compatible with Single-agent-Budgeted."""
        prompt = (
            f"You are participating in an art auction. You have been given a budget of \u20ac{budget:.2f} "
            f"to bid on {num_items} artworks.\n\n"
            f"Above is a contact sheet showing all {num_items} candidate artworks in this auction round.\n\n"
            "Please analyze these artworks and provide your preference ranking from most to least preferred. "
            "Consider artistic quality, style, composition, and potential value.\n\n"
            "Return your response in the following JSON format:\n"
            "{\n"
            f'    "preference_ranking": [1, 2, 3, ...],\n'
            '    "reasoning": "Your reasoning for this ranking"\n'
            "}\n\n"
            "IMPORTANT: Return only valid JSON without any additional text or markdown formatting."
        )

        try:
            raw = self._call_local(prompt, images=[contact_sheet])
            result = _parse_json(raw)
            if "preference_ranking" not in result:
                result["preference_ranking"] = list(range(1, num_items + 1))
            self.preference_sequence = result.get("preference_ranking", [])
            return result
        except Exception as e:
            print(f"Error getting preference ranking: {e}")
            return {
                "preference_ranking": list(range(1, num_items + 1)),
                "reasoning": f"Error: {e}",
            }

    def rank_items(self, all_items: List[Dict]) -> Dict:
        """Rank all items.  Compatible with Single-agent_pair."""
        if not all_items:
            return {"ranking": [], "reasoning": "No items", "scores": []}

        num_items = len(all_items)

        prompt = (
            f"You are an art expert evaluating {num_items} artworks in an auction.\n\n"
            f"Below are {num_items} artworks (numbered 1 to {num_items}). "
            "Each artwork is shown with all its images.\n\n"
            "Please rank these artworks from BEST to WORST based on:\n"
            "1. Artistic quality and technique\n"
            "2. Composition and aesthetics\n"
            "3. Creativity and originality\n"
            "4. Overall visual appeal\n"
            "5. Potential market value\n\n"
            "Return your response in the following JSON format:\n"
            "{\n"
            '    "ranking": [<best>, ..., <worst>],\n'
            '    "reasoning": "Brief explanation of your ranking",\n'
            '    "scores": [<score1>, <score2>, ...]\n'
            "}\n\n"
            "IMPORTANT:\n"
            f"- The ranking array must contain all numbers from 1 to {num_items}, each exactly once\n"
            "- The first number is the BEST artwork, the last is the WORST\n"
            "- Return only valid JSON without any additional text or markdown formatting."
        )

        # Collect all images from each item (grouped by item)
        image_groups = []
        for idx, item in enumerate(all_items, 1):
            if item.get("images"):
                image_groups.append(item["images"])
            else:
                image_groups.append([])

        try:
            raw = self._call_local(prompt, images=image_groups, image_labels=True, max_new_tokens=2048)
            result = _parse_json(raw)
            ranking = result.get("ranking", [])

            if len(ranking) != num_items or set(ranking) != set(range(1, num_items + 1)):
                print(f"[WARNING] Invalid ranking {ranking}, using default")
                ranking = list(range(1, num_items + 1))
            result["ranking"] = ranking

            self.ranking_history.append(
                {"num_items": num_items, "ranking": ranking, "reasoning": result.get("reasoning", ""), "scores": result.get("scores", [])}
            )
            return result
        except Exception as e:
            print(f"Error ranking items: {e}")
            return {"ranking": list(range(1, num_items + 1)), "reasoning": f"Error: {e}", "scores": [50] * num_items}

    # ── contact sheet ────────────────────────────────────────────────────

    def create_contact_sheet(
        self,
        image_paths: List[str],
        thumbnail_size: Tuple[int, int] = (300, 300),
        grid_cols: int = None,
        output_path: str = None,
    ) -> Image.Image:
        num_images = len(image_paths)
        if grid_cols is None:
            grid_cols = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_cols)

        canvas_width = grid_cols * thumbnail_size[0]
        canvas_height = grid_rows * thumbnail_size[1]
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                continue
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * thumbnail_size[0]
            y = row * thumbnail_size[1]
            img = Image.open(img_path)
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            canvas.paste(img, (x, y))

        if output_path:
            canvas.save(output_path)
        return canvas

    # ── utilities ────────────────────────────────────────────────────────

    def image_to_base64(self, image) -> str:
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode()
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def reset(self):
        self.score_history.clear()
        self.decision_history.clear()
        self.ranking_history.clear()
        self.preference_sequence.clear()

    def get_score_summary(self) -> Dict:
        if not self.score_history:
            return {}
        scores = [s["score"] for s in self.score_history]
        return {
            "total_evaluations": len(self.score_history),
            "average_score": round(sum(scores) / len(scores), 2),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_history": self.score_history,
        }

    def get_decision_summary(self) -> Dict:
        if not self.decision_history:
            return {}
        total_spent = sum(d["bid_price"] for d in self.decision_history)
        return {
            "total_decisions": len(self.decision_history),
            "total_spent": total_spent,
            "num_positive_bids": len([d for d in self.decision_history if d["bid_price"] > 0]),
            "decision_history": self.decision_history,
            "preference_sequence": self.preference_sequence,
        }

    def get_ranking_summary(self) -> Dict:
        if not self.ranking_history:
            return {}
        return {"total_rankings": len(self.ranking_history), "ranking_history": self.ranking_history}

    # ── private ──────────────────────────────────────────────────────────

    def _call_local(self, prompt: str, images=None, image_labels: bool = False, max_new_tokens: int = None) -> str:
        """
        Build Qwen-format messages and call the local model.

        images: list of file paths (str) or PIL Images.
        image_labels: if True, interleave "[Artwork N]" text between images.
        """
        content: list = []

        if images and not image_labels:
            # single-image or multi-image mode: all images first, then text
            for img in images:
                content.append(_make_image_entry(img))
            content.append({"type": "text", "text": prompt})
        elif images and image_labels:
            # multi-item ranking mode: text first, then each item's images with label
            # images can be a list of lists (grouped by item) or a flat list
            content.append({"type": "text", "text": prompt})
            for idx, img_group in enumerate(images, 1):
                if isinstance(img_group, list):
                    for img in img_group:
                        content.append(_make_image_entry(img))
                else:
                    content.append(_make_image_entry(img_group))
                content.append({"type": "text", "text": f"[Artwork {idx}]"})
        else:
            content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        return self._model.generate(messages, max_new_tokens=max_new_tokens or 512, temperature=self.temperature)

    def _format_decision_history(self) -> str:
        if not self.decision_history:
            return "This is your first bid in this auction round."
        lines = ["Your previous decisions in this auction:"]
        for d in self.decision_history:
            lines.append(
                f"  Item #{d['item_index']}: Bid \u20ac{d['bid_price']:.2f} "
                f"(Remaining: \u20ac{d['remaining_budget_after']:.2f})"
            )
        return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────────

def _make_image_entry(img) -> dict:
    """Create a Qwen-format image content entry."""
    if isinstance(img, str):
        return {"type": "image", "image": img}
    else:
        # PIL Image object
        return {"type": "image", "image": img}


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict:
    """Robustly extract JSON from model output."""
    text = text.strip()
    # direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # from markdown block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # first {...}
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from model output: {text[:200]}...")
