"""
Singleton model loader for Qwen3-VL-8B-Instruct.

Loads the model once and provides inference via generate() and generate_batch().
"""

import os
import re
import sys
import time
import threading
from pathlib import Path

# Model weight path (relative to this file)
_WEIGHT_DIR = str(Path(__file__).resolve().parent.parent / "weight")

# ── NVIDIA library path setup (needed for torch on this machine) ──────────
def _setup_ld_library_path():
    """Add nvidia pip package lib dirs to LD_LIBRARY_PATH if missing."""
    try:
        import importlib
        extra_dirs = []
        for pkg in ("nvidia.cudnn.lib", "nvidia.cuda_cupti.lib"):
            try:
                mod = importlib.import_module(pkg)
                d = os.path.dirname(mod.__file__)
                if d not in os.environ.get("LD_LIBRARY_PATH", ""):
                    extra_dirs.append(d)
            except ImportError:
                pass
        if extra_dirs:
            cur = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = ":".join(extra_dirs) + (":" + cur if cur else "")
    except Exception:
        pass

_setup_ld_library_path()


class QwenModelSingleton:
    """Thread-safe singleton wrapper around Qwen3-VL-8B-Instruct.

    Note: In multiprocessing mode, each process gets its own instance.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.processor = None
        self._loaded = False
        self._process_id = os.getpid()  # Track which process owns this instance

    @classmethod
    def get_instance(cls):
        # Check if we're in a new process (multiprocessing)
        current_pid = os.getpid()
        if cls._instance is None or cls._instance._process_id != current_pid:
            with cls._lock:
                if cls._instance is None or cls._instance._process_id != current_pid:
                    cls._instance = cls()
        return cls._instance

    # ── loading ──────────────────────────────────────────────────────────
    def load_model(self, weight_dir: str = None):
        """Lazy-load model + processor (called automatically on first generate)."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            wd = weight_dir or _WEIGHT_DIR
            print(f"[model_loader] Loading Qwen3-VL-8B-Instruct from {wd} ...")

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                wd,
                dtype=torch.bfloat16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(wd)
            self._loaded = True
            print(f"[model_loader] Model loaded successfully. device={self.model.device}")

    # ── inference ────────────────────────────────────────────────────────
    def generate(
        self,
        messages: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Run inference and return the response text.

        Parameters
        ----------
        messages : list
            OpenAI-style messages.  Each message is a dict with "role" and
            "content".  Content can be a plain string or a list of dicts
            with "type" = "text" | "image".

            Image entries should use:
                {"type": "image", "image": "<file_path>"}
            or PIL Image objects:
                {"type": "image", "image": <PIL.Image>}

        max_new_tokens : int
        temperature : float  (0 = greedy)

        Returns
        -------
        str : model response text.
        """
        self.load_model()
        import torch

        # Normalise messages: ensure content is always a list of dicts
        messages = _normalise_messages(messages)

        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Remove token_type_ids if present (not needed by Qwen3-VL)
        inputs.pop("token_type_ids", None)
        # Move all tensors to model's first device
        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        has_images = image_inputs is not None and len(image_inputs) > 0
        print(f"[generate] input_tokens={input_len}, max_new={max_new_tokens}, "
              f"temp={temperature}, has_images={has_images}", flush=True)

        # Build generation kwargs.
        # Instruct model recommends: temperature=0.7, top_p=0.8, top_k=20
        if temperature > 0:
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.8,
                top_k=20,
                use_cache=True,
            )
        else:
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim prompt tokens
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        gen_len = len(generated_ids[0])
        elapsed = time.time() - t0

        print(f"[generate] done in {elapsed:.1f}s, gen_tokens={gen_len}", flush=True)

        raw_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        # Clear KV cache after each generate to prevent accumulation
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[generate] cache cleared after generation", flush=True)

        return raw_text

    # ── batch inference ───────────────────────────────────────────────────
    def generate_batch(
        self,
        messages_list: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> list:
        """
        Run batched inference on multiple messages and return a list of response texts.

        Parameters
        ----------
        messages_list : list of list
            Each element is an OpenAI-style messages list (same format as generate()).
        max_new_tokens : int
        temperature : float  (0 = greedy)

        Returns
        -------
        list of str : model response texts (one per input).
        """
        if len(messages_list) == 1:
            return [self.generate(messages_list[0], max_new_tokens, temperature)]

        self.load_model()
        import torch
        from qwen_vl_utils import process_vision_info

        # Prepare each sample
        texts = []
        all_image_inputs = []
        for messages in messages_list:
            messages = _normalise_messages(messages)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            texts.append(text)
            image_inputs, video_inputs = process_vision_info(messages)
            if image_inputs:
                all_image_inputs.extend(image_inputs)

        # Use left-padding for batched generation (required for causal LMs)
        orig_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = 'left'

        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.model.device)

        # Restore original padding side
        self.processor.tokenizer.padding_side = orig_padding_side

        batch_size = inputs["input_ids"].shape[0]
        max_input_len = inputs["input_ids"].shape[1]
        print(f"[generate_batch] batch_size={batch_size}, max_input_tokens={max_input_len}, "
              f"max_new={max_new_tokens}, temp={temperature}", flush=True)

        if temperature > 0:
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.8,
                top_k=20,
                use_cache=True,
            )
        else:
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        # Trim prompt tokens: use the padded input length (same for all samples)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        results = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        results = [r.strip() for r in results]

        print(f"[generate_batch] done in {elapsed:.1f}s for {batch_size} samples", flush=True)

        import gc
        torch.cuda.empty_cache()
        gc.collect()

        return results

    # ── cache management ─────────────────────────────────────────────────
    def clear_cache(self):
        """
        Clear KV cache and GPU memory.

        Call this after completing a group to prevent memory accumulation.
        """
        import torch
        if self._loaded and torch.cuda.is_available():
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
            # Reset any model-level past_key_values if they exist
            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None
            print("[model_loader] Cache cleared", flush=True)


# ── helpers ──────────────────────────────────────────────────────────────

def _normalise_messages(messages: list) -> list:
    """Ensure every message's content is a list of typed dicts."""
    out = []
    for msg in messages:
        msg = dict(msg)  # shallow copy
        c = msg.get("content")
        if isinstance(c, str):
            msg["content"] = [{"type": "text", "text": c}]
        out.append(msg)
    return out


# ── module-level accessor ────────────────────────────────────────────────

def get_qwen_model(weight_dir: str = None) -> QwenModelSingleton:
    """Return (and lazily load) the global Qwen3-VL model singleton."""
    inst = QwenModelSingleton.get_instance()
    if not inst._loaded:
        inst.load_model(weight_dir)
    return inst
