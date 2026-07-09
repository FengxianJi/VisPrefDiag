"""Singleton wrapper for remote UniAPI-backed Qwen3-VL inference."""

import base64
import io
import json
import mimetypes
import os
import re
import threading
import time

import requests


DEFAULT_API_BASE = "https://api.uniapi.io/v1"
DEFAULT_MODEL_NAME = "qwen3-vl-30b-a3b-instruct"


class QwenModelSingleton:
    """Thread-safe singleton for remote Qwen3-VL calls."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._loaded = False
        self._process_id = os.getpid()
        self.api_key = None
        self.api_base = None
        self.model_name = None
        self.timeout = None

    @classmethod
    def get_instance(cls):
        current_pid = os.getpid()
        if cls._instance is None or cls._instance._process_id != current_pid:
            with cls._lock:
                if cls._instance is None or cls._instance._process_id != current_pid:
                    cls._instance = cls()
        return cls._instance

    def load_model(self, weight_dir: str = None):
        """Validate and cache remote API configuration."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            del weight_dir

            self.api_key = (
                os.getenv("QWEN3VL_API_KEY")
                or os.getenv("UNIAPI_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            if not self.api_key:
                raise RuntimeError(
                    "Missing API key. Set QWEN3VL_API_KEY or UNIAPI_API_KEY."
                )

            self.api_base = (
                os.getenv("QWEN3VL_API_BASE")
                or os.getenv("UNIAPI_API_BASE")
                or DEFAULT_API_BASE
            ).rstrip("/")
            self.model_name = os.getenv("QWEN3VL_MODEL_NAME", DEFAULT_MODEL_NAME)
            self.timeout = float(os.getenv("QWEN3VL_TIMEOUT", "300"))
            self._loaded = True

            print(
                f"[model_loader] Using remote model {self.model_name} via {self.api_base}",
                flush=True,
            )

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Call the remote chat completion API and return response text."""
        self.load_model()
        request_messages = _to_api_messages(_normalise_messages(messages))
        payload = {
            "model": self.model_name,
            "messages": request_messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(max(temperature, 0.0)),
        }

        url = f"{self.api_base}/chat/completions"
        last_error = None
        for attempt in range(1, 4):
            try:
                t0 = time.time()
                response = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "Valuator-Qwen3vl-30B/1.0",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed = time.time() - t0
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code} from model API: {response.text[:500]}"
                    )
                data = response.json()
                text = _extract_response_text(data)
                print(
                    f"[generate] remote call ok in {elapsed:.1f}s, max_new={max_new_tokens}",
                    flush=True,
                )
                return text
            except (requests.RequestException, RuntimeError, json.JSONDecodeError) as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code is None and isinstance(exc, RuntimeError):
                    match = re.search(r"HTTP (\d+)", str(exc))
                    if match:
                        status_code = int(match.group(1))
                retryable = status_code in {408, 409, 429, 500, 502, 503, 504} or status_code is None
                last_error = exc
                if not retryable or attempt == 3:
                    raise last_error

            time.sleep(min(2 ** (attempt - 1), 4))

        raise RuntimeError(f"Remote model call failed: {last_error}")

    def clear_cache(self):
        """Remote API backend has no local cache to clear."""
        return


def _normalise_messages(messages: list) -> list:
    """Ensure every message has list-based content."""
    normalised = []
    for message in messages:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            msg["content"] = content
        else:
            msg["content"] = [{"type": "text", "text": str(content)}]
        normalised.append(msg)
    return normalised


def _to_api_messages(messages: list) -> list:
    api_messages = []
    for message in messages:
        role = message.get("role", "user")
        content = []
        has_image = False
        text_chunks = []

        for part in message.get("content", []):
            part_type = part.get("type", "text")
            if part_type == "text":
                text = part.get("text", "")
                text_chunks.append(text)
                content.append({"type": "text", "text": text})
            elif part_type == "image":
                has_image = True
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(part.get("image"))},
                    }
                )
            elif part_type == "image_url":
                has_image = True
                content.append(part)
            else:
                text = str(part.get("text", ""))
                text_chunks.append(text)
                content.append({"type": "text", "text": text})

        if has_image:
            api_messages.append({"role": role, "content": content})
        else:
            api_messages.append({"role": role, "content": "\n".join(text_chunks).strip()})

    return api_messages


def _image_to_data_url(image_obj) -> str:
    if image_obj is None:
        raise ValueError("Image content is missing")

    if isinstance(image_obj, str):
        if image_obj.startswith(("http://", "https://", "data:")):
            return image_obj
        if image_obj.startswith("file://"):
            image_obj = image_obj[7:]
        mime_type, _ = mimetypes.guess_type(image_obj)
        mime_type = mime_type or "image/jpeg"
        with open(image_obj, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    buf = io.BytesIO()
    image_obj.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_response_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Invalid model response: {json.dumps(data)[:500]}")

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(t for t in texts if t).strip()
    return str(content).strip()


def get_qwen_model(weight_dir: str = None) -> QwenModelSingleton:
    """Return the process-local singleton."""
    inst = QwenModelSingleton.get_instance()
    if not inst._loaded:
        inst.load_model(weight_dir)
    return inst
