"""
LangChain-compatible local LLM wrapper for Qwen3-VL-4B-Thinking.

Used by Valuator and Valuator_judge methods which call Bidder._run_llm_standalone().
The Bidder class expects self.llm to be callable with LangChain messages.
"""

from model_loader import get_qwen_model


class LocalQwenChatModel:
    """
    Mimics LangChain ChatOpenAI interface for local Qwen3-VL inference.

    Usage in Valuator:
        bidder.llm = LocalQwenChatModel(temperature=0.7)
        result = bidder.llm(messages, max_tokens=2048)  # returns AIMessage-like
    """

    def __init__(self, temperature: float = 0.7, max_tokens: int = 4096):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = get_qwen_model()

    def __call__(self, messages, **kwargs):
        """
        Call interface compatible with LangChain ChatOpenAI.

        Parameters
        ----------
        messages : list of HumanMessage / AIMessage / SystemMessage
        **kwargs : max_tokens, etc.

        Returns
        -------
        AIMessage-like object with .content attribute
        """
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Convert LangChain messages to OpenAI dict format
        openai_msgs = _langchain_to_openai(messages)

        # Check if any message has an associated image (for enhanced Valuator_judge)
        # Item image can be passed via kwargs
        item_image = kwargs.pop("item_image", None)
        if item_image:
            # Inject image into the last user message
            for msg in reversed(openai_msgs):
                if msg["role"] == "user":
                    if isinstance(msg["content"], str):
                        msg["content"] = [
                            {"type": "image", "image": f"file://{item_image}"},
                            {"type": "text", "text": msg["content"]},
                        ]
                    break

        raw = self._model.generate(openai_msgs, max_new_tokens=max_tokens, temperature=self.temperature)
        return _FakeAIMessage(raw)

    def get_num_tokens_from_messages(self, messages):
        """Estimate token count (~4 chars per token)."""
        total_chars = sum(len(str(getattr(m, "content", m))) for m in messages)
        return total_chars // 4


class _FakeAIMessage:
    """Minimal AIMessage-like object so Bidder._run_llm_standalone works."""

    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content


def _langchain_to_openai(messages) -> list:
    """Convert LangChain message objects to OpenAI dict format."""
    result = []
    for msg in messages:
        cls_name = type(msg).__name__
        if "System" in cls_name:
            role = "system"
        elif "AI" in cls_name or "Assistant" in cls_name:
            role = "assistant"
        else:
            role = "user"
        content = getattr(msg, "content", str(msg))
        result.append({"role": role, "content": content})
    return result
