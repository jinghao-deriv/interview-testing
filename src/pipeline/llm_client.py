"""OpenAI-compatible LLM client backed by the LiteLLM router.

Configuration via environment variables:
  LITELLM_BASE_URL  – e.g. http://localhost:4000/v1
  LITELLM_API_KEY   – your API key
  LITELLM_MODEL     – model name, e.g. qwen-max, gpt-4o

If LITELLM_BASE_URL is not set the client returns None and the
pipeline enters stub/fallback mode for all model-backed stages.
"""

import os
import time
from typing import List, Optional

try:
    from openai import OpenAI, APIError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openai>=1.30.0 is required. Run: pip install openai"
    ) from exc


class LLMClient:
    """Thin wrapper around the OpenAI-compatible client."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self.model = model

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> tuple[str, int, int]:
        """Send a chat completion request.

        Returns:
            (content, prompt_tokens, completion_tokens)
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                return content, prompt_tokens, completion_tokens
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(
                        f"[llm] Attempt {attempt} failed ({exc}). "
                        f"Retrying in {wait}s…"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc


def get_llm_client() -> Optional[LLMClient]:
    """Build and return an LLMClient from env vars, or None if not configured."""
    base_url = os.environ.get("LITELLM_BASE_URL", "").strip()
    if not base_url:
        print(
            "[llm] LITELLM_BASE_URL not set – running in stub/mock mode. "
            "Set LITELLM_BASE_URL, LITELLM_API_KEY, and LITELLM_MODEL to enable LLM calls."
        )
        return None

    api_key = os.environ.get("LITELLM_API_KEY")
    model = os.environ.get("LITELLM_MODEL")

    raw_client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"[llm] LLM client ready: model={model}, base_url={base_url}")
    return LLMClient(client=raw_client, model=model)
