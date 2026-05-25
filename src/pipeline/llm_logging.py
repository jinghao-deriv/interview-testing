"""Append-only LLM call logger.

Each call is recorded as a JSON entry in llm_calls.log.json with:
  stage, timestamp, model, provider, input_artifacts, output_artifact,
  prompt_tokens, completion_tokens.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

LOG_FILENAME = "llm_calls.log.json"


def _derive_provider(model: str) -> str:
    model_lower = model.lower()
    if model_lower.startswith("gpt") or model_lower.startswith("o1"):
        return "openai"
    if model_lower.startswith("claude"):
        return "anthropic"
    if model_lower.startswith("qwen"):
        return "alibaba"
    if model_lower.startswith("gemini"):
        return "google"
    if model_lower.startswith("llama") or model_lower.startswith("mistral"):
        return "meta/mistral"
    return "litellm_router"


def log_llm_call(
    stage: str,
    model: str,
    input_artifacts: List[str],
    output_artifact: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    output_dir: Path = Path("."),
) -> None:
    """Append a structured LLM call record to llm_calls.log.json."""
    entry = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": _derive_provider(model),
        "input_artifacts": input_artifacts,
        "output_artifact": output_artifact,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }

    log_path = output_dir / LOG_FILENAME
    entries: list = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append(entry)

    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)

    print(
        f"[llm_log] {stage} | model={model} | "
        f"tokens={prompt_tokens}+{completion_tokens} -> {output_artifact}"
    )
