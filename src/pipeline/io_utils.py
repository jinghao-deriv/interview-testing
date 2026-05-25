"""Safe JSON read/write helpers and artifact path management."""

import json
import re
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)
    print(f"[io] Written: {path}")


def extract_json_from_text(text: str) -> Any:
    """Extract the first JSON array or object from a text response.

    Handles:
    - Raw JSON
    - JSON wrapped in ```json ... ``` code fences
    - JSON preceded/followed by prose
    """
    # Strip leading/trailing whitespace
    text = text.strip()

    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code fence
    fence_match = re.search(
        r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```", text
    )
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Greedy search for outermost array then object
    for pattern in (r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"):
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"Could not extract valid JSON from LLM response. "
        f"First 300 chars: {text[:300]!r}"
    )
