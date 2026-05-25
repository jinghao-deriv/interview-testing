"""Multilingual preprocessing stage.

All non-English tickets are translated to English via a single batched
LLM call before retrieval and classification. English tickets pass through
unchanged. If no LLM client is available, the pipeline proceeds with
the original text and marks the ticket as untranslated.
"""

import json
from typing import List, Optional

from .io_utils import extract_json_from_text
from .llm_client import LLMClient
from .models import PreprocessedTicket, RawTicket

_TRANSLATION_PROMPT_SYSTEM = """You are a professional translation assistant.
You will receive a JSON array of tickets written in non-English languages.
Translate each ticket's subject and message fields into English.
Preserve all factual details exactly; do not add, remove, or interpret content.

Return ONLY a valid JSON array – no explanatory text, no markdown.
Each element must follow this exact schema:
{
  "ticket_id": "<original ticket_id>",
  "translated_subject": "<English translation of subject>",
  "translated_message": "<English translation of message>"
}"""


def _build_translation_payload(tickets: List[RawTicket]) -> str:
    items = [
        {
            "ticket_id": t.ticket_id,
            "language": t.language,
            "subject": t.subject,
            "message": t.message,
        }
        for t in tickets
    ]
    return json.dumps(items, ensure_ascii=False)


def _translate_batch(
    tickets: List[RawTicket],
    client: LLMClient,
) -> dict[str, tuple[str, str]]:
    """Return {ticket_id: (translated_subject, translated_message)}."""
    payload = _build_translation_payload(tickets)
    messages = [
        {"role": "system", "content": _TRANSLATION_PROMPT_SYSTEM},
        {"role": "user", "content": payload},
    ]
    raw, prompt_tokens, completion_tokens = client.chat(messages, temperature=0.0)
    parsed = extract_json_from_text(raw)
    if not isinstance(parsed, list):
        parsed = parsed.get("translations", list(parsed.values())[0])

    result: dict[str, tuple[str, str]] = {}
    for item in parsed:
        tid = item["ticket_id"]
        result[tid] = (item["translated_subject"], item["translated_message"])
    return result, prompt_tokens, completion_tokens


def preprocess_tickets(
    tickets: List[RawTicket],
    client: Optional[LLMClient] = None,
) -> tuple[List[PreprocessedTicket], int, int]:
    """Translate non-English tickets and build preprocessed ticket records.

    Returns:
        (preprocessed_tickets, total_prompt_tokens, total_completion_tokens)
    """
    non_english = [t for t in tickets if t.language != "en"]
    translations: dict[str, tuple[str, str]] = {}
    prompt_tokens = 0
    completion_tokens = 0

    if non_english:
        if client:
            print(
                f"[preprocess] Translating {len(non_english)} non-English ticket(s) "
                f"via LLM…"
            )
            try:
                translations, prompt_tokens, completion_tokens = _translate_batch(
                    non_english, client
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[preprocess] Translation failed ({exc}). "
                    "Non-English tickets will be flagged as untranslated."
                )
        else:
            print(
                "[preprocess] No LLM client – non-English tickets will not be "
                "translated (stub mode). Classification accuracy may be reduced."
            )

    preprocessed: List[PreprocessedTicket] = []
    for ticket in tickets:
        if ticket.language == "en":
            preprocessed.append(
                PreprocessedTicket(
                    ticket_id=ticket.ticket_id,
                    original_language=ticket.language,
                    original_subject=ticket.subject,
                    original_message=ticket.message,
                    subject_for_processing=ticket.subject,
                    message_for_processing=ticket.message,
                    translated=False,
                    customer_tier=ticket.customer_tier,
                    channel=ticket.channel,
                    created_at=ticket.created_at,
                )
            )
        else:
            if ticket.ticket_id in translations:
                trans_subject, trans_message = translations[ticket.ticket_id]
                preprocessed.append(
                    PreprocessedTicket(
                        ticket_id=ticket.ticket_id,
                        original_language=ticket.language,
                        original_subject=ticket.subject,
                        original_message=ticket.message,
                        subject_for_processing=trans_subject,
                        message_for_processing=trans_message,
                        translated=True,
                        customer_tier=ticket.customer_tier,
                        channel=ticket.channel,
                        created_at=ticket.created_at,
                    )
                )
            else:
                # Stub fallback: carry original text with a clear prefix
                preprocessed.append(
                    PreprocessedTicket(
                        ticket_id=ticket.ticket_id,
                        original_language=ticket.language,
                        original_subject=ticket.subject,
                        original_message=ticket.message,
                        subject_for_processing=(
                            f"[{ticket.language.upper()} – UNTRANSLATED] {ticket.subject}"
                        ),
                        message_for_processing=(
                            f"[{ticket.language.upper()} – UNTRANSLATED] {ticket.message}"
                        ),
                        translated=False,
                        customer_tier=ticket.customer_tier,
                        channel=ticket.channel,
                        created_at=ticket.created_at,
                    )
                )

    return preprocessed, prompt_tokens, completion_tokens
