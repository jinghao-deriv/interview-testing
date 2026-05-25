"""Stage 2: Constrained reply drafting.

A SINGLE LLM call receives, for every ticket:
  - preprocessed ticket text
  - Stage 1 classification output
  - deterministic decision output
  - the selected help-center article content (when available)

The model is constrained by explicit rules to produce safe, grounded replies.
If no LLM client is available a template-based fallback is used.
"""

import json
from typing import Dict, List, Optional

from .io_utils import extract_json_from_text
from .llm_client import LLMClient
from .models import (
    Answerability,
    ClassifiedTicket,
    Decision,
    DraftReply,
    EscalationTeam,
    IssueType,
    PreprocessedTicket,
    RawArticle,
    ReplyAction,
    RiskFlag,
)

_ESCALATION_TEAMS = ", ".join(f'"{e.value}"' for e in EscalationTeam)

_SYSTEM_PROMPT = f"""You are a support reply drafting assistant.

## Context
You will receive a JSON array of tickets. Each entry includes:
- Preprocessed ticket text (translated to English)
- Classification result (issue_type, answerability, risk_flag, reply_action)
- Deterministic decision result (auto_send_eligible, priority, risk_points)
- The selected help-center article content (may be null)

## Drafting rules (mandatory – never violate):
1. If auto_send_eligible=true, draft a concise, friendly, article-grounded reply that directly answers the customer.
2. If auto_send_eligible=false, draft a human-handoff message or information request as dictated by reply_action.
3. For reply_action=refuse_and_redirect (trading advice): refuse politely, NEVER suggest strategies, redirect to educational materials and risk disclosures.
4. For tickets with contains_legal_threat=true: do NOT admit fault, do NOT promise outcomes or timelines, do NOT speculate. Acknowledge receipt and state the matter will be reviewed.
5. For tickets with contains_security_access_issue=true or issue_type=login_access: NEVER ask for passwords, PINs, or full credentials.
6. For reply_action=escalate_to_human: write a polite holding reply that confirms escalation; provide no unauthorised commitments.
7. reply_body must be written in a professional, empathetic tone. Keep it under 150 words.
8. agent_note is an INTERNAL note for the human agent — not shown to the customer. It must describe action needed, risk context, and why auto-send was or was not granted.
9. escalation_team must be exactly one of: {_ESCALATION_TEAMS}
10. send_gate must be exactly "auto_send" when auto_send_eligible=true, otherwise "human_review".
11. reply_subject should be a concise, professional subject line.
12. Do NOT include chain-of-thought in any field.

## Escalation team selection guide:
- legal threat / risk_flag=legal               -> "Legal"
- duplicate charge / financial risk             -> "Payments"
- account_closure / data deletion              -> "Data Privacy"
- compliance / withdrawal KYC                  -> "Compliance"
- login / security / account access            -> "Account Security"
- all other cases                              -> "Customer Support"

## Output format
Return ONLY a valid JSON array. One object per ticket in input order:
[
  {{
    "ticket_id": "<string>",
    "reply_subject": "<string>",
    "reply_body": "<string – customer-facing reply>",
    "agent_note": "<string – internal only>",
    "escalation_team": "<one of the allowed teams>",
    "send_gate": "auto_send" | "human_review"
  }}
]"""


def _build_user_message(
    tickets: List[PreprocessedTicket],
    classified_map: Dict[str, ClassifiedTicket],
    decisions_map: Dict[str, Decision],
    articles_map: Dict[str, RawArticle],
) -> str:
    items = []
    for ticket in tickets:
        ct = classified_map[ticket.ticket_id]
        dec = decisions_map[ticket.ticket_id]
        article_content: Optional[str] = None
        if ct.best_article_id and ct.best_article_id in articles_map:
            art = articles_map[ct.best_article_id]
            article_content = f"[{art.title}] {art.content}"

        items.append(
            {
                "ticket_id": ticket.ticket_id,
                "subject": ticket.subject_for_processing,
                "message": ticket.message_for_processing,
                "original_language": ticket.original_language,
                "customer_tier": ticket.customer_tier,
                "classification": {
                    "issue_type": ct.issue_type.value,
                    "answerability": ct.answerability.value,
                    "risk_flag": ct.risk_flag.value,
                    "reply_action": ct.reply_action.value,
                    "best_article_id": ct.best_article_id,
                    "needs_more_info": ct.needs_more_info,
                    "contains_legal_threat": ct.contains_legal_threat,
                    "contains_security_access_issue": ct.contains_security_access_issue,
                },
                "decision": {
                    "auto_send_eligible": dec.auto_send_eligible,
                    "priority": dec.priority,
                    "risk_points": dec.risk_points,
                    "retrieval_score": dec.retrieval_score,
                    "decision_score": dec.decision_score,
                },
                "article_evidence": article_content,
            }
        )
    return json.dumps(items, ensure_ascii=False)


def _parse_llm_response(raw: str, expected_ids: List[str]) -> List[DraftReply]:
    parsed = extract_json_from_text(raw)
    if isinstance(parsed, dict):
        for key in ("replies", "drafts", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            lists = [v for v in parsed.values() if isinstance(v, list)]
            if lists:
                parsed = lists[0]
            else:
                raise ValueError(
                    f"Expected JSON array from LLM, got dict keys: {list(parsed.keys())}"
                )

    drafts = [DraftReply(**item) for item in parsed]
    returned_ids = {d.ticket_id for d in drafts}
    missing = set(expected_ids) - returned_ids
    if missing:
        raise ValueError(f"LLM response missing draft replies for: {missing}")
    return drafts


# ---------------------------------------------------------------------------
# Template-based fallback
# ---------------------------------------------------------------------------

_FALLBACK_BODIES = {
    ReplyAction.send_article_based_reply: (
        "Thank you for contacting us. We have reviewed your query and our "
        "help-center article should address your concern. Please check the "
        "information provided and let us know if you need further assistance."
    ),
    ReplyAction.escalate_to_human: (
        "Thank you for reaching out. Your request requires review by our "
        "specialist team. A member of our team will contact you as soon as "
        "possible."
    ),
    ReplyAction.request_missing_info: (
        "Thank you for contacting us. To assist you further, could you "
        "please provide additional details about your request?"
    ),
    ReplyAction.refuse_and_redirect: (
        "Thank you for your message. Our support team is unable to provide "
        "trading advice or strategy recommendations. We encourage you to "
        "review our educational materials and risk disclosure documents "
        "available in your account portal."
    ),
}

_TEAM_BY_ISSUE: Dict[IssueType, EscalationTeam] = {
    IssueType.trading_advice_request: EscalationTeam.customer_support,
    IssueType.duplicate_charge: EscalationTeam.payments,
    IssueType.deposit_missing: EscalationTeam.payments,
    IssueType.account_closure: EscalationTeam.data_privacy,
    IssueType.login_access: EscalationTeam.account_security,
    IssueType.withdrawal_delay: EscalationTeam.compliance,
    IssueType.verification_docs: EscalationTeam.compliance,
}


def _stub_draft_one(
    ticket: PreprocessedTicket,
    ct: ClassifiedTicket,
    dec: Decision,
) -> DraftReply:
    # Override escalation team for legal threats
    if ct.contains_legal_threat or ct.risk_flag == RiskFlag.legal:
        team = EscalationTeam.legal
    else:
        team = _TEAM_BY_ISSUE.get(ct.issue_type, EscalationTeam.customer_support)

    body = _FALLBACK_BODIES.get(ct.reply_action, _FALLBACK_BODIES[ReplyAction.escalate_to_human])

    send_gate = "auto_send" if dec.auto_send_eligible else "human_review"

    agent_note = (
        f"[STUB MODE] issue_type={ct.issue_type.value}, "
        f"risk_flag={ct.risk_flag.value}, "
        f"risk_points={dec.risk_points}, "
        f"auto_send_eligible={dec.auto_send_eligible}, "
        f"priority={dec.priority}. "
        "Template reply – human review recommended."
    )

    return DraftReply(
        ticket_id=ticket.ticket_id,
        reply_subject=f"Re: {ticket.subject_for_processing}",
        reply_body=body,
        agent_note=agent_note,
        escalation_team=team,
        send_gate=send_gate,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draft_replies(
    preprocessed_tickets: List[PreprocessedTicket],
    classified_tickets: List[ClassifiedTicket],
    decisions: List[Decision],
    articles_map: Dict[str, RawArticle],
    client: Optional[LLMClient] = None,
) -> tuple[List[DraftReply], int, int]:
    """Draft replies for all tickets in a single LLM call (or stub).

    Returns:
        (draft_replies, prompt_tokens, completion_tokens)
    """
    classified_map = {ct.ticket_id: ct for ct in classified_tickets}
    decisions_map = {d.ticket_id: d for d in decisions}

    if not client:
        print("[reply_drafting] No LLM client – using template-based stub replies.")
        stubs = [
            _stub_draft_one(t, classified_map[t.ticket_id], decisions_map[t.ticket_id])
            for t in preprocessed_tickets
        ]
        return stubs, 0, 0

    print(
        f"[reply_drafting] Sending Stage 2 LLM call for "
        f"{len(preprocessed_tickets)} ticket(s)…"
    )
    user_msg = _build_user_message(
        preprocessed_tickets, classified_map, decisions_map, articles_map
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw, prompt_tokens, completion_tokens = client.chat(messages, temperature=0.0)

    expected_ids = [t.ticket_id for t in preprocessed_tickets]
    drafts = _parse_llm_response(raw, expected_ids)

    order_map = {tid: i for i, tid in enumerate(expected_ids)}
    drafts.sort(key=lambda d: order_map.get(d.ticket_id, 9999))

    return drafts, prompt_tokens, completion_tokens
