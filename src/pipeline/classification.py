"""Stage 1: Ticket classification and evidence assessment.

A SINGLE LLM call receives all preprocessed tickets together with their
top retrieved candidate articles. The model is constrained to return
only values from the defined controlled vocabularies.

If no LLM client is available the module falls back to keyword-based
heuristics that produce structurally valid (but lower-accuracy) output.
"""

import json
from typing import Dict, List, Optional

from .io_utils import extract_json_from_text
from .llm_client import LLMClient
from .models import (
    Answerability,
    ClassifiedTicket,
    IssueType,
    PreprocessedTicket,
    RawArticle,
    ReplyAction,
    RetrievalResult,
    RiskFlag,
)

# ---------------------------------------------------------------------------
# Controlled-vocabulary reference strings injected into the prompt
# ---------------------------------------------------------------------------

_ISSUE_TYPES = ", ".join(e.value for e in IssueType)
_ANSWERABILITY = ", ".join(e.value for e in Answerability)
_RISK_FLAGS = ", ".join(e.value for e in RiskFlag)
_REPLY_ACTIONS = ", ".join(e.value for e in ReplyAction)

_SYSTEM_PROMPT = f"""You are a support ticket classification engine.

## Your task
Analyze each ticket using the provided candidate help-center articles as evidence.
For every ticket return one classification object.

## Controlled Vocabularies – use ONLY these exact string values:
issue_type:      {_ISSUE_TYPES}
answerability:   {_ANSWERABILITY}
risk_flag:       {_RISK_FLAGS}
reply_action:    {_REPLY_ACTIONS}

## Decision rules (apply these strictly):
- answerability=auto_answer only when a candidate article fully addresses the issue and no action is needed beyond information delivery.
- answerability=needs_human when a human must take an account action (refunds, manual unlocks, closures).
- answerability=insufficient_context when the ticket content is unclear or missing critical details.
- reply_action=refuse_and_redirect ONLY for trading_advice_request tickets.
- reply_action=escalate_to_human whenever answerability is needs_human or the risk_flag is legal/financial/compliance/security.
- reply_action=request_missing_info when answerability=insufficient_context.
- risk_flag=legal when the ticket contains explicit threats of legal action, regulator reports, or litigation language.
- risk_flag=financial when unresolved financial loss is claimed (duplicate charges, missing large amounts).
- risk_flag=compliance when regulatory or KYC/AML processes are involved.
- risk_flag=security when account access, authentication, or credentials are at risk.
- contains_legal_threat=true when the customer explicitly threatens legal action or regulatory reporting.
- contains_security_access_issue=true for login failures, account lockouts, password resets, or stolen-access concerns.
- best_article_id must be one of the candidate article IDs listed for that ticket, or null if none is relevant.
- needs_more_info=true when essential details are missing to resolve the issue.

## Output format
Return ONLY a valid JSON array – no prose, no markdown fences, no chain-of-thought.
One object per ticket, in input order:
[
  {{
    "ticket_id": "<string>",
    "issue_type": "<one of the allowed values>",
    "answerability": "<one of the allowed values>",
    "risk_flag": "<one of the allowed values>",
    "reply_action": "<one of the allowed values>",
    "best_article_id": "<article_id or null>",
    "needs_more_info": <true|false>,
    "contains_legal_threat": <true|false>,
    "contains_security_access_issue": <true|false>,
    "rationale": "<one concise sentence – NO chain-of-thought>"
  }}
]"""


def _build_user_message(
    tickets: List[PreprocessedTicket],
    retrieval_results: List[RetrievalResult],
    articles_map: Dict[str, RawArticle],
) -> str:
    retrieval_map = {r.ticket_id: r.candidate_articles for r in retrieval_results}
    items = []
    for ticket in tickets:
        candidates_raw = retrieval_map.get(ticket.ticket_id, [])
        candidates = []
        for ca in candidates_raw:
            art = articles_map.get(ca.article_id)
            if art:
                candidates.append(
                    {
                        "article_id": ca.article_id,
                        "score": ca.score,
                        "title": art.title,
                        "category": art.category,
                        "content": art.content,
                    }
                )
        items.append(
            {
                "ticket_id": ticket.ticket_id,
                "subject": ticket.subject_for_processing,
                "message": ticket.message_for_processing,
                "customer_tier": ticket.customer_tier,
                "channel": ticket.channel,
                "candidate_articles": candidates,
            }
        )
    return json.dumps(items, ensure_ascii=False)


def _parse_llm_response(raw: str, expected_ids: List[str]) -> List[ClassifiedTicket]:
    parsed = extract_json_from_text(raw)
    if isinstance(parsed, dict):
        # Some models wrap the array in a key
        for key in ("tickets", "results", "classifications"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            # Try to grab the first list value
            lists = [v for v in parsed.values() if isinstance(v, list)]
            if lists:
                parsed = lists[0]
            else:
                raise ValueError(f"Expected JSON array from LLM, got dict keys: {list(parsed.keys())}")

    classified = [ClassifiedTicket(**item) for item in parsed]

    # Validate all expected ticket_ids are present
    returned_ids = {c.ticket_id for c in classified}
    missing = set(expected_ids) - returned_ids
    if missing:
        raise ValueError(f"LLM response is missing classifications for ticket_id(s): {missing}")

    return classified


# ---------------------------------------------------------------------------
# Stub / fallback heuristics
# ---------------------------------------------------------------------------

_ISSUE_KEYWORDS: Dict[IssueType, List[str]] = {
    IssueType.withdrawal_delay: [
        "withdraw", "withdrawal", "pending", "release", "funds not released",
    ],
    IssueType.deposit_missing: [
        "deposit", "transfer", "funds", "bank transfer", "missing",
        "not appeared", "not received",
    ],
    IssueType.login_access: [
        "login", "log in", "password", "reset", "access", "verification code",
        "2fa", "otp", "cannot enter", "regain access", "reset link",
    ],
    IssueType.account_closure: [
        "close", "closure", "delete", "deletion", "shut", "remove account",
    ],
    IssueType.duplicate_charge: [
        "charged twice", "double charge", "duplicate", "refund", "overcharged",
        "same charge", "card charged",
    ],
    IssueType.verification_docs: [
        "document", "verification", "proof", "address", "identity", "rejected",
        "refused", "id", "kyc", "accepted documents", "proof-of-address",
    ],
    IssueType.trading_advice_request: [
        "strategy", "trading", "trade", "invest", "return", "safest",
        "profit", "high-return", "advise",
    ],
}


def _keyword_score(text: str, keywords: List[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def _stub_classify_one(ticket: PreprocessedTicket) -> ClassifiedTicket:
    text = (
        f"{ticket.subject_for_processing} {ticket.message_for_processing}"
    ).lower()

    # Determine issue_type
    best_type = IssueType.other
    best_score = 0
    for issue, kws in _ISSUE_KEYWORDS.items():
        sc = _keyword_score(text, kws)
        if sc > best_score:
            best_score = sc
            best_type = issue

    # Legal threat detection
    legal_kws = ["legal", "lawsuit", "sue", "regulator", "report", "legal action", "solicitor"]
    contains_legal_threat = any(kw in text for kw in legal_kws)

    # Security/access
    security_kws = ["login", "log in", "password", "reset", "access", "cannot enter", "regain"]
    contains_security_access_issue = best_type == IssueType.login_access or any(
        kw in text for kw in security_kws
    )

    # Risk flag
    if contains_legal_threat:
        risk_flag = RiskFlag.legal
    elif best_type == IssueType.duplicate_charge:
        risk_flag = RiskFlag.financial
    elif best_type in (IssueType.withdrawal_delay, IssueType.verification_docs):
        risk_flag = RiskFlag.compliance
    elif best_type == IssueType.login_access:
        risk_flag = RiskFlag.security
    else:
        risk_flag = RiskFlag.none

    # Answerability
    if contains_legal_threat or best_type in (
        IssueType.duplicate_charge,
        IssueType.account_closure,
    ):
        answerability = Answerability.needs_human
    elif best_type == IssueType.trading_advice_request:
        answerability = Answerability.auto_answer
    else:
        answerability = Answerability.auto_answer

    # Reply action
    if best_type == IssueType.trading_advice_request:
        reply_action = ReplyAction.refuse_and_redirect
    elif answerability == Answerability.needs_human or risk_flag in (
        RiskFlag.legal, RiskFlag.financial
    ):
        reply_action = ReplyAction.escalate_to_human
    elif answerability == Answerability.insufficient_context:
        reply_action = ReplyAction.request_missing_info
    else:
        reply_action = ReplyAction.send_article_based_reply

    needs_more_info = answerability == Answerability.insufficient_context

    return ClassifiedTicket(
        ticket_id=ticket.ticket_id,
        issue_type=best_type,
        answerability=answerability,
        risk_flag=risk_flag,
        reply_action=reply_action,
        best_article_id=None,
        needs_more_info=needs_more_info,
        contains_legal_threat=contains_legal_threat,
        contains_security_access_issue=contains_security_access_issue,
        rationale="[STUB MODE] Classified via keyword heuristics; no LLM available.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_tickets(
    preprocessed_tickets: List[PreprocessedTicket],
    retrieval_results: List[RetrievalResult],
    articles_map: Dict[str, RawArticle],
    client: Optional[LLMClient] = None,
) -> tuple[List[ClassifiedTicket], int, int]:
    """Classify all tickets in a single call (or stub if no client).

    Returns:
        (classified_tickets, prompt_tokens, completion_tokens)
    """
    if not client:
        print("[classification] No LLM client – using keyword-based stub classification.")
        stubs = [_stub_classify_one(t) for t in preprocessed_tickets]
        # Best-effort: assign best_article_id from top retrieval candidate
        retrieval_map = {r.ticket_id: r.candidate_articles for r in retrieval_results}
        for ct in stubs:
            candidates = retrieval_map.get(ct.ticket_id, [])
            if candidates and candidates[0].score > 0:
                ct.best_article_id = candidates[0].article_id
        return stubs, 0, 0

    print(
        f"[classification] Sending Stage 1 LLM call for "
        f"{len(preprocessed_tickets)} ticket(s)…"
    )
    user_msg = _build_user_message(preprocessed_tickets, retrieval_results, articles_map)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw, prompt_tokens, completion_tokens = client.chat(messages, temperature=0.0)

    expected_ids = [t.ticket_id for t in preprocessed_tickets]
    classified = _parse_llm_response(raw, expected_ids)

    # Ensure ticket order matches input order
    order_map = {tid: i for i, tid in enumerate(expected_ids)}
    classified.sort(key=lambda c: order_map.get(c.ticket_id, 9999))

    return classified, prompt_tokens, completion_tokens
