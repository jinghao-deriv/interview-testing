"""Deterministic decisioning stage.

All scoring, eligibility, and priority logic is computed entirely in
Python from defined rules. No LLM call is made in this stage.

Rules (canonical reference):
  retrieval_score = score of best_article_id from retrieval results, else 0

  risk_points base (from risk_flag):
    none=0, financial=20, legal=35, compliance=20, security=25

  Additive flags:
    +20  contains_legal_threat=true
    +15  contains_security_access_issue=true
    +10  customer_tier=vip
    +10  answerability=needs_human
    +15  answerability=insufficient_context
    +10  needs_more_info=true

  decision_score = round((retrieval_score * 100) - risk_points)

  auto_send_eligible = true only if ALL of:
    answerability = auto_answer
    reply_action  in {send_article_based_reply, refuse_and_redirect}
    retrieval_score >= 0.65
    contains_legal_threat = false
    contains_security_access_issue = false
    risk_flag != legal

  priority = "urgent" if risk_points >= 35 else "normal"
"""

from typing import Dict, List

from .models import (
    Answerability,
    ClassifiedTicket,
    Decision,
    PreprocessedTicket,
    ReplyAction,
    RetrievalResult,
    RiskFlag,
)

# ---------------------------------------------------------------------------
# Risk point tables (defined once, validated by tests)
# ---------------------------------------------------------------------------

RISK_BASE: Dict[str, int] = {
    RiskFlag.none.value: 0,
    RiskFlag.financial.value: 20,
    RiskFlag.legal.value: 35,
    RiskFlag.compliance.value: 20,
    RiskFlag.security.value: 25,
}

RETRIEVAL_SCORE_THRESHOLD = 0.65

_AUTO_SEND_ACTIONS = {
    ReplyAction.send_article_based_reply,
    ReplyAction.refuse_and_redirect,
}


def compute_risk_points(
    classified: ClassifiedTicket,
    customer_tier: str,
) -> int:
    """Return the total risk_points for one ticket."""
    points = RISK_BASE.get(classified.risk_flag.value, 0)

    if classified.contains_legal_threat:
        points += 20
    if classified.contains_security_access_issue:
        points += 15
    if customer_tier.lower() == "vip":
        points += 10
    if classified.answerability == Answerability.needs_human:
        points += 10
    elif classified.answerability == Answerability.insufficient_context:
        points += 15
    if classified.needs_more_info:
        points += 10

    return points


def compute_auto_send_eligible(
    classified: ClassifiedTicket,
    retrieval_score: float,
) -> bool:
    """Return True only when all eligibility guards pass."""
    return (
        classified.answerability == Answerability.auto_answer
        and classified.reply_action in _AUTO_SEND_ACTIONS
        and retrieval_score >= RETRIEVAL_SCORE_THRESHOLD
        and not classified.contains_legal_threat
        and not classified.contains_security_access_issue
        and classified.risk_flag != RiskFlag.legal
    )


def compute_decisions(
    classified_tickets: List[ClassifiedTicket],
    retrieval_results: List[RetrievalResult],
    preprocessed_tickets: List[PreprocessedTicket],
) -> List[Decision]:
    """Compute deterministic Decision records for all tickets."""
    # Index for O(1) lookups
    retrieval_map: Dict[str, Dict[str, float]] = {
        r.ticket_id: {ca.article_id: ca.score for ca in r.candidate_articles}
        for r in retrieval_results
    }
    ticket_meta: Dict[str, PreprocessedTicket] = {
        t.ticket_id: t for t in preprocessed_tickets
    }

    decisions: List[Decision] = []
    for ct in classified_tickets:
        ticket = ticket_meta[ct.ticket_id]

        # retrieval_score
        best_id = ct.best_article_id
        retrieval_score: float = 0.0
        if best_id:
            retrieval_score = retrieval_map.get(ct.ticket_id, {}).get(best_id, 0.0)

        risk_points = compute_risk_points(ct, ticket.customer_tier)
        decision_score = round((retrieval_score * 100) - risk_points)
        auto_send = compute_auto_send_eligible(ct, retrieval_score)
        priority = "urgent" if risk_points >= 35 else "normal"

        decisions.append(
            Decision(
                ticket_id=ct.ticket_id,
                retrieval_score=round(retrieval_score, 6),
                risk_points=risk_points,
                decision_score=decision_score,
                auto_send_eligible=auto_send,
                priority=priority,
            )
        )

    return decisions
