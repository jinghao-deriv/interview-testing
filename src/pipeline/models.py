"""Pydantic models and controlled vocabulary enums for the support pipeline."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Controlled vocabularies (enums)
# ---------------------------------------------------------------------------

class IssueType(str, Enum):
    withdrawal_delay = "withdrawal_delay"
    deposit_missing = "deposit_missing"
    login_access = "login_access"
    account_closure = "account_closure"
    duplicate_charge = "duplicate_charge"
    verification_docs = "verification_docs"
    trading_advice_request = "trading_advice_request"
    other = "other"


class Answerability(str, Enum):
    auto_answer = "auto_answer"
    needs_human = "needs_human"
    insufficient_context = "insufficient_context"


class RiskFlag(str, Enum):
    none = "none"
    financial = "financial"
    legal = "legal"
    compliance = "compliance"
    security = "security"


class ReplyAction(str, Enum):
    send_article_based_reply = "send_article_based_reply"
    request_missing_info = "request_missing_info"
    escalate_to_human = "escalate_to_human"
    refuse_and_redirect = "refuse_and_redirect"


class EscalationTeam(str, Enum):
    customer_support = "Customer Support"
    payments = "Payments"
    compliance = "Compliance"
    account_security = "Account Security"
    legal = "Legal"
    data_privacy = "Data Privacy"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class RawTicket(BaseModel):
    ticket_id: str
    customer_tier: str
    channel: str
    language: str
    subject: str
    message: str
    created_at: str


class RawArticle(BaseModel):
    article_id: str
    title: str
    category: str
    language: str
    content: str


# ---------------------------------------------------------------------------
# Stage artifacts
# ---------------------------------------------------------------------------

class PreprocessedTicket(BaseModel):
    """Output of the preprocessing stage (preprocessed_tickets.json)."""
    ticket_id: str
    original_language: str
    original_subject: str
    original_message: str
    subject_for_processing: str
    message_for_processing: str
    translated: bool
    # Carried through for downstream decisioning
    customer_tier: str
    channel: str
    created_at: str


class CandidateArticle(BaseModel):
    article_id: str
    score: float


class RetrievalResult(BaseModel):
    """Output of the retrieval stage (retrieval_results.json)."""
    ticket_id: str
    candidate_articles: List[CandidateArticle]


class ClassifiedTicket(BaseModel):
    """Output of the Stage 1 LLM classification (classified_tickets.json)."""
    ticket_id: str
    issue_type: IssueType
    answerability: Answerability
    risk_flag: RiskFlag
    reply_action: ReplyAction
    best_article_id: Optional[str] = None
    needs_more_info: bool
    contains_legal_threat: bool
    contains_security_access_issue: bool
    rationale: str

    @field_validator("issue_type", mode="before")
    @classmethod
    def validate_issue_type(cls, v: str) -> str:
        allowed = {e.value for e in IssueType}
        if v not in allowed:
            raise ValueError(f"issue_type '{v}' not in allowed values: {allowed}")
        return v

    @field_validator("answerability", mode="before")
    @classmethod
    def validate_answerability(cls, v: str) -> str:
        allowed = {e.value for e in Answerability}
        if v not in allowed:
            raise ValueError(f"answerability '{v}' not in allowed values: {allowed}")
        return v

    @field_validator("risk_flag", mode="before")
    @classmethod
    def validate_risk_flag(cls, v: str) -> str:
        allowed = {e.value for e in RiskFlag}
        if v not in allowed:
            raise ValueError(f"risk_flag '{v}' not in allowed values: {allowed}")
        return v

    @field_validator("reply_action", mode="before")
    @classmethod
    def validate_reply_action(cls, v: str) -> str:
        allowed = {e.value for e in ReplyAction}
        if v not in allowed:
            raise ValueError(f"reply_action '{v}' not in allowed values: {allowed}")
        return v


class Decision(BaseModel):
    """Output of the deterministic decisioning stage (decisions.json)."""
    ticket_id: str
    retrieval_score: float
    risk_points: int
    decision_score: int
    auto_send_eligible: bool
    priority: str  # "urgent" | "normal"

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        if v not in {"urgent", "normal"}:
            raise ValueError(f"priority must be 'urgent' or 'normal', got '{v}'")
        return v


class DraftReply(BaseModel):
    """Output of the Stage 2 LLM reply drafting (draft_replies.json)."""
    ticket_id: str
    reply_subject: str
    reply_body: str
    agent_note: str
    escalation_team: EscalationTeam
    send_gate: str

    @field_validator("escalation_team", mode="before")
    @classmethod
    def validate_escalation_team(cls, v: str) -> str:
        allowed = {e.value for e in EscalationTeam}
        if v not in allowed:
            raise ValueError(f"escalation_team '{v}' not in allowed values: {allowed}")
        return v


class FinalSummary(BaseModel):
    """Output of the final aggregation stage (final_summary.json)."""
    total_tickets: int
    auto_send_count: int
    human_review_count: int
    urgent_ticket_ids: List[str]
    auto_send_ticket_ids: List[str]
    top_human_review_ticket_ids: List[str]
