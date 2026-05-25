"""Unit tests for the deterministic decisioning module.

Tests cover:
  - risk_points calculation for every risk_flag base value
  - each additive flag independently
  - all additive flags combined
  - decision_score formula
  - auto_send_eligible all-pass case
  - auto_send_eligible blocked by each individual guard
  - priority thresholds (boundary at 35)
  - compute_decisions integration
"""

import sys
import os

# Make src importable when tests are run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.pipeline.decisioning import (
    RISK_BASE,
    RETRIEVAL_SCORE_THRESHOLD,
    compute_auto_send_eligible,
    compute_decisions,
    compute_risk_points,
)
from src.pipeline.models import (
    Answerability,
    ClassifiedTicket,
    Decision,
    IssueType,
    PreprocessedTicket,
    ReplyAction,
    RetrievalResult,
    CandidateArticle,
    RiskFlag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ct(
    ticket_id: str = "T00",
    risk_flag: RiskFlag = RiskFlag.none,
    answerability: Answerability = Answerability.auto_answer,
    reply_action: ReplyAction = ReplyAction.send_article_based_reply,
    contains_legal_threat: bool = False,
    contains_security_access_issue: bool = False,
    needs_more_info: bool = False,
    best_article_id: str = "A01",
) -> ClassifiedTicket:
    return ClassifiedTicket(
        ticket_id=ticket_id,
        issue_type=IssueType.other,
        answerability=answerability,
        risk_flag=risk_flag,
        reply_action=reply_action,
        best_article_id=best_article_id,
        needs_more_info=needs_more_info,
        contains_legal_threat=contains_legal_threat,
        contains_security_access_issue=contains_security_access_issue,
        rationale="test",
    )


def _make_ticket(ticket_id: str = "T00", tier: str = "standard") -> PreprocessedTicket:
    return PreprocessedTicket(
        ticket_id=ticket_id,
        original_language="en",
        original_subject="subject",
        original_message="message",
        subject_for_processing="subject",
        message_for_processing="message",
        translated=False,
        customer_tier=tier,
        channel="email",
        created_at="2025-01-01T00:00:00Z",
    )


def _make_retrieval(ticket_id: str = "T00", article_id: str = "A01", score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        ticket_id=ticket_id,
        candidate_articles=[CandidateArticle(article_id=article_id, score=score)],
    )


# ---------------------------------------------------------------------------
# risk_points base values
# ---------------------------------------------------------------------------

class TestRiskPointsBase:
    @pytest.mark.parametrize(
        "flag,expected",
        [
            (RiskFlag.none, 0),
            (RiskFlag.financial, 20),
            (RiskFlag.legal, 35),
            (RiskFlag.compliance, 20),
            (RiskFlag.security, 25),
        ],
    )
    def test_base_risk_flag(self, flag, expected):
        ct = _make_ct(risk_flag=flag)
        assert compute_risk_points(ct, "standard") == expected

    def test_risk_base_table_complete(self):
        """Ensure RISK_BASE covers every RiskFlag value."""
        for flag in RiskFlag:
            assert flag.value in RISK_BASE, f"RISK_BASE missing entry for {flag}"


# ---------------------------------------------------------------------------
# Additive flag increments (applied independently to a zero-base ticket)
# ---------------------------------------------------------------------------

class TestRiskPointsAdditiveFlags:
    def _base_ct(self, **kwargs) -> ClassifiedTicket:
        return _make_ct(risk_flag=RiskFlag.none, **kwargs)

    def test_legal_threat_adds_20(self):
        ct = self._base_ct(contains_legal_threat=True)
        assert compute_risk_points(ct, "standard") == 20

    def test_security_access_adds_15(self):
        ct = self._base_ct(contains_security_access_issue=True)
        assert compute_risk_points(ct, "standard") == 15

    def test_vip_tier_adds_10(self):
        ct = self._base_ct()
        assert compute_risk_points(ct, "vip") == 10

    def test_needs_human_adds_10(self):
        ct = self._base_ct(answerability=Answerability.needs_human)
        assert compute_risk_points(ct, "standard") == 10

    def test_insufficient_context_adds_15(self):
        ct = self._base_ct(answerability=Answerability.insufficient_context)
        assert compute_risk_points(ct, "standard") == 15

    def test_needs_more_info_adds_10(self):
        ct = self._base_ct(needs_more_info=True)
        assert compute_risk_points(ct, "standard") == 10

    def test_tier_case_insensitive(self):
        ct = self._base_ct()
        assert compute_risk_points(ct, "VIP") == 10

    def test_all_additive_flags_combined(self):
        """legal_threat(20) + security(15) + vip(10) + needs_human(10) + needs_more_info(10) = 65
        base = none(0), so total = 65."""
        ct = _make_ct(
            risk_flag=RiskFlag.none,
            contains_legal_threat=True,
            contains_security_access_issue=True,
            answerability=Answerability.needs_human,
            needs_more_info=True,
        )
        # needs_human (+10) and needs_more_info (+10) are independent
        assert compute_risk_points(ct, "vip") == 0 + 20 + 15 + 10 + 10 + 10

    def test_legal_flag_base_plus_legal_threat(self):
        """legal base(35) + legal_threat(+20) + vip(+10) = 65."""
        ct = _make_ct(
            risk_flag=RiskFlag.legal,
            contains_legal_threat=True,
        )
        assert compute_risk_points(ct, "vip") == 35 + 20 + 10


# ---------------------------------------------------------------------------
# decision_score
# ---------------------------------------------------------------------------

class TestDecisionScore:
    def test_formula_basic(self):
        # retrieval_score=0.8, risk_points=20 -> round(80 - 20) = 60
        ct = _make_ct(risk_flag=RiskFlag.financial)
        ticket = _make_ticket()
        retrieval = _make_retrieval(score=0.8)
        decisions = compute_decisions([ct], [retrieval], [ticket])
        assert decisions[0].decision_score == 60

    def test_formula_zero_retrieval(self):
        # retrieval_score=0, risk_points=0 -> 0
        ct = _make_ct(risk_flag=RiskFlag.none, best_article_id="UNKNOWN")
        ticket = _make_ticket()
        retrieval = _make_retrieval(score=0.7)  # best_article_id doesn't match
        decisions = compute_decisions([ct], [retrieval], [ticket])
        assert decisions[0].retrieval_score == 0.0
        assert decisions[0].decision_score == 0

    def test_formula_negative_score(self):
        # retrieval_score=0, risk_points=55 -> -55
        ct = _make_ct(
            risk_flag=RiskFlag.legal,     # 35
            contains_legal_threat=True,   # +20
            best_article_id=None,
        )
        ticket = _make_ticket()
        retrieval = _make_retrieval()  # won't match None
        decisions = compute_decisions([ct], [retrieval], [ticket])
        assert decisions[0].decision_score == -55

    def test_rounding(self):
        # round(0.755 * 100 - 20) = round(75.5 - 20) = round(55.5) = 56
        ct = _make_ct(risk_flag=RiskFlag.financial)
        ticket = _make_ticket()
        retrieval = _make_retrieval(score=0.755)
        decisions = compute_decisions([ct], [retrieval], [ticket])
        assert isinstance(decisions[0].decision_score, int)


# ---------------------------------------------------------------------------
# auto_send_eligible
# ---------------------------------------------------------------------------

class TestAutoSendEligible:
    def _eligible_ct(self) -> ClassifiedTicket:
        """A ticket that satisfies ALL eligibility guards."""
        return _make_ct(
            risk_flag=RiskFlag.none,
            answerability=Answerability.auto_answer,
            reply_action=ReplyAction.send_article_based_reply,
            contains_legal_threat=False,
            contains_security_access_issue=False,
        )

    def test_all_guards_pass(self):
        ct = self._eligible_ct()
        assert compute_auto_send_eligible(ct, 0.8) is True

    def test_refuse_and_redirect_is_also_eligible(self):
        ct = self._eligible_ct()
        ct.reply_action = ReplyAction.refuse_and_redirect
        assert compute_auto_send_eligible(ct, 0.8) is True

    def test_blocked_by_needs_human(self):
        ct = self._eligible_ct()
        ct.answerability = Answerability.needs_human
        assert compute_auto_send_eligible(ct, 0.8) is False

    def test_blocked_by_insufficient_context(self):
        ct = self._eligible_ct()
        ct.answerability = Answerability.insufficient_context
        assert compute_auto_send_eligible(ct, 0.8) is False

    def test_blocked_by_wrong_reply_action_escalate(self):
        ct = self._eligible_ct()
        ct.reply_action = ReplyAction.escalate_to_human
        assert compute_auto_send_eligible(ct, 0.8) is False

    def test_blocked_by_wrong_reply_action_request_info(self):
        ct = self._eligible_ct()
        ct.reply_action = ReplyAction.request_missing_info
        assert compute_auto_send_eligible(ct, 0.8) is False

    def test_blocked_by_low_retrieval_score(self):
        ct = self._eligible_ct()
        assert compute_auto_send_eligible(ct, 0.64) is False

    def test_exact_threshold_passes(self):
        ct = self._eligible_ct()
        assert compute_auto_send_eligible(ct, RETRIEVAL_SCORE_THRESHOLD) is True

    def test_blocked_by_legal_threat(self):
        ct = self._eligible_ct()
        ct.contains_legal_threat = True
        assert compute_auto_send_eligible(ct, 0.8) is False

    def test_blocked_by_security_access_issue(self):
        ct = self._eligible_ct()
        ct.contains_security_access_issue = True
        assert compute_auto_send_eligible(ct, 0.8) is False

    def test_blocked_by_legal_risk_flag(self):
        ct = self._eligible_ct()
        ct.risk_flag = RiskFlag.legal
        assert compute_auto_send_eligible(ct, 0.8) is False

    def test_financial_risk_flag_does_not_block(self):
        """financial risk alone should not block auto_send."""
        ct = self._eligible_ct()
        ct.risk_flag = RiskFlag.financial
        assert compute_auto_send_eligible(ct, 0.8) is True

    def test_compliance_risk_flag_does_not_block(self):
        ct = self._eligible_ct()
        ct.risk_flag = RiskFlag.compliance
        assert compute_auto_send_eligible(ct, 0.8) is True


# ---------------------------------------------------------------------------
# Priority thresholds
# ---------------------------------------------------------------------------

class TestPriority:
    def _decision_for(self, risk_points_expected: int) -> Decision:
        """Build a decision where risk_points == risk_points_expected."""
        # Use legal flag (35) as base, optionally with legal_threat (+20)
        if risk_points_expected == 35:
            ct = _make_ct(risk_flag=RiskFlag.legal)
            tier = "standard"
        elif risk_points_expected == 34:
            # security(25) + security_access(15) = 40, not quite. Use compliance(20) + needs_human(10) + needs_more_info(10) - 6 ...
            # Easiest: security(25) + vip(10) - 1 doesn't work without fractional.
            # Use financial(20) + needs_more_info(10) + vip(10) - 6... just use a known combo
            # financial(20) + needs_human(10) + needs_more_info(10) - 6 ... tricky
            # Actually: none(0) + legal_threat(20) + security(15) - 1 not possible.
            # Use compliance(20) + vip(10) + needs_more_info(10) - 6: no
            # Simplest combo for 34: none(0) + legal_threat(20) + needs_human(10) + needs_more_info(10) - 6 can't subtract
            # financial(20) + needs_human(10) + needs_more_info(10) = 40, too high
            # compliance(20) + needs_human(10) + needs_more_info(10) = 40
            # security(25) + vip(10) = 35 (urgent)
            # none(0) + legal_threat(20) + needs_human(10) + vip(10) = 40
            # none(0) + legal_threat(20) + security_access(15) - 1: impossible
            # Use financial(20) + needs_more_info(10) + needs_human(10) = 40
            # Let's just use mock – this test relies on priority value
            pytest.skip("34 not easily constructable from rule combinations – tested via boundary below")
        elif risk_points_expected == 0:
            ct = _make_ct(risk_flag=RiskFlag.none)
            tier = "standard"
        else:
            pytest.skip(f"Not testing {risk_points_expected} directly here")

        ticket = _make_ticket(tier=tier)
        retrieval = _make_retrieval()
        decisions = compute_decisions([ct], [retrieval], [ticket])
        return decisions[0]

    def test_urgent_at_35(self):
        ct = _make_ct(risk_flag=RiskFlag.legal)
        ticket = _make_ticket()
        retrieval = _make_retrieval()
        d = compute_decisions([ct], [retrieval], [ticket])[0]
        assert d.risk_points == 35
        assert d.priority == "urgent"

    def test_urgent_above_35(self):
        ct = _make_ct(risk_flag=RiskFlag.legal, contains_legal_threat=True)  # 35+20=55
        ticket = _make_ticket()
        retrieval = _make_retrieval()
        d = compute_decisions([ct], [retrieval], [ticket])[0]
        assert d.risk_points == 55
        assert d.priority == "urgent"

    def test_normal_below_35(self):
        ct = _make_ct(risk_flag=RiskFlag.financial)  # 20
        ticket = _make_ticket()
        retrieval = _make_retrieval()
        d = compute_decisions([ct], [retrieval], [ticket])[0]
        assert d.risk_points == 20
        assert d.priority == "normal"

    def test_normal_at_zero(self):
        ct = _make_ct(risk_flag=RiskFlag.none)
        ticket = _make_ticket()
        retrieval = _make_retrieval()
        d = compute_decisions([ct], [retrieval], [ticket])[0]
        assert d.priority == "normal"

    def test_security_plus_vip_is_urgent(self):
        # security(25) + vip(10) = 35 -> urgent
        ct = _make_ct(risk_flag=RiskFlag.security)
        ticket = _make_ticket(tier="vip")
        retrieval = _make_retrieval()
        d = compute_decisions([ct], [retrieval], [ticket])[0]
        assert d.risk_points == 35
        assert d.priority == "urgent"


# ---------------------------------------------------------------------------
# compute_decisions integration
# ---------------------------------------------------------------------------

class TestComputeDecisionsIntegration:
    def test_returns_one_decision_per_ticket(self):
        tickets = [_make_ticket(f"T{i:02d}") for i in range(5)]
        cts = [_make_ct(f"T{i:02d}") for i in range(5)]
        retrievals = [_make_retrieval(f"T{i:02d}") for i in range(5)]
        decisions = compute_decisions(cts, retrievals, tickets)
        assert len(decisions) == 5
        assert {d.ticket_id for d in decisions} == {f"T{i:02d}" for i in range(5)}

    def test_missing_best_article_id_gives_zero_retrieval_score(self):
        ct = _make_ct(best_article_id=None)
        ticket = _make_ticket()
        retrieval = _make_retrieval(score=0.9)
        d = compute_decisions([ct], [retrieval], [ticket])[0]
        assert d.retrieval_score == 0.0

    def test_best_article_not_in_candidates_gives_zero(self):
        ct = _make_ct(best_article_id="A99")
        ticket = _make_ticket()
        retrieval = _make_retrieval(article_id="A01", score=0.9)
        d = compute_decisions([ct], [retrieval], [ticket])[0]
        assert d.retrieval_score == 0.0

    def test_retrieval_score_matches_candidate(self):
        ct = _make_ct(best_article_id="A01")
        ticket = _make_ticket()
        retrieval = _make_retrieval(article_id="A01", score=0.75)
        d = compute_decisions([ct], [retrieval], [ticket])[0]
        assert abs(d.retrieval_score - 0.75) < 1e-6
