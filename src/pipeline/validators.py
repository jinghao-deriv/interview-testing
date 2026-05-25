"""Cross-artifact validation.

Checks performed:
  1. Schema conformance – every artifact can be re-parsed into its Pydantic model.
  2. Controlled vocabulary – all enum fields use only allowed values.
  3. Cross-artifact ID consistency – every ticket_id flows through all stages.
  4. Retrieval integrity – best_article_id in classified_tickets resolves to
     a candidate from retrieval_results (or score is correctly 0).
  5. Decisioning parity – recompute decision fields from scratch and compare
     to persisted decisions.json.
  6. Auto-send guard – no auto_send_eligible=true ticket violates a guard rule.
"""

from pathlib import Path
from typing import Dict, List

from .decisioning import compute_auto_send_eligible, compute_risk_points
from .io_utils import read_json
from .models import (
    ClassifiedTicket,
    Decision,
    DraftReply,
    FinalSummary,
    PreprocessedTicket,
    RetrievalResult,
)


class ValidationError(Exception):
    """Raised when one or more validation checks fail."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s):\n" + "\n".join(errors))


def _load_artifact(path: Path, model, plural=True):
    """Load a JSON artifact and parse into a list of Pydantic models."""
    data = read_json(path)
    if plural:
        return [model(**item) for item in data]
    return model(**data)


def validate_artifacts(output_dir: Path, require_final_summary: bool = True) -> None:
    """Validate all pipeline artifacts in output_dir.

    Raises ValidationError with a full list of issues found.
    """
    errors: List[str] = []

    # -----------------------------------------------------------------
    # Load artifacts
    # -----------------------------------------------------------------
    def safe_load(name, model, plural=True):
        path = output_dir / name
        try:
            return _load_artifact(path, model, plural)
        except FileNotFoundError:
            errors.append(f"MISSING artifact: {name}")
            return None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"PARSE ERROR in {name}: {exc}")
            return None

    preprocessed: List[PreprocessedTicket] = safe_load(
        "preprocessed_tickets.json", PreprocessedTicket
    )
    retrieval: List[RetrievalResult] = safe_load(
        "retrieval_results.json", RetrievalResult
    )
    classified: List[ClassifiedTicket] = safe_load(
        "classified_tickets.json", ClassifiedTicket
    )
    decisions: List[Decision] = safe_load("decisions.json", Decision)
    drafts: List[DraftReply] = safe_load("draft_replies.json", DraftReply)
    # final_summary.json is written AFTER validation completes during a run.
    # The run command disables this check to avoid comparing against a stale
    # summary from a previous execution.
    summary: FinalSummary = (
        safe_load("final_summary.json", FinalSummary, plural=False)
        if require_final_summary
        else None
    )

    if errors:
        raise ValidationError(errors)

    # -----------------------------------------------------------------
    # ID consistency checks
    # -----------------------------------------------------------------
    preprocessed_ids = {t.ticket_id for t in preprocessed}
    retrieval_ids = {r.ticket_id for r in retrieval}
    classified_ids = {c.ticket_id for c in classified}
    decision_ids = {d.ticket_id for d in decisions}
    draft_ids = {d.ticket_id for d in drafts}

    for stage_name, stage_ids in [
        ("retrieval_results", retrieval_ids),
        ("classified_tickets", classified_ids),
        ("decisions", decision_ids),
        ("draft_replies", draft_ids),
    ]:
        missing = preprocessed_ids - stage_ids
        extra = stage_ids - preprocessed_ids
        if missing:
            errors.append(
                f"ID consistency: {stage_name} is missing ticket_id(s): {sorted(missing)}"
            )
        if extra:
            errors.append(
                f"ID consistency: {stage_name} has unexpected ticket_id(s): {sorted(extra)}"
            )

    # Duplicate IDs
    for stage_name, items in [
        ("preprocessed_tickets", [t.ticket_id for t in preprocessed]),
        ("classified_tickets", [c.ticket_id for c in classified]),
        ("decisions", [d.ticket_id for d in decisions]),
        ("draft_replies", [d.ticket_id for d in drafts]),
    ]:
        seen: Dict[str, int] = {}
        for tid in items:
            seen[tid] = seen.get(tid, 0) + 1
        dups = [tid for tid, cnt in seen.items() if cnt > 1]
        if dups:
            errors.append(f"DUPLICATE ticket_id(s) in {stage_name}: {dups}")

    # -----------------------------------------------------------------
    # Retrieval integrity
    # -----------------------------------------------------------------
    retrieval_map = {
        r.ticket_id: {ca.article_id: ca.score for ca in r.candidate_articles}
        for r in retrieval
    }
    for ct in classified:
        if ct.best_article_id is not None:
            if ct.best_article_id not in retrieval_map.get(ct.ticket_id, {}):
                errors.append(
                    f"[{ct.ticket_id}] best_article_id '{ct.best_article_id}' not in "
                    f"retrieval candidates."
                )

    # -----------------------------------------------------------------
    # Decisioning parity – recompute from classified and check key fields
    # -----------------------------------------------------------------
    preprocessed_map = {t.ticket_id: t for t in preprocessed}
    decision_map = {d.ticket_id: d for d in decisions}

    for ct in classified:
        ticket = preprocessed_map.get(ct.ticket_id)
        dec = decision_map.get(ct.ticket_id)
        if not ticket or not dec:
            continue

        best_id = ct.best_article_id
        rs: float = 0.0
        if best_id:
            rs = retrieval_map.get(ct.ticket_id, {}).get(best_id, 0.0)

        expected_rp = compute_risk_points(ct, ticket.customer_tier)
        if dec.risk_points != expected_rp:
            errors.append(
                f"[{ct.ticket_id}] risk_points mismatch: "
                f"stored={dec.risk_points}, recomputed={expected_rp}"
            )

        expected_auto = compute_auto_send_eligible(ct, rs)
        if dec.auto_send_eligible != expected_auto:
            errors.append(
                f"[{ct.ticket_id}] auto_send_eligible mismatch: "
                f"stored={dec.auto_send_eligible}, recomputed={expected_auto}"
            )

        expected_priority = "urgent" if expected_rp >= 35 else "normal"
        if dec.priority != expected_priority:
            errors.append(
                f"[{ct.ticket_id}] priority mismatch: "
                f"stored={dec.priority}, recomputed={expected_priority}"
            )

    # -----------------------------------------------------------------
    # Send-gate consistency
    # -----------------------------------------------------------------
    for draft in drafts:
        dec = decision_map.get(draft.ticket_id)
        if not dec:
            continue
        expected_gate = "auto_send" if dec.auto_send_eligible else "human_review"
        if draft.send_gate != expected_gate:
            errors.append(
                f"[{draft.ticket_id}] send_gate mismatch: "
                f"stored='{draft.send_gate}', expected='{expected_gate}'"
            )

    # -----------------------------------------------------------------
    # Final summary consistency (only when the file exists)
    # -----------------------------------------------------------------
    if summary is not None:
        auto_ids = {d.ticket_id for d in decisions if d.auto_send_eligible}
        if set(summary.auto_send_ticket_ids) != auto_ids:
            errors.append(
                f"final_summary.auto_send_ticket_ids {sorted(summary.auto_send_ticket_ids)} "
                f"does not match decisions {sorted(auto_ids)}"
            )

        urgent_ids = {d.ticket_id for d in decisions if d.priority == "urgent"}
        if set(summary.urgent_ticket_ids) != urgent_ids:
            errors.append(
                f"final_summary.urgent_ticket_ids {sorted(summary.urgent_ticket_ids)} "
                f"does not match decisions {sorted(urgent_ids)}"
            )

        if summary.total_tickets != len(preprocessed):
            errors.append(
                f"final_summary.total_tickets={summary.total_tickets} "
                f"but preprocessed count={len(preprocessed)}"
            )

    if errors:
        raise ValidationError(errors)

    print(
        f"[validate] All checks passed for {len(preprocessed)} tickets "
        f"across {len(decisions)} decision records."
    )
