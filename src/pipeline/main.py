"""Support Pipeline – CLI entrypoint.

Usage:
  python -m src.pipeline.main run [options]
  python -m src.pipeline.main validate [options]

Run options:
  --tickets   PATH   Path to tickets.json   (default: tickets.json)
  --articles  PATH   Path to articles.json  (default: articles.json)
  --output    DIR    Directory for artifacts (default: artifacts)
  --mock-llm         Use stub mode even if LLM env vars are set

Validate options:
  --output    DIR    Directory containing artifacts to validate (default: artifacts)

Environment variables (for LLM integration):
  LITELLM_BASE_URL   LiteLLM router base URL, e.g. http://localhost:4000/v1
  LITELLM_API_KEY    API key for the router
  LITELLM_MODEL      Model name, e.g. qwen-max, gpt-4o
  MOCK_LLM           Set to "true" to force stub mode
"""

import argparse
import sys
from pathlib import Path

# Load .env if present (optional, never required)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .classification import classify_tickets
from .decisioning import compute_decisions
from .io_utils import read_json, write_json
from .llm_client import get_llm_client
from .llm_logging import log_llm_call
from .models import (
    Decision,
    FinalSummary,
    PreprocessedTicket,
    RawArticle,
    RawTicket,
)
from .preprocess import preprocess_tickets
from .reply_drafting import draft_replies
from .retrieval import retrieve_candidates
from .stages import PipelineStage, PipelineState
from .validators import ValidationError, validate_artifacts


# ---------------------------------------------------------------------------
# Final summary builder
# ---------------------------------------------------------------------------

def build_final_summary(
    decisions: list[Decision],
    preprocessed: list[PreprocessedTicket],
) -> FinalSummary:
    auto_ids = sorted([d.ticket_id for d in decisions if d.auto_send_eligible])
    urgent_ids = sorted([d.ticket_id for d in decisions if d.priority == "urgent"])

    human_review_ids = [d.ticket_id for d in decisions if not d.auto_send_eligible]
    # Top 3: sort human-review tickets by risk_points desc, then decision_score asc
    human_decision_map = {d.ticket_id: d for d in decisions if not d.auto_send_eligible}
    top_human = sorted(
        human_review_ids,
        key=lambda tid: (
            -human_decision_map[tid].risk_points,
            human_decision_map[tid].decision_score,
        ),
    )[:3]

    return FinalSummary(
        total_tickets=len(preprocessed),
        auto_send_count=len(auto_ids),
        human_review_count=len(human_review_ids),
        urgent_ticket_ids=urgent_ids,
        auto_send_ticket_ids=auto_ids,
        top_human_review_ticket_ids=top_human,
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    tickets_path: Path,
    articles_path: Path,
    output_dir: Path,
    mock_llm: bool = False,
) -> FinalSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = PipelineState()

    # ── INIT -> INPUTS_LOADED ────────────────────────────────────────────
    print("\n[pipeline] === STAGE: INPUTS_LOADED ===")
    raw_tickets = [RawTicket(**t) for t in read_json(tickets_path)]
    raw_articles = [RawArticle(**a) for a in read_json(articles_path)]
    print(
        f"[pipeline] Loaded {len(raw_tickets)} ticket(s), "
        f"{len(raw_articles)} article(s)."
    )
    state.advance(PipelineStage.INPUTS_LOADED)

    # Set up LLM client (None triggers stub mode everywhere)
    import os
    force_mock = mock_llm or os.environ.get("MOCK_LLM", "").lower() in ("1", "true", "yes")
    client = None if force_mock else get_llm_client()

    articles_map = {a.article_id: a for a in raw_articles}

    # ── INPUTS_LOADED -> TICKETS_PREPROCESSED ───────────────────────────
    print("\n[pipeline] === STAGE: TICKETS_PREPROCESSED ===")
    preprocessed, pp_tokens, pp_comp = preprocess_tickets(raw_tickets, client)
    write_json(output_dir / "preprocessed_tickets.json", [t.model_dump() for t in preprocessed])
    if pp_tokens or pp_comp:
        log_llm_call(
            stage="TICKETS_PREPROCESSED",
            model=client.model if client else "stub",
            input_artifacts=["tickets.json"],
            output_artifact="preprocessed_tickets.json",
            prompt_tokens=pp_tokens,
            completion_tokens=pp_comp,
            output_dir=output_dir,
        )
    state.advance(PipelineStage.TICKETS_PREPROCESSED)

    # ── TICKETS_PREPROCESSED -> CANDIDATE_ARTICLES_RETRIEVED ────────────
    print("\n[pipeline] === STAGE: CANDIDATE_ARTICLES_RETRIEVED ===")
    retrieval_results = retrieve_candidates(preprocessed, raw_articles)
    write_json(
        output_dir / "retrieval_results.json",
        [r.model_dump() for r in retrieval_results],
    )
    state.advance(PipelineStage.CANDIDATE_ARTICLES_RETRIEVED)

    # ── CANDIDATE_ARTICLES_RETRIEVED -> TICKETS_CLASSIFIED ───────────────
    print("\n[pipeline] === STAGE: TICKETS_CLASSIFIED ===")
    classified, cl_tokens, cl_comp = classify_tickets(
        preprocessed, retrieval_results, articles_map, client
    )
    write_json(
        output_dir / "classified_tickets.json",
        [c.model_dump() for c in classified],
    )
    log_llm_call(
        stage="TICKETS_CLASSIFIED",
        model=client.model if client else "stub",
        input_artifacts=["preprocessed_tickets.json", "retrieval_results.json"],
        output_artifact="classified_tickets.json",
        prompt_tokens=cl_tokens,
        completion_tokens=cl_comp,
        output_dir=output_dir,
    )
    state.advance(PipelineStage.TICKETS_CLASSIFIED)

    # ── TICKETS_CLASSIFIED -> DECISIONS_COMPUTED ─────────────────────────
    print("\n[pipeline] === STAGE: DECISIONS_COMPUTED ===")
    decisions = compute_decisions(classified, retrieval_results, preprocessed)
    write_json(
        output_dir / "decisions.json",
        [d.model_dump() for d in decisions],
    )
    state.advance(PipelineStage.DECISIONS_COMPUTED)

    # ── DECISIONS_COMPUTED -> REPLIES_DRAFTED ────────────────────────────
    print("\n[pipeline] === STAGE: REPLIES_DRAFTED ===")
    draft_list, dr_tokens, dr_comp = draft_replies(
        preprocessed, classified, decisions, articles_map, client
    )
    write_json(
        output_dir / "draft_replies.json",
        [d.model_dump() for d in draft_list],
    )
    log_llm_call(
        stage="REPLIES_DRAFTED",
        model=client.model if client else "stub",
        input_artifacts=[
            "preprocessed_tickets.json",
            "classified_tickets.json",
            "decisions.json",
        ],
        output_artifact="draft_replies.json",
        prompt_tokens=dr_tokens,
        completion_tokens=dr_comp,
        output_dir=output_dir,
    )
    state.advance(PipelineStage.REPLIES_DRAFTED)

    # ── REPLIES_DRAFTED -> VALIDATION_COMPLETE ───────────────────────────
    print("\n[pipeline] === STAGE: VALIDATION_COMPLETE ===")
    try:
        validate_artifacts(output_dir, require_final_summary=False)
    except ValidationError as exc:
        print(f"[validate] WARNING – {len(exc.errors)} issue(s) found:")
        for err in exc.errors:
            print(f"  ✗ {err}")
        print("[validate] Pipeline completed with validation warnings.")
    state.advance(PipelineStage.VALIDATION_COMPLETE)

    # ── VALIDATION_COMPLETE -> RESULTS_FINALISED ─────────────────────────
    print("\n[pipeline] === STAGE: RESULTS_FINALISED ===")
    summary = build_final_summary(decisions, preprocessed)
    write_json(output_dir / "final_summary.json", summary.model_dump())
    state.advance(PipelineStage.RESULTS_FINALISED)

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total tickets       : {summary.total_tickets}")
    print(f"  Auto-send eligible  : {summary.auto_send_count}  {summary.auto_send_ticket_ids}")
    print(f"  Human review        : {summary.human_review_count}")
    print(f"  Urgent tickets      : {summary.urgent_ticket_ids}")
    print(f"  Top human review    : {summary.top_human_review_ticket_ids}")
    print("=" * 60 + "\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.main",
        description="Replayable support-ticket pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Execute the full pipeline")
    run_p.add_argument(
        "--tickets",
        default="tickets.json",
        metavar="PATH",
        help="Path to tickets.json (default: tickets.json)",
    )
    run_p.add_argument(
        "--articles",
        default="articles.json",
        metavar="PATH",
        help="Path to articles.json (default: articles.json)",
    )
    run_p.add_argument(
        "--output",
        default="artifacts",
        metavar="DIR",
        help="Directory to write artifact files (default: artifacts)",
    )
    run_p.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use stub/template mode instead of live LLM calls",
    )

    # validate
    val_p = sub.add_parser("validate", help="Validate existing pipeline artifacts")
    val_p.add_argument(
        "--output",
        default="artifacts",
        metavar="DIR",
        help="Directory containing artifact files (default: artifacts)",
    )

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            run_pipeline(
                tickets_path=Path(args.tickets),
                articles_path=Path(args.articles),
                output_dir=Path(args.output),
                mock_llm=args.mock_llm,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n[ERROR] Pipeline aborted: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "validate":
        try:
            validate_artifacts(Path(args.output))
            print("[validate] All artifacts are valid.")
        except ValidationError as exc:
            print(f"[validate] FAILED – {len(exc.errors)} error(s):", file=sys.stderr)
            for err in exc.errors:
                print(f"  ✗ {err}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[validate] ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
