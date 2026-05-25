# Replayable Support Pipeline

A deterministic, stage-driven pipeline that ingests support tickets and help-center articles, retrieves relevant evidence, classifies tickets, computes risk-based decisions in code, and drafts constrained support replies with safe escalation paths.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LLM access (optional)

Copy `.env.example` to `.env` and fill in your LiteLLM router details:

```bash
cp .env.example .env
# Edit .env with LITELLM_BASE_URL, LITELLM_API_KEY, LITELLM_MODEL
```

If no LLM is configured the pipeline runs in **stub/mock mode**: preprocessing passes through original text for non-English tickets, classification uses keyword heuristics, and reply drafting uses templates. All stage artifacts, validation, and deterministic decisioning still work fully.

### 3. Run the full pipeline

```bash
# With LLM (reads .env automatically)
python -m src.pipeline.main run

# Force stub mode (no LLM needed)
python -m src.pipeline.main run --mock-llm

# Custom input paths and output directory
python -m src.pipeline.main run \
  --tickets path/to/tickets.json \
  --articles path/to/articles.json \
  --output ./artifacts
```

### 4. Validate artifacts

```bash
python -m src.pipeline.main validate
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LITELLM_BASE_URL` | *(none)* | LiteLLM router URL, e.g. `http://localhost:4000/v1` |
| `LITELLM_API_KEY` | `dummy-key` | API key for the LiteLLM router |
| `LITELLM_MODEL` | `gpt-4o` | Model name passed to the router |
| `MOCK_LLM` | `false` | Set `true` to force stub mode regardless of other vars |

---

## Pipeline Stages

The pipeline enforces a strict forward-only state machine:

```
INIT
 -> INPUTS_LOADED              tickets.json + articles.json loaded and validated
 -> TICKETS_PREPROCESSED       non-English tickets translated via LLM (one call)
 -> CANDIDATE_ARTICLES_RETRIEVED  top-3 articles per ticket via TF-IDF cosine sim
 -> TICKETS_CLASSIFIED         Stage 1 LLM: classify all tickets in one call
 -> DECISIONS_COMPUTED         deterministic risk/score/eligibility — NO LLM
 -> REPLIES_DRAFTED            Stage 2 LLM: draft replies for all tickets
 -> VALIDATION_COMPLETE        cross-artifact schema + consistency checks
 -> RESULTS_FINALISED          final_summary.json written
```

---

## Artifact Files

All files are written to the output directory (default: `artifacts/`).

| File | Stage | Description |
|---|---|---|
| `preprocessed_tickets.json` | TICKETS_PREPROCESSED | Translated + normalized ticket records |
| `retrieval_results.json` | CANDIDATE_ARTICLES_RETRIEVED | Top-3 article candidates per ticket with TF-IDF scores |
| `classified_tickets.json` | TICKETS_CLASSIFIED | LLM-assigned issue type, answerability, risk flag, reply action |
| `decisions.json` | DECISIONS_COMPUTED | Deterministic: risk_points, decision_score, auto_send_eligible, priority |
| `draft_replies.json` | REPLIES_DRAFTED | Constrained LLM-drafted reply body, agent note, escalation team |
| `final_summary.json` | RESULTS_FINALISED | Counts, urgent IDs, auto-send IDs, top human-review IDs |
| `llm_calls.log.json` | (appended at each LLM stage) | Stage, timestamp, model, token counts, artifact paths |

---

## Controlled Vocabularies

All model outputs are validated against these values in code (Pydantic enums):

**issue_type:** `withdrawal_delay`, `deposit_missing`, `login_access`, `account_closure`, `duplicate_charge`, `verification_docs`, `trading_advice_request`, `other`

**answerability:** `auto_answer`, `needs_human`, `insufficient_context`

**risk_flag:** `none`, `financial`, `legal`, `compliance`, `security`

**reply_action:** `send_article_based_reply`, `request_missing_info`, `escalate_to_human`, `refuse_and_redirect`

**escalation_team:** `Customer Support`, `Payments`, `Compliance`, `Account Security`, `Legal`, `Data Privacy`

---

## Deterministic Decision Rules

The `decisions.json` stage is computed entirely in Python — no LLM is involved.

```
retrieval_score = score of best_article_id from retrieval_results (0 if absent)

risk_points:
  base by risk_flag: none=0, financial=20, legal=35, compliance=20, security=25
  +20  if contains_legal_threat
  +15  if contains_security_access_issue
  +10  if customer_tier = vip
  +10  if answerability = needs_human
  +15  if answerability = insufficient_context
  +10  if needs_more_info

decision_score = round((retrieval_score * 100) - risk_points)

auto_send_eligible = true only when ALL of:
  answerability = auto_answer
  reply_action in {send_article_based_reply, refuse_and_redirect}
  retrieval_score >= 0.65
  contains_legal_threat = false
  contains_security_access_issue = false
  risk_flag != legal

priority = "urgent" if risk_points >= 35 else "normal"
```

---

## Project Structure

```
.
├── tickets.json              # Input: support tickets
├── articles.json             # Input: help-center articles
├── artifacts/                # Generated pipeline JSON artifacts
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   └── pipeline/
│       ├── main.py           # CLI entrypoint + orchestrator
│       ├── stages.py         # Stage enum + state machine
│       ├── models.py         # Pydantic models + controlled vocab enums
│       ├── io_utils.py       # JSON read/write helpers
│       ├── llm_client.py     # OpenAI-compatible LiteLLM client
│       ├── llm_logging.py    # Append-only LLM call logger
│       ├── preprocess.py     # Multilingual preprocessing / translation
│       ├── retrieval.py      # TF-IDF top-3 article retrieval
│       ├── classification.py # Stage 1 LLM classification
│       ├── decisioning.py    # Deterministic risk/score/eligibility
│       ├── reply_drafting.py # Stage 2 LLM reply drafting
│       └── validators.py     # Cross-artifact validation
└── tests/
    └── test_decisioning.py   # Unit tests for deterministic decisioning
```

---

## Evaluator Notes

- The pipeline reads only from `tickets.json` and `articles.json`; no hardcoded ticket IDs or expected outcomes.
- Retrieval is fully deterministic: TF-IDF with fixed preprocessing, score-then-ID tie-breaking.
- All final eligibility and priority decisions are computed in `decisioning.py` with no model involvement.
- Reply drafting receives the decision output and is explicitly constrained by documented rules (no legal admissions, no password requests, trading advice refused).
- The `validate` command re-derives decisioning results from scratch and checks parity against stored artifacts.
