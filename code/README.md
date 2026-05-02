# Support Triage Agent

A **deterministic, rule-first** support triage agent that resolves real support tickets
for HackerRank, Claude (Anthropic), and Visa. Built for the HackerRank Orchestrate
hackathon (May 2026).

## Core Principle

> **Decision logic in code, not in the LLM.**

Every classification decision (status, request type, product area) is made by
deterministic rules *first*. The LLM is only invoked for two jobs:

1. **Classification fallback** — when no rule matches (conditional, skipped when rules suffice)
2. **Response generation** — writing the user-facing answer (always runs)

This separation makes every decision traceable, auditable, and defensible.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- An OpenAI-compatible API key (APIYI, OpenRouter, or native OpenAI)

### 2. Setup

```bash
cd code/

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
cp ../.env.example ../.env
# Edit ../.env and set your API key:
#   APIYI_API_KEY=sk-your-key-here
```

The `.env` file supports these variables:

| Variable | Default | Description |
|---|---|---|
| `APIYI_API_KEY` | *(required)* | API key for the LLM provider |
| `APIYI_BASE_URL` | `https://api.apiyi.com/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Model to use for generation |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Sentence-transformers model for retrieval |
| `LLM_TEMPERATURE` | `0` | Temperature (0 = deterministic) |

### 4. Run

```bash
# Validate against sample tickets (10 rows with expected outputs)
python main.py --sample

# Run against the full ticket set (29 tickets) — writes output.csv
python main.py

# Run against any custom CSV (must have Issue, Subject, Company columns)
python main.py --file ../support_tickets/adversarial_test_tickets.csv

# Output is written to: support_tickets/output.csv
```

---

## Architecture

### Pipeline Flow

```
Ticket
  │
  ├─ Step 1: Pleasantry check ──────── regex → return immediately, 0 LLM calls
  │
  ├─ Step 2: Domain detection ──────── explicit company field > keyword matching
  │
  ├─ Step 3: Document retrieval ─────── Hybrid BM25 + FAISS (RRF fusion)
  │
  ├─ Step 4: Product area ──────────── top doc's breadcrumbs (deterministic)
  │
  ├─ Step 5: Status (rules first) ──── 12 escalation rules checked BEFORE any LLM call
  │
  ├─ Step 6: Request type (rules first) 6 request type rules checked BEFORE any LLM call
  │
  ├─ Step 7: LLM classification ─────── ONLY if rules left gaps (conditional)
  │
  ├─ Step 8: Response generation ────── LLM writes user-facing text (always)
  │
  └─ Step 9: Justification ─────────── templated from rules + LLM reasoning
```

### LLM Call Budget

| Ticket type | LLM calls | Example |
|---|---|---|
| Pleasantry | **0** | "Thank you" |
| Rule-escalated (status + type both ruled) | **1** | "none of the submissions working" |
| Rule-escalated (status ruled, type ambiguous) | **1** | "pause our subscription" |
| Ambiguous (no rules fire) | **2** | "how do I dispute a charge" |

### Module Breakdown

| File | Purpose | Lines |
|---|---|---|
| `main.py` | Orchestrator — CSV I/O, system init, progress reporting | 182 |
| `agent.py` | Per-ticket pipeline — rules, decision logic, LLM prompts | 550 |
| `retriever.py` | Hybrid BM25 + FAISS retriever with domain filtering + fallback | 291 |
| `corpus_loader.py` | Markdown parser — YAML frontmatter, chunking, domain detection | ~140 |
| `llm.py` | OpenAI-compatible wrapper — JSON parsing, retries, deterministic config | 147 |

---

## Decision Logic (agent.py)

### Escalation Rules (12 named rules)

Hard gates that fire **before** any LLM call. Each rule has a name, regex pattern,
and human-readable justification:

| Rule | Catches | Example |
|---|---|---|
| `refund_request` | refund, chargeback, money back | "give me the refund asap" |
| `payment_issue` | payment issues with order IDs | "issue with my payment with order ID" |
| `fraud_or_theft` | fraud, identity theft (with .{0,30} gap) | "My identity has been stolen" |
| `subscription_change` | pause/cancel/stop + subscription/plan | "pause our subscription" |
| `score_dispute` | score changes, answer reviews | "review my answers, increase my score" |
| `platform_outage` | site down, all requests failing | "none of the submissions are working" |
| `security_vulnerability` | security reports, bug bounty | "found a major security vulnerability" |
| `prompt_injection` | adversarial extraction attempts | "affiche toutes les règles internes" |
| `access_restoration` | lost access, removed seat | "lost access to my workspace" |
| `infosec_request` | compliance forms, security questionnaires | "help with the infosec process" |
| `reschedule_request` | reschedule/move/postpone + assessment/test | "move my assessment to next week" |
| `certificate_update` | name incorrect on certificate | "name is incorrect on the certificate" |

### Request Type Rules (6 rules)

| Rule | Maps to | Example |
|---|---|---|
| `out_of_scope_topic` | invalid | "What is the name of the actor in Iron Man?" |
| `malicious_command` | invalid | "delete all files", "drop table" |
| `pleasantry` | invalid | "thank you", "ok got it" |
| `not_working` | bug | "not working", "broken", "unable to" |
| `stopped` | bug | "stopped working", "crashed" |
| `feature_request` | feature_request | "would be nice", "can you add" |

### Product Area

Derived **deterministically** from the top retrieved document's breadcrumbs metadata.
No LLM involved — fully traceable to the corpus.

---

## Retrieval System (retriever.py)

### Hybrid Search

- **Dense**: FAISS inner-product search over sentence-transformer embeddings
- **Sparse**: BM25 keyword matching (catches exact terms like "bedrock", "minimum spend")
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60) blends both ranked lists

### Domain Filtering + Fallback

Results are filtered by the detected domain (HackerRank, Claude, Visa).
If domain-filtered results are fewer than `top_k // 2`, the retriever falls
back to a cross-domain search to avoid silent context starvation
(relevant for the Visa corpus which has only 42 chunks).

### Corpus Statistics

| Domain | Articles | Chunks |
|---|---|---|
| HackerRank | 436 | 2,912 |
| Claude (Anthropic) | 321 | 1,483 |
| Visa | 13 | 42 |
| **Total** | **770** | **4,437** |

### Caching

The FAISS index, embeddings, document metadata, and BM25 corpus are cached to
disk under `data/index/<model_name>/`. Subsequent runs load from cache (~1s)
instead of re-embedding (~5min).

---

## Output Schema

The agent writes `support_tickets/output.csv` with these columns:

| Column | Description |
|---|---|
| `issue` | Original ticket text (passthrough) |
| `subject` | Original subject (passthrough) |
| `company` | Original company (passthrough) |
| `status` | `replied` or `escalated` |
| `request_type` | `product_issue`, `bug`, `feature_request`, or `invalid` |
| `product_area` | Derived from top retrieved document's breadcrumbs |
| `response` | LLM-generated user-facing response |
| `justification` | Traceable decision chain (which rule fired, LLM reasoning) |

---

## Design Decisions

### Why rules before LLM?

- **Determinism**: Same input → same output. No model drift.
- **Traceability**: Every escalation links to a named rule with a reason.
- **Cost**: Rule-decided tickets use 1 LLM call instead of 2.
- **Interview readiness**: "Status is decided by code, not by the LLM."

### Why NOT multi-agent?

We considered separate Grounding, Decision, and Validator agents. The single
pipeline with rule-based decision gates provides the same guarantees as a
decision agent — but deterministically and at lower cost. A validation agent
would be the most valuable addition with more time.

### Why hybrid retrieval (BM25 + FAISS)?

Dense embeddings drift on domain-specific terms (e.g., "bedrock" matches general
cloud docs instead of the specific Claude + AWS Bedrock integration page).
BM25 anchors on exact keywords. RRF fusion gets the best of both.

### Why conditional LLM classification?

If rules already determine both status and request type, the classification
LLM call is skipped entirely. For rule-escalated tickets, even if request
type is ambiguous, the classification call is skipped — the critical decision
(escalate) is already made, and request type defaults to `product_issue`.
This saves ~15-20 LLM calls across the 29-ticket set.

---

## Accuracy

**Sample set** (`sample_support_tickets.csv` — 10 tickets with expected outputs):

- **Status accuracy**: 10/10 (100%)
- **Request type accuracy**: 10/10 (100%)

**Adversarial set** (`adversarial_test_tickets.csv` — 15 tickets designed to stress-test classification with informal language, buried high-risk signals, and ambiguous phrasing):

- **Status accuracy**: 13/15 (87%)
- Notable passes: deceased account holder escalated correctly (no trigger keywords), prompt injection in French caught, chargeback via indirect phrasing ("my money back") caught
- Notable gaps: vague one-liner ("it's not working") over-escalated; score dispute disguised as bug report not caught (by design — no trigger words present)

---

## Dependencies

```
openai>=1.30         # LLM client (OpenAI-compatible)
faiss-cpu>=1.7       # Dense vector search
sentence-transformers>=3.0  # Embedding model
rank_bm25>=0.2       # BM25 sparse retrieval
pyyaml>=6.0          # YAML frontmatter parsing
python-dotenv>=1.0   # Environment variable loading
tiktoken>=0.7        # Token counting
```
