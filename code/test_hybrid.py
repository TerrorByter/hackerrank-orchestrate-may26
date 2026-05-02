"""
Targeted test: BM25 hybrid vs pure semantic (bge-large) on the 3 queries
where pure bge-large underperformed MiniLM in the benchmark.

Uses the cached bge-large index so re-embedding is not needed.

Usage:
    python test_hybrid.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from corpus_loader import load_corpus
from retriever import Retriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "index"

MODEL = "BAAI/bge-large-en-v1.5"

PROBLEM_QUERIES = [
    {
        "id": "CL-2",
        "label": "Data retention for model training",
        "domain": "claude",
        "text": "I am allowing Claude to use my data to improve the models, how long will the data be used for?",
        "expected_signal": "privacy / training data",
        "wrong_result": "Use Claude for Excel / Word",
    },
    {
        "id": "CL-3",
        "label": "AWS Bedrock requests failing",
        "domain": "claude",
        "text": "I am facing multiple issues in my project, all requests to claude with aws bedrock is failing",
        "expected_signal": "Amazon Bedrock docs",
        "wrong_result": "Troubleshoot Claude Code installation",
    },
    {
        "id": "VI-2",
        "label": "Merchant minimum spend requirement",
        "domain": "visa",
        "text": "I am in US Virgin Islands and the merchant is saying I have to spend minimum 10$ on my VISA card, is that right",
        "expected_signal": "Visa rules / merchant minimum",
        "wrong_result": "Travel Services / Fraud Prevention",
    },
]

TOP_K = 3


def build_retriever(corpus) -> Retriever:
    r = Retriever(model_name=MODEL, cache_dir=str(CACHE_DIR))
    print(f"  Building index (will use cache if available)...")
    t0 = time.time()
    r.build_index(corpus)
    print(f"  Ready in {time.time() - t0:.1f}s\n")
    return r


def run_query(retriever: Retriever, query: dict, use_bm25: bool) -> list:
    """Run a single query. Temporarily disables BM25 if use_bm25=False."""
    saved_bm25 = retriever.bm25
    if not use_bm25:
        retriever.bm25 = None

    t0 = time.time()
    hits = retriever.retrieve(query["text"], domain=query["domain"], top_k=TOP_K)
    elapsed_ms = (time.time() - t0) * 1000

    retriever.bm25 = saved_bm25
    return hits, elapsed_ms


def fmt_hit(doc, score) -> str:
    breadcrumb = " > ".join(doc.breadcrumbs[-2:]) if doc.breadcrumbs else doc.domain
    return f"[{score:.4f}] {doc.title[:45]:<45}  ({breadcrumb[:40]})"


def main():
    print("=" * 72)
    print("  Hybrid BM25 + bge-large vs pure bge-large — targeted test")
    print("=" * 72)
    print(f"\n  Queries  : {len(PROBLEM_QUERIES)} (the 3 cases where pure bge-large failed)")
    print(f"  Model    : {MODEL}")
    print(f"  Cache    : {CACHE_DIR}\n")

    print("Loading corpus...")
    corpus = load_corpus(str(DATA_DIR))
    print()

    retriever = build_retriever(corpus)

    verdicts = []

    for q in PROBLEM_QUERIES:
        print("─" * 72)
        print(f"  [{q['id']}] {q['label']}")
        print(f"  Query  : \"{q['text']}\"")
        print(f"  Expect : {q['expected_signal']}")
        print(f"  Old #1 : {q['wrong_result']} (pure bge-large from benchmark)")
        print()

        pure_hits, pure_ms = run_query(retriever, q, use_bm25=False)
        hybrid_hits, hybrid_ms = run_query(retriever, q, use_bm25=True)

        print(f"  {'#':<3} {'Pure bge-large':<55}  {'Hybrid (bge + BM25)'}")
        print(f"  {'─'*3} {'─'*55}  {'─'*55}")

        for i in range(TOP_K):
            left = fmt_hit(*pure_hits[i]) if i < len(pure_hits) else "(no result)"
            right = fmt_hit(*hybrid_hits[i]) if i < len(hybrid_hits) else "(no result)"
            print(f"  {i+1}.  {left}  {right}")

        print(f"\n  Latency — pure: {pure_ms:.0f}ms | hybrid: {hybrid_ms:.0f}ms")

        # Auto-verdict: did hybrid fix the top result?
        pure_top = pure_hits[0][0].title if pure_hits else ""
        hybrid_top = hybrid_hits[0][0].title if hybrid_hits else ""
        changed = pure_top != hybrid_top
        verdicts.append({"id": q["id"], "changed": changed, "hybrid_top": hybrid_top})
        print(f"  Top-1 changed: {'YES ✓' if changed else 'NO — same result'}")
        print()

    # Summary
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for v in verdicts:
        status = "FIXED" if v["changed"] else "unchanged"
        print(f"  {v['id']}: {status} → top result now: \"{v['hybrid_top'][:60]}\"")

    fixed = sum(1 for v in verdicts if v["changed"])
    print(f"\n  {fixed}/{len(verdicts)} queries have a different (hopefully better) top result with hybrid.")
    print()


if __name__ == "__main__":
    main()
