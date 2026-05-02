"""
Retrieval benchmark: all-MiniLM-L6-v2 vs BAAI/bge-large-en-v1.5

Runs 8 representative queries (spanning all 3 domains) through both models
and prints top-3 results side by side so you can judge relevance manually.

Usage:
    python benchmark_retrieval.py

Output:
    Console table + benchmark_results.txt
"""

import os
import sys
import time
import textwrap
from pathlib import Path

# Make sure we can import from code/
sys.path.insert(0, str(Path(__file__).parent))

from corpus_loader import load_corpus
from retriever import Retriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = Path(__file__).resolve().parent / "benchmark_results.txt"

# ---------------------------------------------------------------------------
# 8 queries chosen to cover all 3 domains + varying difficulty.
# Excluded tickets that are escalated by hard rules (refund, score disputes,
# fraud, etc.) since retrieval quality doesn't affect those outcomes.
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "id": "HR-1",
        "domain": "hackerrank",
        "label": "Apply tab not visible",
        "text": "i can not able to see apply tab",
    },
    {
        "id": "HR-2",
        "domain": "hackerrank",
        "label": "Zoom compatibility blocker",
        "text": (
            "I am facing a blocker while doing compatible check, "
            "all the criterias are matching other than zoom"
        ),
    },
    {
        "id": "HR-3",
        "domain": "hackerrank",
        "label": "Inactivity timeout settings",
        "text": (
            "Can you please confirm the inactivity times currently set "
            "and are they different for candidate vs interviewer"
        ),
    },
    {
        "id": "HR-4",
        "domain": "hackerrank",
        "label": "Remove interviewer from platform",
        "text": (
            "I am trying to remove an interviewer from the platform. "
            "I am not seeing this as an option when I click on their profile"
        ),
    },
    {
        "id": "CL-1",
        "domain": "claude",
        "label": "Stop Claude crawling website",
        "text": "I want Claude to stop crawling my website",
    },
    {
        "id": "CL-2",
        "domain": "claude",
        "label": "Data retention for model training",
        "text": (
            "I am allowing Claude to use my data to improve the models, "
            "how long will the data be used for?"
        ),
    },
    {
        "id": "CL-3",
        "domain": "claude",
        "label": "AWS Bedrock requests failing",
        "text": (
            "I am facing multiple issues in my project, "
            "all requests to claude with aws bedrock is failing"
        ),
    },
    {
        "id": "CL-4",
        "domain": "claude",
        "label": "LTI key setup for students",
        "text": (
            "I am a professor in a college and wanted to setup "
            "a claude lti key for my students"
        ),
    },
    {
        "id": "VI-1",
        "domain": "visa",
        "label": "ATM / cash advance",
        "text": "I need urgent cash but don't have any right now & only the VISA card",
    },
    {
        "id": "VI-2",
        "domain": "visa",
        "label": "Minimum spend requirement",
        "text": (
            "I am in US Virgin Islands and the merchant is saying "
            "I have to spend minimum 10$ on my VISA card, is that right"
        ),
    },
]

MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-large-en-v1.5",
]

TOP_K = 3
WRAP = 72


def build_retriever(model_name: str, corpus) -> tuple[Retriever, float]:
    r = Retriever(model_name=model_name, cache_dir=None)  # no cache: force fresh embed
    t0 = time.time()
    r.build_index(corpus)
    elapsed = time.time() - t0
    return r, elapsed


def truncate(text: str, max_chars: int = 140) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


def run_benchmark(corpus) -> list[dict]:
    print(f"\nBuilding indexes for both models (this may take a minute)…\n")

    retrievers = {}
    index_times = {}
    for model in MODELS:
        short = model.split("/")[-1]
        print(f"  [{short}] embedding {len(corpus)} chunks…")
        r, t = build_retriever(model, corpus)
        retrievers[model] = r
        index_times[model] = t
        print(f"  [{short}] done in {t:.1f}s\n")

    results = []
    for q in QUERIES:
        row = {"query": q, "results": {}}
        for model in MODELS:
            t0 = time.time()
            hits = retrievers[model].retrieve(q["text"], domain=q["domain"], top_k=TOP_K)
            elapsed = time.time() - t0
            row["results"][model] = {"hits": hits, "query_ms": elapsed * 1000}
        results.append(row)

    return results, index_times


def format_results(results, index_times) -> str:
    lines = []

    header = (
        "=" * 90 + "\n"
        "  RETRIEVAL BENCHMARK: all-MiniLM-L6-v2  vs  BAAI/bge-large-en-v1.5\n"
        "=" * 90
    )
    lines.append(header)

    m1, m2 = MODELS
    s1, s2 = m1.split("/")[-1], m2.split("/")[-1]

    lines.append(f"\n  Index build time — {s1}: {index_times[m1]:.1f}s | {s2}: {index_times[m2]:.1f}s\n")

    for row in results:
        q = row["query"]
        lines.append("─" * 90)
        lines.append(f"  [{q['id']}] {q['label']}  (domain: {q['domain']})")
        lines.append(f"  Query: \"{q['text']}\"")
        lines.append("")

        h1 = row["results"][m1]["hits"]
        h2 = row["results"][m2]["hits"]
        t1 = row["results"][m1]["query_ms"]
        t2 = row["results"][m2]["query_ms"]

        # Column headers
        col_w = 42
        lines.append(f"  {'#':<3} {'MiniLM (score)':<22} {'bge-large (score)':<22}")
        lines.append(f"  {'':<3} {s1:<22} {s2:<22}")
        lines.append(f"  {'─'*3} {'─'*43} {'─'*43}")

        max_hits = max(len(h1), len(h2))
        for i in range(max_hits):
            left = right = "  (no result)"

            if i < len(h1):
                doc1, score1 = h1[i]
                breadcrumb1 = " > ".join(doc1.breadcrumbs[-2:]) if doc1.breadcrumbs else doc1.domain
                left = f"[{score1:.3f}] {doc1.title[:35]}\n{'':>8}{breadcrumb1[:40]}"
            if i < len(h2):
                doc2, score2 = h2[i]
                breadcrumb2 = " > ".join(doc2.breadcrumbs[-2:]) if doc2.breadcrumbs else doc2.domain
                right = f"[{score2:.3f}] {doc2.title[:35]}\n{'':>8}{breadcrumb2[:40]}"

            left_lines = left.split("\n")
            right_lines = right.split("\n")
            for j in range(max(len(left_lines), len(right_lines))):
                l = left_lines[j] if j < len(left_lines) else ""
                r = right_lines[j] if j < len(right_lines) else ""
                prefix = f"  {i+1}. " if j == 0 else "     "
                lines.append(f"{prefix}{l:<45}  {r}")

            lines.append("")

        lines.append(f"  Query latency — {s1}: {t1:.0f}ms | {s2}: {t2:.0f}ms")
        lines.append("")

    lines.append("=" * 90)
    lines.append("  MANUAL SCORING GUIDE")
    lines.append("=" * 90)
    lines.append("""
  For each query, rate each model's top-3 on relevance:
    3 = directly answers the question
    2 = related / useful context
    1 = tangentially related
    0 = irrelevant

  Sum the scores per model across all queries.
  A difference of 5+ points is a meaningful improvement.
""")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  Retrieval Model Benchmark")
    print("=" * 60)
    print(f"\n  Data dir : {DATA_DIR}")
    print(f"  Queries  : {len(QUERIES)} (spanning hackerrank, claude, visa)")
    print(f"  Top-K    : {TOP_K}")

    print(f"\nLoading corpus from {DATA_DIR}…")
    corpus = load_corpus(str(DATA_DIR))

    results, index_times = run_benchmark(corpus)

    output = format_results(results, index_times)
    print("\n" + output)

    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print(f"\nResults also saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
