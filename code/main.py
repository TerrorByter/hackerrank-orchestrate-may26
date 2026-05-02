"""
Main — Orchestrator for the HackerRank Orchestrate support triage agent.

Reads support tickets from CSV, processes each through the agent pipeline,
and writes structured results to output.csv.

Usage:
    python main.py [--sample]

Flags:
    --sample    Run against sample_support_tickets.csv instead of support_tickets.csv
"""

import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from corpus_loader import load_corpus
from retriever import Retriever
from llm import LLMClient
from agent import process_ticket

# Load environment variables
load_dotenv()

# Paths relative to this file
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
TICKETS_DIR = PROJECT_ROOT / "support_tickets"
CACHE_DIR = DATA_DIR / "index"  # gitignored

# CSV files
SUPPORT_TICKETS = TICKETS_DIR / "support_tickets.csv"
SAMPLE_TICKETS = TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"

# Output CSV columns
OUTPUT_COLUMNS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type", "justification",
]


def read_tickets(filepath: Path) -> list[dict]:
    """Read support tickets from CSV."""
    tickets = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append({
                "issue": row.get("Issue", row.get("issue", "")).strip(),
                "subject": row.get("Subject", row.get("subject", "")).strip(),
                "company": row.get("Company", row.get("company", "")).strip(),
            })
    return tickets


def write_output(results: list[dict], filepath: Path):
    """Write agent results to output CSV."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\n✅ Output written to {filepath}")


def main():
    """Main entry point."""
    use_sample = "--sample" in sys.argv
    custom_file = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--file" and i + 1 < len(sys.argv)), None)

    print("=" * 60)
    print("  HackerRank Orchestrate — Support Triage Agent")
    print("=" * 60)
    print()

    # === Step 1: Load corpus ===
    print("[1/4] Loading support corpus...")
    corpus = load_corpus(str(DATA_DIR))
    print()

    # === Step 2: Build retriever ===
    print("[2/4] Building retriever index...")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    retriever = Retriever(model_name=embedding_model, cache_dir=str(CACHE_DIR))
    retriever.build_index(corpus)
    print()

    # === Step 3: Initialize LLM ===
    print("[3/4] Initializing LLM client...")
    llm = LLMClient()
    print(f"  Model: {llm.model}")
    print(f"  Base URL: {llm.base_url}")
    print()

    # === Step 4: Process tickets ===
    if custom_file:
        ticket_file = Path(custom_file)
    elif use_sample:
        ticket_file = SAMPLE_TICKETS
    else:
        ticket_file = SUPPORT_TICKETS
    print(f"[4/4] Processing tickets from {ticket_file.name}...")
    tickets = read_tickets(ticket_file)
    print(f"  Found {len(tickets)} tickets to process")
    print()

    results = []
    total_start = time.time()

    for i, ticket in enumerate(tickets, 1):
        issue = ticket["issue"]
        subject = ticket["subject"]
        company = ticket["company"]

        # Truncate display
        display_issue = issue[:80] + "..." if len(issue) > 80 else issue
        display_issue = display_issue.replace("\n", " ")
        print(f"  [{i}/{len(tickets)}] {display_issue}")

        try:
            start = time.time()
            result = process_ticket(
                issue=issue,
                subject=subject,
                company=company,
                retriever=retriever,
                llm=llm,
            )
            elapsed = time.time() - start

            # Add input columns to result
            result["issue"] = issue
            result["subject"] = subject
            result["company"] = company

            results.append(result)
            print(f"         → {result['status']} | {result['request_type']} | "
                  f"{result['product_area']} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"         ⚠ Error processing ticket: {e}")
            # Fallback: escalate on error
            results.append({
                "issue": issue,
                "subject": subject,
                "company": company,
                "status": "escalated",
                "product_area": "unknown",
                "response": "Unable to process this ticket automatically. Escalating to a human agent.",
                "justification": f"Agent encountered an error: {str(e)[:100]}",
                "request_type": "product_issue",
            })

    total_elapsed = time.time() - total_start

    # === Write output ===
    write_output(results, OUTPUT_CSV)

    # === Summary ===
    replied = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  Total tickets: {len(results)}")
    print(f"  Replied:       {replied}")
    print(f"  Escalated:     {escalated}")
    print(f"  Total time:    {total_elapsed:.1f}s")
    print(f"  Avg per ticket: {total_elapsed / len(results):.1f}s")
    print()


if __name__ == "__main__":
    main()
