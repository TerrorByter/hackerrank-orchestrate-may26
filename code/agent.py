"""
Agent — per-ticket processing pipeline with decomposed decision logic.

Architecture principle: DECISION LOGIC IN CODE, NOT IN THE LLM.

The pipeline separates concerns clearly:
  - Status (replied/escalated)    → Rule-based hard gate first, LLM only for ambiguous
  - Request type                  → Rule-based classification, LLM only for ambiguous
  - Product area                  → Deterministic from retrieved doc breadcrumbs
  - Response                      → LLM generates (this IS what LLMs are for)
  - Justification                 → Templated from rules + LLM reasoning
"""

import re
from dataclasses import dataclass
from typing import Optional

from corpus_loader import Document
from retriever import Retriever
from llm import LLMClient


# =============================================================================
# ESCALATION RULES — Hard gate, deterministic, no LLM involved
# =============================================================================

@dataclass
class EscalationRule:
    """A named rule that triggers hard escalation."""
    name: str
    pattern: str  # regex pattern
    reason: str   # human-readable justification

ESCALATION_RULES = [
    # Financial / billing — broadened to catch natural phrasings
    EscalationRule(
        name="refund_request",
        pattern=r"\b(refund|charge\s*back|money\s+back|get\s+my\s+money|want\s+my\s+money)\b",
        reason="Refund/chargeback requests require human review of billing records.",
    ),
    EscalationRule(
        name="payment_issue",
        pattern=r"\b(payment\s+issue|payment.*order\s+id|billing\s+issue|issue\s+with\s+my\s+payment)\b",
        reason="Payment issues with specific order IDs require human investigation.",
    ),
    EscalationRule(
        name="fraud_or_theft",
        # Allow up to 30 chars between "identity" and "theft/stolen" to catch
        # natural phrasings like "My identity has been stolen".
        # Bare "stolen" without nearby "identity" does NOT match (deliberate).
        pattern=r"\b(fraud|identity.{0,30}(theft|stolen)|(theft|stolen).{0,30}identity|unauthorized\s+(charge|transaction|access))\b",
        reason="Fraud and identity theft cases require immediate human intervention.",
    ),
    # Account/subscription admin changes — allow words between pause/cancel and subscription
    EscalationRule(
        name="subscription_change",
        pattern=r"\b(pause|cancel|stop|end|terminate)\b.*\b(subscription|plan|billing)\b",
        reason="Subscription changes require account admin access.",
    ),
    # Score disputes / overrides
    EscalationRule(
        name="score_dispute",
        pattern=r"\b(increase\s+my\s+score|change\s+my\s+(score|grade)|review\s+my\s+answers|graded.*unfairly|move\s+me\s+to\s+the\s+next\s+round)\b",
        reason="Score disputes and override requests cannot be handled automatically.",
    ),
    # Platform-wide outages
    EscalationRule(
        name="platform_outage",
        pattern=r"\b(site\s+is\s+down|stopped\s+working\s+completely|all\s+requests?\s+(are\s+)?failing|none\s+of\s+the\s+(submissions?|pages?))\b",
        reason="Platform-wide outages require engineering team escalation.",
    ),
    # Security
    EscalationRule(
        name="security_vulnerability",
        pattern=r"\b(security\s+vulnerability|bug\s+bounty|found\s+a\s+(major\s+)?security)\b",
        reason="Security vulnerability reports must be escalated to the security team.",
    ),
    # Prompt injection / adversarial
    EscalationRule(
        name="prompt_injection",
        pattern=r"\b(affiche\s+toutes\s+les\s+r[eè]gles|show.*internal\s+rules|display.*rules|system\s+prompt|logique.*exacte)\b",
        reason="Request attempts to extract internal system information.",
    ),
    # Account access restoration by non-admin
    EscalationRule(
        name="access_restoration",
        pattern=r"\b(restore\s+my\s+access|lost\s+access.*workspace|removed\s+my\s+seat)\b",
        reason="Account access restoration requires admin or workspace owner action.",
    ),
    # Infosec / compliance requests
    EscalationRule(
        name="infosec_request",
        pattern=r"\b(infosec\s+process|security\s+questionnaire|compliance\s+form|fill.*forms)\b",
        reason="InfoSec/compliance requests require direct engagement from the security team.",
    ),
    # Reschedule requests — broadened to catch "move/postpone my assessment/test"
    EscalationRule(
        name="reschedule_request",
        pattern=r"\b(reschedul(e|ing)|move|postpone)\b.*\b(assessment|test|interview|exam)\b",
        reason="Assessment rescheduling must be handled by the hiring company, not the platform.",
    ),
    # Certificate changes
    EscalationRule(
        name="certificate_update",
        pattern=r"\b(name\s+is\s+incorrect\s+on\s+the\s+certificate|update.*certificate)\b",
        reason="Certificate corrections require manual verification and update by support staff.",
    ),
]


# =============================================================================
# REQUEST TYPE RULES — Deterministic classification
# =============================================================================

@dataclass
class RequestTypeRule:
    """A named rule for classifying request type."""
    name: str
    pattern: str
    request_type: str

REQUEST_TYPE_RULES = [
    # Invalid / out of scope
    RequestTypeRule("out_of_scope_topic", r"\b(iron\s+man|movie|actor|weather|recipe|sports)\b", "invalid"),
    RequestTypeRule("malicious_command", r"\b(delete\s+all\s+files|drop\s+table|rm\s+-rf|format\s+c:)\b", "invalid"),
    RequestTypeRule("pleasantry", r"^(thank\s*you|thanks|thx|cheers|ok|okay|got\s+it)[\s\.\!]*$", "invalid"),

    # Bug signals
    RequestTypeRule("not_working", r"\b(not\s+working|broken|error|failing|can\s*not|unable\s+to|blocker|down)\b", "bug"),
    RequestTypeRule("stopped", r"\b(stopped\s+working|crashed|freezing|stuck)\b", "bug"),

    # Feature request signals
    RequestTypeRule("feature_request", r"\b(feature\s+request|would\s+be\s+nice|can\s+you\s+add|wish.*had)\b", "feature_request"),
]


# =============================================================================
# PLEASANTRY PATTERNS — Handled without LLM
# =============================================================================

PLEASANTRY_PATTERNS = [
    r"^(thank\s*you|thanks|thx|cheers|appreciate\s+it|appreciate)[\s\.\!\,]*$",
    r"^(ok|okay|got\s+it|understood|great|perfect)[\s\.\!\,]*$",
    r"^(hi|hello|hey)[\s\.\!\,]*$",
    r"^(thank\s+you\s+for\s+helping\s+me)[\s\.\!\,]*$",
]


# =============================================================================
# SYSTEM PROMPT — LLM's ONLY job: write the response text
# =============================================================================

RESPONSE_GENERATION_PROMPT = """You are a support agent. Your ONLY job is to write a helpful, user-facing response to a support ticket.

IMPORTANT RULES:
- Use ONLY the provided documentation to write your response. Do NOT use your general knowledge.
- If the documentation doesn't contain enough information, say so honestly.
- Be concise but helpful. Include specific steps when available.
- NEVER invent policies, URLs, phone numbers, or procedures not in the docs.
- If the ticket is in a non-English language, respond in English.
- For out-of-scope or invalid requests, politely explain that it's outside your support scope.

Respond with ONLY the response text. No JSON, no metadata, just the response."""


CLASSIFICATION_PROMPT = """You are a support ticket classifier. Analyze this ticket and the retrieved documentation.

Respond with ONLY valid JSON (no code fences, no extra text) with exactly these fields:
{
  "request_type": "product_issue" or "feature_request" or "bug" or "invalid",
  "should_escalate": true or false,
  "escalation_reason": "reason if escalating, empty string otherwise",
  "brief_justification": "1-2 sentence explanation of your classification"
}

Rules for request_type:
- "product_issue": how-to questions, configuration help, general product questions
- "feature_request": user wants something that doesn't exist
- "bug": something is broken, errors, not working as expected
- "invalid": off-topic, spam, pleasantries, or completely unrelated to ALL THREE products.
  NEVER use "invalid" for security concerns, suspicious account activity, or access issues — those are always "product_issue" even when the user is uncertain or indirect.
  NOTE: questions about Visa card features (ATM, cash, travel, disputes, merchant rules) are ALWAYS "product_issue", even if phrased unusually or with urgency.

Rules for should_escalate:
- Default to false. ONLY set to true when the user's request genuinely cannot be answered using the retrieved documentation.
- If the retrieved docs contain relevant information (even partially), set should_escalate=false — the agent CAN reply with what's available.
- Set should_escalate=true ONLY for: requests requiring admin/backend access, account changes the user cannot self-serve, or when the docs are completely irrelevant to the question.
- ALWAYS set should_escalate=true if the user reports fraud, identity theft, stolen credentials, or unauthorised account access — even if phrased indirectly or with spelling errors.
- ALWAYS set should_escalate=true if the user describes a charge, transaction, or account activity they do not recognise or cannot explain — even if the amount is small or the user seems uncertain."""


# =============================================================================
# DOMAIN DETECTION — Deterministic
# =============================================================================

DOMAIN_KEYWORDS = {
    "hackerrank": [
        "hackerrank", "test", "assessment", "candidate", "interviewer",
        "screen", "coding", "interview", "proctor", "proctoring",
        "skillup", "library", "question", "recruit", "hiring",
        "mock interview", "certificate", "resume builder", "apply tab",
        "submission", "challenge",
    ],
    "claude": [
        "claude", "anthropic", "conversation", "artifact", "claude code",
        "bedrock", "workspace", "team plan", "enterprise plan", "pro plan",
        "max plan", "lti", "crawl",
    ],
    "visa": [
        "visa", "card", "merchant", "atm", "cheque", "traveller",
        "transaction", "payment card", "carte", "tarjeta",
    ],
}


def determine_domain(company: str, issue: str, subject: str) -> Optional[str]:
    """
    Determine domain deterministically.
    Priority: explicit company field > keyword matching > None (search all).
    """
    # 1. Explicit company field
    if company and company.strip().lower() not in ("none", ""):
        company_lower = company.strip().lower()
        for domain in ("hackerrank", "claude", "visa"):
            if domain in company_lower:
                return domain

    # 2. Keyword matching on issue + subject
    combined = f"{issue} {subject}".lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in combined)

    if max(scores.values()) > 0:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best

    return None  # Search all domains


# =============================================================================
# PRODUCT AREA — Derived from retrieved docs (deterministic)
# =============================================================================

def determine_product_area(
    retrieved_docs: list[tuple[Document, float]],
    domain: Optional[str],
) -> str:
    """
    Determine product area from the top retrieved document's breadcrumbs.
    This is deterministic and traceable — the product area comes directly
    from the corpus metadata, not from the LLM.
    """
    if not retrieved_docs:
        if domain:
            return f"{domain}_general"
        return "general_support"

    top_doc = retrieved_docs[0][0]

    # Use breadcrumbs if available
    if top_doc.breadcrumbs:
        # Use the most specific breadcrumb (last one)
        area = top_doc.breadcrumbs[-1]
        return area.lower().strip().replace(" ", "_").replace("-", "_")

    # Fallback: derive from file path
    if top_doc.source_path:
        from pathlib import Path
        parts = Path(top_doc.source_path).parts
        # Find the domain directory and take the next level
        for i, part in enumerate(parts):
            if part.lower() in ("hackerrank", "claude", "visa"):
                if i + 1 < len(parts) and not parts[i + 1].endswith(".md"):
                    return parts[i + 1].lower().replace("-", "_").replace(" ", "_")
                break

    return f"{top_doc.domain}_general" if top_doc.domain else "general_support"


# =============================================================================
# STATUS DECISION — Rule-based hard gate
# =============================================================================

def decide_status(
    issue: str,
    subject: str,
    llm_classification: Optional[dict] = None,
) -> tuple[str, str, Optional[str]]:
    """
    Decide status deterministically using hard rules.

    Returns:
        (status, decision_method, rule_name_or_none)
        - decision_method: "rule" or "llm" (for traceability)
    """
    combined = f"{issue} {subject}".strip()

    # Hard gate: check all escalation rules
    for rule in ESCALATION_RULES:
        if re.search(rule.pattern, combined, re.IGNORECASE):
            return "escalated", "rule", rule.name

    # If no rule triggered, defer to LLM classification
    if llm_classification and llm_classification.get("should_escalate"):
        return "escalated", "llm", None

    return "replied", "llm", None


# =============================================================================
# REQUEST TYPE DECISION — Rule-based with LLM fallback
# =============================================================================

def classify_request_type(
    issue: str,
    subject: str,
    llm_classification: Optional[dict] = None,
) -> tuple[str, str]:
    """
    Classify request type deterministically where possible.

    Returns:
        (request_type, decision_method)
    """
    combined = f"{issue} {subject}".strip().lower()

    # Check rule-based patterns (order matters: invalid first, then bug, then feature)
    for rule in REQUEST_TYPE_RULES:
        if re.search(rule.pattern, combined, re.IGNORECASE):
            return rule.request_type, "rule"

    # Defer to LLM classification
    if llm_classification and llm_classification.get("request_type"):
        rt = llm_classification["request_type"].lower()
        valid = {"product_issue", "feature_request", "bug", "invalid"}
        if rt in valid:
            return rt, "llm"

    return "product_issue", "default"


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def format_retrieved_docs(results: list[tuple[Document, float]]) -> str:
    """Format retrieved documents for LLM context."""
    if not results:
        return "No relevant documentation found."

    parts = []
    for i, (doc, score) in enumerate(results, 1):
        breadcrumb_str = " > ".join(doc.breadcrumbs) if doc.breadcrumbs else doc.domain
        parts.append(
            f"--- Document {i} (relevance: {score:.3f}) ---\n"
            f"Title: {doc.title}\n"
            f"Category: {breadcrumb_str}\n"
            f"Domain: {doc.domain}\n"
            f"Content:\n{doc.content}\n"
        )
    return "\n".join(parts)


def process_ticket(
    issue: str,
    subject: str,
    company: str,
    retriever: Retriever,
    llm: LLMClient,
    top_k: int = 5,
) -> dict:
    """
    Process a single support ticket through the decomposed pipeline.

    Decision flow:
      1. Pleasantry check        → rule-based, no LLM
      2. Domain detection         → rule-based
      3. Document retrieval       → FAISS
      4. Product area             → from retrieved doc breadcrumbs (deterministic)
      5. Status (rules first)     → hard gate, check BEFORE any LLM call
      6. Request type (rules first) → check BEFORE any LLM call
      7. LLM classification       → ONLY if rules left gaps (conditional)
      8. Response generation      → LLM writes the user-facing text (always)
      9. Justification            → templated from rules + LLM reasoning

    LLM call budget:
      - Rule-decided tickets: 1 LLM call  (response only)
      - Ambiguous tickets:    2 LLM calls (classification + response)

    Returns:
        Dict with keys: status, product_area, response, justification, request_type
    """
    combined_text = f"{subject}\n{issue}".strip() if subject else issue.strip()

    # === Step 1: Pleasantry check (no LLM needed) ===
    issue_clean = issue.strip().lower()
    for pattern in PLEASANTRY_PATTERNS:
        if re.match(pattern, issue_clean, re.IGNORECASE):
            return {
                "status": "replied",
                "product_area": "general_support",
                "response": "You're welcome! Feel free to reach out if you need any further assistance.",
                "justification": "Rule: pleasantry detected — no actionable request.",
                "request_type": "invalid",
            }

    # === Step 2: Domain detection (rule-based) ===
    domain = determine_domain(company, issue, subject)

    # === Step 3: Retrieve relevant docs ===
    results = retriever.retrieve(combined_text, domain=domain, top_k=top_k)
    retrieved_context = format_retrieved_docs(results)

    # === Step 4: Product area (deterministic from retrieved docs) ===
    product_area = determine_product_area(results, domain)

    # === Step 5: Try rules FIRST for status ===
    status_by_rule, status_method, triggered_rule = decide_status(
        issue, subject, llm_classification=None  # No LLM yet
    )

    # === Step 6: Try rules FIRST for request type ===
    request_type_by_rule, type_method = classify_request_type(
        issue, subject, llm_classification=None  # No LLM yet
    )

    # === Step 7: LLM classification — ONLY if rules left gaps ===
    #   Rules left a gap when:
    #   - status_method is "llm" (no escalation rule fired, need LLM opinion)
    #   - type_method is "default" AND status wasn't already decided by rule
    #     (if status is rule-escalated, type defaults to "product_issue" — not
    #      worth an LLM call for secondary metadata on an already-escalated ticket)
    needs_llm_classification = status_method == "llm" or (type_method == "default" and status_method != "rule")
    llm_classification = None

    if needs_llm_classification:
        classification_prompt = f"""Analyze this support ticket:

TICKET:
- Issue: {issue}
- Subject: {subject}
- Company: {company}

RETRIEVED DOCUMENTATION:
{retrieved_context}

Respond with ONLY valid JSON."""

        llm_classification = llm.generate_json(CLASSIFICATION_PROMPT, classification_prompt)

        # Re-evaluate with LLM input for the fields that rules couldn't decide
        if status_method == "llm":
            status_by_rule, status_method, triggered_rule = decide_status(
                issue, subject, llm_classification
            )
        if type_method == "default":
            request_type_by_rule, type_method = classify_request_type(
                issue, subject, llm_classification
            )

    status = status_by_rule
    request_type = request_type_by_rule

    # Override: if status is escalated by rule, don't let request_type be "invalid"
    # unless it was classified as invalid by its own rule
    if status == "escalated" and status_method == "rule" and request_type == "invalid" and type_method != "rule":
        request_type = "product_issue"

    # === Step 8: Response generation (LLM's actual job — always runs) ===
    if status == "escalated":
        # For escalated tickets, generate a brief explanation
        response_prompt = f"""This support ticket is being escalated to a human agent.

TICKET:
- Issue: {issue}
- Subject: {subject}
- Company: {company}

Write a brief, empathetic response to the user explaining:
1. Their issue is being escalated to a specialist/human agent
2. Why it needs human attention (without revealing internal rules)
3. What they can expect next

Keep it concise (2-4 sentences). Be empathetic and professional."""

        response = llm.generate(RESPONSE_GENERATION_PROMPT, response_prompt, max_tokens=500)
    elif request_type == "invalid":
        # For invalid requests, generate a polite out-of-scope response
        response_prompt = f"""The user sent a message that is out of scope for support:

Message: {issue}

Write a brief, polite response explaining this is outside your support scope for HackerRank, Claude, and Visa products. Keep it to 1-2 sentences."""

        response = llm.generate(RESPONSE_GENERATION_PROMPT, response_prompt, max_tokens=300)
    else:
        # For replied tickets, generate a grounded answer
        response_prompt = f"""Answer this support ticket using ONLY the retrieved documentation below.

TICKET:
- Issue: {issue}
- Subject: {subject}
- Company: {company}

RETRIEVED DOCUMENTATION:
{retrieved_context}

Write a helpful response with specific steps from the documentation. If the docs don't fully cover the issue, say so honestly."""

        response = llm.generate(RESPONSE_GENERATION_PROMPT, response_prompt, max_tokens=1000)

    # === Step 9: Build justification (traceable) ===
    justification_parts = []

    # Document the decision chain
    if status_method == "rule":
        rule = next((r for r in ESCALATION_RULES if r.name == triggered_rule), None)
        if rule:
            justification_parts.append(f"Escalation rule '{rule.name}': {rule.reason}")
    elif status_method == "llm" and status == "escalated":
        llm_reason = (llm_classification or {}).get("escalation_reason", "LLM determined escalation needed")
        justification_parts.append(f"LLM escalation: {llm_reason}")

    justification_parts.append(f"Request type ({type_method}): {request_type}")
    justification_parts.append(f"Product area (from retrieved docs): {product_area}")

    if results:
        top_doc_title = results[0][0].title
        justification_parts.append(f"Top retrieved doc: '{top_doc_title}' (score: {results[0][1]:.3f})")

    # Add LLM's own brief justification if available
    if llm_classification:
        llm_justification = llm_classification.get("brief_justification", "")
        if llm_justification:
            justification_parts.append(f"LLM assessment: {llm_justification}")
    else:
        justification_parts.append("LLM classification: skipped (rules sufficient)")

    justification = " | ".join(justification_parts)

    return {
        "status": status,
        "product_area": product_area,
        "response": response.strip(),
        "justification": justification,
        "request_type": request_type,
    }
