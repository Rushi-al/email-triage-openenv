"""
graders.py - Deterministic graders for Email Triage tasks.
All scores are NATURALLY in (0.0, 1.0) exclusive — never exactly 0 or 1.
Scoring base: wrong=0.10, correct=0.90, partial credit between.
"""

import re
from .models import TriageAction, TriageReward

WRONG  = 0.10   # never 0.0
RIGHT  = 0.90   # never 1.0


def clamp_score(score: float) -> float:
    """Ensure score is strictly within (0, 1) and survives coarse rounding."""
    return max(0.05, min(0.95, score))


def grade_classify(action: TriageAction, ground_truth: dict) -> TriageReward:
    correct = action.urgency == ground_truth["urgency"]
    urgency_score = RIGHT if correct else WRONG
    penalties = 0.0
    if action.department is not None:
        penalties += 0.05
    if action.response is not None:
        penalties += 0.05
    total = max(WRONG, urgency_score - penalties)
    detail = (
        f"Urgency '{action.urgency}' {'correct' if correct else 'wrong'} "
        f"(expected '{ground_truth['urgency']}')"
    )
    return TriageReward(
        total=clamp_score(total),
        urgency_score=clamp_score(urgency_score),
        routing_score=clamp_score(0.10),
        response_score=clamp_score(0.10),
        penalties=round(max(0.01, penalties) if penalties > 0 else 0.01, 4),
        details=detail,
    )


def grade_route(action: TriageAction, ground_truth: dict) -> TriageReward:
    u_correct = action.urgency   == ground_truth["urgency"]
    d_correct = action.department == ground_truth["department"]
    urgency_score  = RIGHT if u_correct else WRONG
    routing_score  = RIGHT if d_correct else WRONG
    penalties = 0.05 if action.response is not None else 0.0
    raw_total = (urgency_score * 0.5) + (routing_score * 0.5) - penalties
    total = max(WRONG, raw_total)
    return TriageReward(
        total=clamp_score(total),
        urgency_score=clamp_score(urgency_score),
        routing_score=clamp_score(routing_score),
        response_score=clamp_score(0.10),
        penalties=round(max(0.01, penalties) if penalties > 0 else 0.01, 4),
        details=(
            f"Urgency {'OK' if u_correct else 'WRONG'} | "
            f"Dept {'OK' if d_correct else 'WRONG'}"
        ),
    )


def grade_respond(action: TriageAction, ground_truth: dict) -> TriageReward:
    u_correct = action.urgency    == ground_truth["urgency"]
    d_correct = action.department == ground_truth["department"]
    required  = [k.lower() for k in ground_truth.get("response_keywords", [])]

    urgency_score = RIGHT if u_correct else WRONG
    routing_score = RIGHT if d_correct else WRONG

    # ── Response quality scoring ──────────────────────────────────────
    if action.response is None or action.response.strip() == "":
        response_score = 0.10
        penalties = 0.15
    else:
        penalties = 0.0
        resp  = action.response.lower()
        words = len(action.response.split())

        score = 0.20  # base for having a response

        # Length bonuses
        if words >= 30:  score += 0.08
        if words >= 60:  score += 0.07
        if words >= 100: score += 0.05

        # Keyword matching
        found = [k for k in required if k in resp]
        if len(found) >= 1: score += 0.08
        if len(found) >= 3: score += 0.08
        if len(found) == len(required) and required: score += 0.09

        # Professional closing
        closing = [r"\bbest regards\b", r"\bsincerely\b", r"\bthanks\b",
                   r"\bthank you\b", r"\bkind regards\b"]
        if any(re.search(p, resp) for p in closing):
            score += 0.05

        # No unfilled placeholders
        if not re.search(r"\[.*?\]", action.response):
            score += 0.05

        # ── NEW: Penalize generic/template responses ──────────────────
        generic_phrases = [
            "dear customer",
            "dear valued customer",
            "thank you for contacting us",
            "we will look into this",
            "please don't hesitate to contact us",
            "we appreciate your patience",
        ]
        generic_count = sum(1 for p in generic_phrases if p in resp)
        if generic_count >= 3:
            score -= 0.10

        # ── NEW: Reward personalization ───────────────────────────────
        # Response references specific order/account/amount details
        has_personal = bool(re.search(
            r'\b(order\s*#?\w+|account\s*#?\w+|\$[\d,]+|#\w+[-]\w+)\b',
            action.response, re.IGNORECASE
        ))
        if has_personal:
            score += 0.08

        # ── NEW: Urgency-matched tone for urgent emails ───────────────
        if ground_truth["urgency"] == "urgent":
            urgent_words = ["immediately", "right away", "priority",
                            "escalat", "urgent", "now", "asap"]
            if any(w in resp for w in urgent_words):
                score += 0.05

        # ── NEW: Low urgency should NOT use urgent language ───────────
        if ground_truth["urgency"] == "low":
            urgent_words = ["immediately", "escalate", "critical", "emergency"]
            if any(w in resp for w in urgent_words):
                score -= 0.05  # penalize mismatched tone

        response_score = min(0.88, max(0.10, score))

    # Weighted total: urgency 25%, routing 25%, response 50%
    raw   = (urgency_score * 0.25) + (routing_score * 0.25) + (response_score * 0.50) - penalties
    total = max(WRONG, raw)

    return TriageReward(
        total=clamp_score(total),
        urgency_score=clamp_score(urgency_score),
        routing_score=clamp_score(routing_score),
        response_score=clamp_score(response_score),
        penalties=round(max(0.01, penalties), 4),
        details=(
            f"Urgency {'OK' if u_correct else 'WRONG'} | "
            f"Dept {'OK' if d_correct else 'WRONG'} | "
            f"Response score={round(response_score, 3)} | "
            f"Keywords found={len(found) if action.response else 0}/{len(required)}"
        ),
    )


TASK_DESCRIPTIONS = {
    "task_classify": (
        "Classify the urgency level of this support email. "
        "Set 'urgency' to one of: urgent | normal | low. "
        "urgent = time-critical, financial loss, production down, very distressed customer. "
        "normal = standard issue needing timely response. "
        "low = informational, feature requests, general questions, compliments. "
        "Do NOT include department or response fields."
    ),
    "task_route": (
        "Classify the urgency level AND route to the correct department. "
        "Set 'urgency' (urgent|normal|low) and 'department' (billing|technical|returns|general). "
        "billing = charges, invoices, subscriptions, refunds, fraud, account suspension. "
        "technical = API errors, bugs, login issues, integrations, webhooks, security. "
        "returns = wrong items, damaged goods, missing items, replacements, defects. "
        "general = product questions, feature requests, feedback, data export, compliments. "
        "Do NOT include a response field."
    ),
    "task_respond": (
        "Perform a full triage: classify urgency, route to department, AND draft a professional reply. "
        "Set 'urgency', 'department', and 'response' (your draft reply to the customer). "
        "Your response MUST: "
        "1. Reference specific details from the email (order number, account ID, amount, name). "
        "2. Match the urgency tone — urgent emails need words like 'immediately', 'priority', 'now'. "
        "3. Include domain-specific keywords relevant to the issue type. "
        "4. Be at least 60 words long. "
        "5. End with a professional closing (Best regards, Sincerely, etc). "
        "6. Avoid generic filler phrases like 'dear valued customer' or 'please dont hesitate'. "
        "Grading penalizes generic templates and rewards specific, personalized responses."
    ),
}


def grade(task_id: str, action: TriageAction, ground_truth: dict) -> TriageReward:
    if task_id == "task_classify":
        r = grade_classify(action, ground_truth)
    elif task_id == "task_route":
        r = grade_route(action, ground_truth)
    elif task_id == "task_respond":
        r = grade_respond(action, ground_truth)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
    # Final safety clamp — always strictly (0,1)
    r.total          = clamp_score(r.total)
    r.urgency_score  = clamp_score(r.urgency_score)
    r.routing_score  = clamp_score(r.routing_score)
    r.response_score = clamp_score(r.response_score)
    return r
