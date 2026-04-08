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
    u_correct = action.urgency   == ground_truth["urgency"]
    d_correct = action.department == ground_truth["department"]
    required  = [k.lower() for k in ground_truth.get("response_keywords", [])]

    urgency_score = RIGHT if u_correct else WRONG
    routing_score = RIGHT if d_correct else WRONG

    # Response quality: starts at 0.10, earns up to 0.80
    if action.response is None or action.response.strip() == "":
        response_score = 0.10
        penalties = 0.15
    else:
        penalties = 0.0
        resp = action.response.lower()
        words = len(action.response.split())
        score = 0.20  # base for having a response
        if words >= 30:  score += 0.10
        if words >= 60:  score += 0.10
        if words >= 100: score += 0.10
        found = [k for k in required if k in resp]
        if len(found) >= 1: score += 0.10
        if len(found) >= 3: score += 0.10
        if len(found) == len(required) and required: score += 0.10
        closing = [r"\bbest regards\b", r"\bsincerely\b", r"\bthanks\b",
                   r"\bthank you\b", r"\bkind regards\b"]
        if any(re.search(p, resp) for p in closing): score += 0.05
        if not re.search(r"\[.*?\]", action.response):  score += 0.05
        response_score = min(0.85, score)

    # Weighted total: urgency 25%, routing 25%, response 50%
    raw = (urgency_score * 0.25) + (routing_score * 0.25) + (response_score * 0.50) - penalties
    total = max(WRONG, raw)

    return TriageReward(
        total=clamp_score(total),
        urgency_score=clamp_score(urgency_score),
        routing_score=clamp_score(routing_score),
        response_score=clamp_score(response_score),
        penalties=round(max(0.01, penalties) if penalties > 0 else 0.01, 4),
        details=(
            f"Urgency {'OK' if u_correct else 'WRONG'} | "
            f"Dept {'OK' if d_correct else 'WRONG'} | "
            f"Response score={round(response_score,3)}"
        ),
    )


TASK_DESCRIPTIONS = {
    "task_classify": (
        "Classify the urgency level of this support email. "
        "Set 'urgency' to one of: urgent | normal | low. "
        "Do NOT include department or response fields."
    ),
    "task_route": (
        "Classify the urgency level AND route to the correct department. "
        "Set 'urgency' (urgent|normal|low) and 'department' (billing|technical|returns|general). "
        "Do NOT include a response field."
    ),
    "task_respond": (
        "Perform a full triage: classify urgency, route to department, AND draft a professional reply. "
        "Set 'urgency', 'department', and 'response' (the draft reply to the customer). "
        "The response should be professional, address the customer's issue, and include a closing."
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
