"""
graders.py - Deterministic graders for the three Email Triage tasks.

Each grader:
  - Takes a TriageAction and the email's ground_truth dict
  - Returns a TriageReward with partial credit breakdown
  - Is fully deterministic (no LLM calls, no randomness)
"""

import re
from .models import TriageAction, TriageReward


# -----------------------------------------------------------------------------
# Task 1 - Easy: Classify urgency only
# -----------------------------------------------------------------------------

def grade_classify(action: TriageAction, ground_truth: dict) -> TriageReward:
    """
    Score: 1.0 if urgency matches, 0.0 otherwise.
    Penalty: -0.1 if department or response is provided (unnecessary fields).
    """
    correct = action.urgency == ground_truth["urgency"]
    urgency_score = 1.0 if correct else 0.0

    # Mild penalty for providing unrequested fields (wastes tokens in real usage)
    penalties = 0.0
    penalty_reasons = []
    if action.department is not None:
        penalties += 0.05
        penalty_reasons.append("unnecessary 'department' field provided")
    if action.response is not None:
        penalties += 0.05
        penalty_reasons.append("unnecessary 'response' field provided")

    total = max(0.0, urgency_score - penalties)

    if correct:
        detail = f"OK Urgency correctly classified as '{action.urgency}'."
    else:
        detail = (
            f"X Urgency misclassified: got '{action.urgency}', "
            f"expected '{ground_truth['urgency']}'."
        )
    if penalty_reasons:
        detail += f" Penalties: {', '.join(penalty_reasons)}."

    return TriageReward(
        total=round(total, 4),
        urgency_score=round(urgency_score, 4),
        routing_score=0.0,
        response_score=0.0,
        penalties=round(penalties, 4),
        details=detail,
    )


# -----------------------------------------------------------------------------
# Task 2 - Medium: Classify urgency + route to department
# -----------------------------------------------------------------------------

def grade_route(action: TriageAction, ground_truth: dict) -> TriageReward:
    """
    Score breakdown:
      - Urgency correct  -> 0.5 points
      - Department correct -> 0.5 points
    Penalty: -0.1 if response provided (not required for this task).
    """
    urgency_correct = action.urgency == ground_truth["urgency"]
    dept_correct = action.department == ground_truth["department"]

    urgency_score = 0.5 if urgency_correct else 0.0
    routing_score = 0.5 if dept_correct else 0.0

    penalties = 0.0
    penalty_reasons = []
    if action.response is not None:
        penalties += 0.1
        penalty_reasons.append("unnecessary 'response' field provided")

    total = max(0.0, urgency_score + routing_score - penalties)

    parts = []
    if urgency_correct:
        parts.append(f"OK Urgency '{action.urgency}' correct (+0.5)")
    else:
        parts.append(
            f"X Urgency got '{action.urgency}', expected '{ground_truth['urgency']}' (+0.0)"
        )
    if dept_correct:
        parts.append(f"OK Department '{action.department}' correct (+0.5)")
    else:
        parts.append(
            f"X Department got '{action.department}', "
            f"expected '{ground_truth['department']}' (+0.0)"
        )
    if penalty_reasons:
        parts.append(f"Penalties: {', '.join(penalty_reasons)}")

    return TriageReward(
        total=round(total, 4),
        urgency_score=round(urgency_score, 4),
        routing_score=round(routing_score, 4),
        response_score=0.0,
        penalties=round(penalties, 4),
        details=" | ".join(parts),
    )


# -----------------------------------------------------------------------------
# Task 3 - Hard: Classify + Route + Draft Response
# -----------------------------------------------------------------------------

def grade_respond(action: TriageAction, ground_truth: dict) -> TriageReward:
    """
    Score breakdown (total = 1.0):
      - Urgency correct    -> 0.25 points
      - Department correct -> 0.25 points
      - Response quality   -> up to 0.50 points (keyword matching + length check)

    Response scoring sub-criteria (0.0-0.50):
      - Has any response at all:           +0.10
      - Length >= 50 words:                +0.05
      - Length >= 100 words:               +0.05
      - Contains >=1 required keywords:    +0.10
      - Contains >=3 required keywords:    +0.10
      - Contains ALL required keywords:    +0.10
      - Ends with a sign-off/closing:      +0.05  (Dear/Hi/Regards/Sincerely/Best)
      - Does NOT use placeholder brackets: +0.05  (no [NAME], [DATE] etc.)
    """
    urgency_correct = action.urgency == ground_truth["urgency"]
    dept_correct = action.department == ground_truth["department"]
    required_keywords = [kw.lower() for kw in ground_truth.get("response_keywords", [])]

    urgency_score = 0.25 if urgency_correct else 0.0
    routing_score = 0.25 if dept_correct else 0.0

    # Response quality scoring
    response_score = 0.0
    response_details = []

    if action.response is None or action.response.strip() == "":
        response_details.append("X No response provided (response is required for task_respond)")
        # Penalty for missing response
        penalties = 0.20
    else:
        penalties = 0.0
        resp_lower = action.response.lower()
        word_count = len(action.response.split())

        # Has a response at all
        response_score += 0.10
        response_details.append("OK Response provided (+0.10)")

        # Length bonuses
        if word_count >= 50:
            response_score += 0.05
            response_details.append(f"OK Response >=50 words ({word_count} words) (+0.05)")
        else:
            response_details.append(f"X Response <50 words ({word_count} words) (+0.00)")

        if word_count >= 100:
            response_score += 0.05
            response_details.append("OK Response >=100 words (+0.05)")
        else:
            response_details.append("X Response <100 words (+0.00)")

        # Keyword matching
        found_keywords = [kw for kw in required_keywords if kw in resp_lower]
        n_found = len(found_keywords)

        if n_found >= 1:
            response_score += 0.10
            response_details.append(f"OK >=1 keyword found: {found_keywords[:1]} (+0.10)")
        if n_found >= 3:
            response_score += 0.10
            response_details.append(f"OK >=3 keywords found: {found_keywords[:3]} (+0.10)")
        if n_found == len(required_keywords) and len(required_keywords) > 0:
            response_score += 0.10
            response_details.append(f"OK All {len(required_keywords)} keywords found (+0.10)")

        # Professional sign-off check
        closing_patterns = [
            r"\bbest regards\b", r"\bsincerely\b", r"\bkind regards\b",
            r"\bthanks\b", r"\bthank you\b", r"\bwarm regards\b",
        ]
        has_closing = any(re.search(p, resp_lower) for p in closing_patterns)
        if has_closing:
            response_score += 0.05
            response_details.append("OK Professional closing detected (+0.05)")
        else:
            response_details.append("X No professional closing found (+0.00)")

        # No placeholder brackets
        has_placeholders = bool(re.search(r"\[.*?\]", action.response))
        if not has_placeholders:
            response_score += 0.05
            response_details.append("OK No unfilled placeholders (+0.05)")
        else:
            response_details.append("X Unfilled placeholders like [NAME] detected (+0.00)")

    total = max(0.0, urgency_score + routing_score + response_score - penalties)

    summary_parts = []
    if urgency_correct:
        summary_parts.append(f"OK Urgency '{action.urgency}' (+0.25)")
    else:
        summary_parts.append(
            f"X Urgency got '{action.urgency}', expected '{ground_truth['urgency']}' (+0.00)"
        )
    if dept_correct:
        summary_parts.append(f"OK Department '{action.department}' (+0.25)")
    else:
        summary_parts.append(
            f"X Department got '{action.department}', "
            f"expected '{ground_truth['department']}' (+0.00)"
        )
    summary_parts.append("Response: " + " | ".join(response_details))

    return TriageReward(
        total=round(min(1.0, total), 4),
        urgency_score=round(urgency_score, 4),
        routing_score=round(routing_score, 4),
        response_score=round(response_score, 4),
        penalties=round(penalties, 4),
        details=" || ".join(summary_parts),
    )


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------

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
    """Dispatch to the correct grader for the given task_id."""
    if task_id == "task_classify":
        return grade_classify(action, ground_truth)
    elif task_id == "task_route":
        return grade_route(action, ground_truth)
    elif task_id == "task_respond":
        return grade_respond(action, ground_truth)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
