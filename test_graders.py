"""
test_graders.py — Unit tests for all three deterministic graders.
"""
import pytest
from server.models import TriageAction
from server.graders import grade_classify, grade_route, grade_respond, grade


# ── Helper ground truth ───────────────────────────────────────────────────

GT_URGENT_BILLING = {
    "urgency": "urgent",
    "department": "billing",
    "response_keywords": ["apologize", "refund", "duplicate", "24", "account"],
}

GT_LOW_GENERAL = {
    "urgency": "low",
    "department": "general",
    "response_keywords": ["mobile", "app", "roadmap", "thank"],
}


# ── Task 1: Classify ──────────────────────────────────────────────────────

class TestGradeClassify:
    def test_correct_urgency_full_score(self):
        action = TriageAction(urgency="urgent")
        r = grade_classify(action, GT_URGENT_BILLING)
        assert r.total == 1.0
        assert r.urgency_score == 1.0

    def test_wrong_urgency_zero_score(self):
        action = TriageAction(urgency="low")
        r = grade_classify(action, GT_URGENT_BILLING)
        assert r.total == 0.0
        assert r.urgency_score == 0.0

    def test_unnecessary_department_penalized(self):
        action = TriageAction(urgency="urgent", department="billing")
        r = grade_classify(action, GT_URGENT_BILLING)
        assert r.total < 1.0
        assert r.penalties > 0.0

    def test_unnecessary_response_penalized(self):
        action = TriageAction(urgency="urgent", response="Dear customer...")
        r = grade_classify(action, GT_URGENT_BILLING)
        assert r.total < 1.0

    def test_score_in_range(self):
        for urgency in ["urgent", "normal", "low"]:
            action = TriageAction(urgency=urgency)
            r = grade_classify(action, GT_URGENT_BILLING)
            assert 0.0 <= r.total <= 1.0

    def test_dispatcher(self):
        action = TriageAction(urgency="urgent")
        r = grade("task_classify", action, GT_URGENT_BILLING)
        assert r.total == 1.0


# ── Task 2: Route ─────────────────────────────────────────────────────────

class TestGradeRoute:
    def test_both_correct_full_score(self):
        action = TriageAction(urgency="urgent", department="billing")
        r = grade_route(action, GT_URGENT_BILLING)
        assert r.total == 1.0
        assert r.urgency_score == 0.5
        assert r.routing_score == 0.5

    def test_only_urgency_correct(self):
        action = TriageAction(urgency="urgent", department="technical")
        r = grade_route(action, GT_URGENT_BILLING)
        assert r.urgency_score == 0.5
        assert r.routing_score == 0.0
        assert abs(r.total - 0.5) < 1e-6

    def test_only_department_correct(self):
        action = TriageAction(urgency="low", department="billing")
        r = grade_route(action, GT_URGENT_BILLING)
        assert r.urgency_score == 0.0
        assert r.routing_score == 0.5
        assert abs(r.total - 0.5) < 1e-6

    def test_both_wrong(self):
        action = TriageAction(urgency="low", department="general")
        r = grade_route(action, GT_URGENT_BILLING)
        assert r.total == 0.0

    def test_unnecessary_response_penalized(self):
        action = TriageAction(urgency="urgent", department="billing", response="Hi")
        r = grade_route(action, GT_URGENT_BILLING)
        assert r.total < 1.0

    def test_score_in_range(self):
        action = TriageAction(urgency="normal", department="returns")
        r = grade_route(action, GT_URGENT_BILLING)
        assert 0.0 <= r.total <= 1.0


# ── Task 3: Respond ───────────────────────────────────────────────────────

class TestGradeRespond:
    def _good_response(self):
        return (
            "Dear John,\n\nI sincerely apologize for the duplicate charge on your account. "
            "Our billing team will process a full refund within 24 hours. "
            "Please be assured your account is our priority and this will be resolved immediately.\n\n"
            "Best regards,\nSupport Team"
        )

    def test_full_correct_action_high_score(self):
        action = TriageAction(
            urgency="urgent",
            department="billing",
            response=self._good_response(),
        )
        r = grade_respond(action, GT_URGENT_BILLING)
        assert r.total > 0.7
        assert r.urgency_score == 0.25
        assert r.routing_score == 0.25
        assert r.response_score > 0.3

    def test_no_response_penalty(self):
        action = TriageAction(urgency="urgent", department="billing")
        r = grade_respond(action, GT_URGENT_BILLING)
        assert r.response_score == 0.0
        assert r.penalties > 0.0

    def test_empty_response_penalty(self):
        action = TriageAction(urgency="urgent", department="billing", response="")
        r = grade_respond(action, GT_URGENT_BILLING)
        assert r.response_score == 0.0

    def test_short_response_no_length_bonus(self):
        action = TriageAction(urgency="urgent", department="billing", response="Sorry about that.")
        r = grade_respond(action, GT_URGENT_BILLING)
        assert r.response_score >= 0.10  # at least "has response" credit
        assert r.response_score < 0.25   # no length bonus

    def test_placeholder_penalty(self):
        action = TriageAction(
            urgency="urgent",
            department="billing",
            response="Dear [NAME], your [ISSUE] will be resolved by [DATE]. Best regards",
        )
        r = grade_respond(action, GT_URGENT_BILLING)
        assert r.response_score < 0.35  # placeholder detection reduces score

    def test_professional_closing_rewarded(self):
        resp_with_closing = self._good_response()
        resp_no_closing = "Dear John, your refund will be processed."
        action_with = TriageAction(urgency="urgent", department="billing", response=resp_with_closing)
        action_without = TriageAction(urgency="urgent", department="billing", response=resp_no_closing)
        r_with = grade_respond(action_with, GT_URGENT_BILLING)
        r_without = grade_respond(action_without, GT_URGENT_BILLING)
        assert r_with.response_score > r_without.response_score

    def test_wrong_urgency_dept_caps_score(self):
        action = TriageAction(
            urgency="low",
            department="general",
            response=self._good_response(),
        )
        r = grade_respond(action, GT_URGENT_BILLING)
        # urgency and dept both wrong = 0.0 + 0.0 = 0, only response can score
        assert r.urgency_score == 0.0
        assert r.routing_score == 0.0
        assert r.total <= 0.55  # capped by missing urgency+dept

    def test_total_always_0_to_1(self):
        for urgency in ["urgent", "normal", "low"]:
            for dept in ["billing", "technical", "returns", "general"]:
                action = TriageAction(urgency=urgency, department=dept, response="A test response. Best regards.")
                r = grade_respond(action, GT_URGENT_BILLING)
                assert 0.0 <= r.total <= 1.0, f"Out of range for {urgency}/{dept}: {r.total}"


# ── Dispatcher ────────────────────────────────────────────────────────────

def test_dispatcher_invalid_task():
    action = TriageAction(urgency="urgent")
    with pytest.raises(ValueError, match="Unknown task_id"):
        grade("task_nonexistent", action, GT_URGENT_BILLING)
