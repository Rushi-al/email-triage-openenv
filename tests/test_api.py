"""
test_api.py — Integration tests for all FastAPI endpoints.
Uses FastAPI's TestClient (no real server needed).
"""
import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)


class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_key(self):
        r = client.get("/health")
        assert r.json()["status"] == "healthy"


class TestReset:
    def test_reset_default_task(self):
        r = client.post("/reset", json={})
        assert r.status_code == 200
        data = r.json()
        assert "email_id" in data
        assert "subject" in data
        assert "body" in data
        assert "task_id" in data
        assert data["task_id"] == "task_classify"

    def test_reset_all_tasks(self):
        for task_id in ["task_classify", "task_route", "task_respond"]:
            r = client.post("/reset", json={"task_id": task_id, "seed": 42})
            assert r.status_code == 200
            assert r.json()["task_id"] == task_id

    def test_reset_invalid_task_400(self):
        r = client.post("/reset", json={"task_id": "task_fake"})
        assert r.status_code == 400

    def test_reset_with_seed_reproducible(self):
        r1 = client.post("/reset", json={"task_id": "task_classify", "seed": 42})
        r2 = client.post("/reset", json={"task_id": "task_classify", "seed": 42})
        assert r1.json()["email_id"] == r2.json()["email_id"]

    def test_reset_done_is_false(self):
        r = client.post("/reset", json={})
        assert r.json()["done"] is False

    def test_reset_score_is_zero(self):
        r = client.post("/reset", json={})
        assert r.json()["score"] == 0.0


class TestStep:
    def setup_method(self):
        client.post("/reset", json={"task_id": "task_classify", "seed": 42})

    def test_step_valid_action(self):
        r = client.post("/step", json={"urgency": "urgent"})
        assert r.status_code == 200

    def test_step_response_structure(self):
        r = client.post("/step", json={"urgency": "urgent"})
        data = r.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_in_range(self):
        for urgency in ["urgent", "normal", "low"]:
            client.post("/reset", json={"task_id": "task_classify", "seed": 42})
            r = client.post("/step", json={"urgency": urgency})
            reward = r.json()["reward"]
            assert 0.0 <= reward <= 1.0, f"reward {reward} out of range for urgency={urgency}"

    def test_step_invalid_urgency_422(self):
        r = client.post("/step", json={"urgency": "super-urgent"})
        assert r.status_code == 422

    def test_step_missing_urgency_422(self):
        r = client.post("/step", json={})
        assert r.status_code == 422

    def test_step_fills_10_emails(self):
        client.post("/reset", json={"task_id": "task_classify", "seed": 42})
        done = False
        count = 0
        while not done:
            r = client.post("/step", json={"urgency": "normal"})
            done = r.json()["done"]
            count += 1
        assert count == 10

    def test_step_after_done_409(self):
        client.post("/reset", json={"task_id": "task_classify", "seed": 42})
        for _ in range(10):
            client.post("/step", json={"urgency": "normal"})
        r = client.post("/step", json={"urgency": "normal"})
        assert r.status_code == 409

    def test_step_task_route_with_department(self):
        client.post("/reset", json={"task_id": "task_route", "seed": 42})
        r = client.post("/step", json={"urgency": "urgent", "department": "billing"})
        assert r.status_code == 200

    def test_step_task_respond_with_response(self):
        client.post("/reset", json={"task_id": "task_respond", "seed": 42})
        r = client.post("/step", json={
            "urgency": "urgent",
            "department": "billing",
            "response": "Dear customer, we apologize. Best regards, Team"
        })
        assert r.status_code == 200
        assert r.json()["reward"] > 0.0


class TestState:
    def test_state_returns_200(self):
        client.post("/reset", json={"task_id": "task_classify", "seed": 42})
        r = client.get("/state")
        assert r.status_code == 200

    def test_state_has_required_fields(self):
        client.post("/reset", json={"task_id": "task_classify", "seed": 42})
        r = client.get("/state")
        data = r.json()
        assert "episode_id" in data
        assert "task_id" in data
        assert "step_count" in data
        assert "done" in data

    def test_state_step_count_increments(self):
        client.post("/reset", json={"task_id": "task_classify", "seed": 42})
        r0 = client.get("/state")
        client.post("/step", json={"urgency": "normal"})
        r1 = client.get("/state")
        assert r1.json()["step_count"] == r0.json()["step_count"] + 1


class TestTasks:
    def test_tasks_returns_200(self):
        r = client.get("/tasks")
        assert r.status_code == 200

    def test_tasks_has_three_tasks(self):
        r = client.get("/tasks")
        assert len(r.json()["tasks"]) == 3

    def test_tasks_difficulty_progression(self):
        tasks = client.get("/tasks").json()["tasks"]
        difficulties = [t["difficulty"] for t in tasks]
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_tasks_have_action_schema(self):
        tasks = client.get("/tasks").json()["tasks"]
        for t in tasks:
            assert "action_schema" in t
            assert "urgency" in t["action_schema"]

    def test_tasks_valid_urgency_values(self):
        r = client.get("/tasks")
        valid = r.json()["valid_urgency_values"]
        assert set(valid) == {"urgent", "normal", "low"}

    def test_tasks_valid_department_values(self):
        r = client.get("/tasks")
        valid = r.json()["valid_department_values"]
        assert set(valid) == {"billing", "technical", "returns", "general"}


class TestGrader:
    def test_grader_correct_action(self):
        r = client.post("/grader", json={
            "task_id": "task_classify",
            "email_id": "E001",
            "urgency": "urgent",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["reward"]["total"] == 1.0

    def test_grader_wrong_action(self):
        r = client.post("/grader", json={
            "task_id": "task_classify",
            "email_id": "E001",
            "urgency": "low",
        })
        assert r.status_code == 200
        assert r.json()["reward"]["total"] == 0.0

    def test_grader_invalid_task_400(self):
        r = client.post("/grader", json={
            "task_id": "task_fake",
            "email_id": "E001",
            "urgency": "urgent",
        })
        assert r.status_code == 400

    def test_grader_invalid_email_404(self):
        r = client.post("/grader", json={
            "task_id": "task_classify",
            "email_id": "E999",
            "urgency": "urgent",
        })
        assert r.status_code == 404

    def test_grader_all_emails_all_tasks(self):
        email_ids = [f"E{str(i).zfill(3)}" for i in range(1, 11)]
        for eid in email_ids:
            for task_id in ["task_classify", "task_route", "task_respond"]:
                r = client.post("/grader", json={
                    "task_id": task_id,
                    "email_id": eid,
                    "urgency": "normal",
                    "department": "general",
                    "response": "Dear customer, thank you. Best regards.",
                })
                assert r.status_code == 200
                reward = r.json()["reward"]["total"]
                assert 0.0 <= reward <= 1.0

    def test_grader_route_task_partial_credit(self):
        # E001 is urgent/billing — correct urgency, wrong dept
        r = client.post("/grader", json={
            "task_id": "task_route",
            "email_id": "E001",
            "urgency": "urgent",
            "department": "general",
        })
        data = r.json()
        assert data["reward"]["urgency_score"] == 0.5
        assert data["reward"]["routing_score"] == 0.0
        assert abs(data["reward"]["total"] - 0.5) < 1e-6


class TestBaseline:
    def test_baseline_returns_200(self):
        r = client.get("/baseline")
        assert r.status_code == 200

    def test_baseline_has_scores_for_all_tasks(self):
        r = client.get("/baseline")
        scores = r.json()["scores"]
        assert "task_classify" in scores
        assert "task_route" in scores
        assert "task_respond" in scores

    def test_baseline_scores_in_range(self):
        r = client.get("/baseline")
        scores = r.json()["scores"]
        for task_id, result in scores.items():
            assert 0.0 <= result["avg_score"] <= 1.0, \
                f"{task_id} avg_score {result['avg_score']} out of range"

    def test_baseline_classify_higher_than_respond(self):
        r = client.get("/baseline")
        scores = r.json()["scores"]
        # Easy task should score at least as well as hard task for baseline
        assert scores["task_classify"]["avg_score"] >= scores["task_respond"]["avg_score"]
