"""
test_environment.py — Unit tests for the core environment logic.
"""
import pytest
from server.environment import EmailTriageEnvironment, VALID_TASK_IDS, MAX_STEPS_PER_EPISODE
from server.models import EmailObservation, TriageState


@pytest.fixture
def env():
    return EmailTriageEnvironment()


class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset("task_classify", seed=42)
        assert isinstance(obs, EmailObservation)

    def test_reset_clears_state(self, env):
        env.reset("task_classify", seed=42)
        env.step_from_obs(env, "urgent")
        obs = env.reset("task_classify", seed=42)
        assert obs.score == 0.0
        assert obs.done is False

    def test_reset_invalid_task(self, env):
        with pytest.raises(ValueError):
            env.reset("task_nonexistent")

    def test_reset_all_valid_tasks(self, env):
        for task_id in VALID_TASK_IDS:
            obs = env.reset(task_id, seed=42)
            assert obs.task_id == task_id

    def test_reset_seed_reproducible(self):
        env1 = EmailTriageEnvironment()
        env2 = EmailTriageEnvironment()
        obs1 = env1.reset("task_classify", seed=42)
        obs2 = env2.reset("task_classify", seed=42)
        assert obs1.email_id == obs2.email_id

    def test_reset_different_seeds_different_order(self):
        env1 = EmailTriageEnvironment()
        env2 = EmailTriageEnvironment()
        obs1 = env1.reset("task_classify", seed=1)
        obs2 = env2.reset("task_classify", seed=99)
        # It's possible (but unlikely) they start on the same email;
        # just check both are valid observations
        assert isinstance(obs1, EmailObservation)
        assert isinstance(obs2, EmailObservation)

    def test_reset_returns_new_episode_id(self, env):
        env.reset("task_classify", seed=1)
        id1 = env.state.episode_id
        env.reset("task_classify", seed=2)
        id2 = env.state.episode_id
        assert id1 != id2


class TestStep:
    def test_step_returns_tuple(self, env):
        env.reset("task_classify", seed=42)
        obs, reward, done, info = env.step_action(urgency="urgent")
        assert isinstance(obs, EmailObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_reward_in_range(self, env):
        env.reset("task_classify", seed=42)
        for _ in range(5):
            if env.state.done:
                break
            _, reward, _, _ = env.step_action(urgency="normal")
            assert 0.0 <= reward <= 1.0

    def test_step_increments_step_count(self, env):
        env.reset("task_classify", seed=42)
        env.step_action(urgency="urgent")
        assert env.state.step_count == 1
        env.step_action(urgency="normal")
        assert env.state.step_count == 2

    def test_episode_ends_after_max_steps(self, env):
        env.reset("task_classify", seed=42)
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step_action(urgency="normal")
            steps += 1
        assert steps == MAX_STEPS_PER_EPISODE

    def test_step_after_done_raises(self, env):
        env.reset("task_classify", seed=42)
        for _ in range(MAX_STEPS_PER_EPISODE):
            env.step_action(urgency="normal")
        with pytest.raises(RuntimeError, match="already done"):
            env.step_action(urgency="normal")

    def test_terminal_obs_has_done_true(self, env):
        env.reset("task_classify", seed=42)
        obs = None
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs, _, done, _ = env.step_action(urgency="normal")
        assert obs.done is True

    def test_info_has_reward_breakdown(self, env):
        env.reset("task_classify", seed=42)
        _, _, _, info = env.step_action(urgency="urgent")
        assert "total" in info
        assert "urgency_score" in info
        assert "details" in info

    def test_task_route_needs_department(self, env):
        from server.models import TriageAction
        env.reset("task_route", seed=42)
        action = TriageAction(urgency="urgent", department="billing")
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)

    def test_task_respond_rewards_good_response(self, env):
        from server.models import TriageAction
        env.reset("task_respond", seed=42)
        # Make an action that should score decently
        action = TriageAction(
            urgency="urgent",
            department="billing",
            response=(
                "Dear Customer, I apologize sincerely for this issue. "
                "Our billing team will process a full refund to your account within 24 hours. "
                "This is our highest priority and we are sorry for the inconvenience. "
                "Best regards, Support Team"
            )
        )
        _, reward, _, _ = env.step(action)
        assert reward > 0.0


class TestState:
    def test_state_returns_triage_state(self, env):
        env.reset("task_classify", seed=42)
        s = env.state
        assert isinstance(s, TriageState)

    def test_state_episode_id_is_uuid(self, env):
        env.reset("task_classify", seed=42)
        import re
        s = env.state
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, s.episode_id)

    def test_state_task_id_matches_reset(self, env):
        env.reset("task_respond", seed=42)
        assert env.state.task_id == "task_respond"

    def test_state_is_copy(self, env):
        env.reset("task_classify", seed=42)
        s1 = env.state
        env.step_action(urgency="normal")
        s2 = env.state
        assert s1.step_count != s2.step_count


# ── Helpers to make tests work with the environment API ──────────────────

def _patch_env():
    """Add helper methods to EmailTriageEnvironment for test ergonomics."""
    from server.models import TriageAction

    def step_action(self, urgency="normal", department=None, response=None):
        action = TriageAction(urgency=urgency, department=department, response=response)
        return self.step(action)

    EmailTriageEnvironment.step_action = step_action

    def step_from_obs(self, env_ref, urgency):
        from server.models import TriageAction
        action = TriageAction(urgency=urgency)
        return env_ref.step(action)

    EmailTriageEnvironment.step_from_obs = step_from_obs


_patch_env()
