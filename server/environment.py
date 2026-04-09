"""
environment.py - Core Email Triage OpenEnv environment.
"""

import random
from uuid import uuid4
from typing import Tuple, Dict, Any, Optional

from .models import TriageAction, EmailObservation, TriageState
from .email_dataset import EMAILS
from .graders import grade, TASK_DESCRIPTIONS, clamp_score

MAX_STEPS_PER_EPISODE = 20
VALID_TASK_IDS = ["task_classify", "task_route", "task_respond"]


class EmailTriageEnvironment:
    """
    Email Triage OpenEnv environment.

    Each episode the agent triages up to MAX_STEPS emails drawn from a shuffled
    dataset. Reward is computed per email via the task-specific grader.
    """

    def __init__(self):
        self._state = TriageState(episode_id=str(uuid4()), task_id="task_classify")
        self._email_queue: list = []
        self._current_email: Dict[str, Any] = {}
        self._cumulative_score: float = 0.0
        self._rng = random.Random()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset(self, task_id: str = "task_classify", seed: Optional[int] = None) -> EmailObservation:
        if task_id not in VALID_TASK_IDS:
            raise ValueError(f"Invalid task_id '{task_id}'. Must be one of {VALID_TASK_IDS}")
        if seed is not None:
            self._rng.seed(seed)

        shuffled = list(EMAILS)
        self._rng.shuffle(shuffled)
        self._email_queue = shuffled[:MAX_STEPS_PER_EPISODE]

        self._state = TriageState(
            episode_id=str(uuid4()),
            task_id=task_id,
            step_count=0,
            max_steps=len(self._email_queue),
            done=False,
        )
        self._cumulative_score = 0.0
        self._state.last_reward = clamp_score(0.0)
        self._state.cumulative_score = clamp_score(0.0)
        self._current_email = self._email_queue[0]

        return self._build_obs("New episode started. Triage this email.", reward=0.0, done=False)

    def step(self, action: TriageAction) -> Tuple[EmailObservation, float, bool, Dict]:
        if self._state.done:
            raise RuntimeError("Episode already done. Call reset() first.")

        gt = self._current_email["ground_truth"]
        reward_obj = grade(self._state.task_id, action, gt)

        self._state.step_count += 1
        self._state.emails_processed += 1
        self._cumulative_score += reward_obj.total
        self._state.last_reward = reward_obj.total
        self._state.current_email_id = self._current_email["email_id"]

        done = self._state.step_count >= len(self._email_queue)
        self._state.done = done
        avg_score = self._cumulative_score / self._state.emails_processed
        self._state.cumulative_score = clamp_score(round(avg_score, 4))

        if done:
            obs = EmailObservation(
                email_id="TERMINAL",
                subject="[Episode Complete]",
                body=(
                    f"Episode finished. Processed {self._state.emails_processed} emails "
                    f"with average score {avg_score:.3f}."
                ),
                sender="system@openenv",
                timestamp=self._current_email["timestamp"],
                task_id=self._state.task_id,
                task_description=TASK_DESCRIPTIONS[self._state.task_id],
                step_feedback=(
                    f"Episode complete! Avg score: {avg_score:.3f}. "
                    f"Last: {reward_obj.details}"
                ),
                reward=clamp_score(reward_obj.total),
                done=True,
                score=clamp_score(round(avg_score, 4)),
            )
        else:
            self._current_email = self._email_queue[self._state.step_count]
            self._state.current_email_id = self._current_email["email_id"]
            obs = self._build_obs(
                f"Scored {reward_obj.total:.3f}. {reward_obj.details}",
                reward=reward_obj.total,
                done=False,
            )

        info = reward_obj.model_dump()
        info["emails_remaining"] = len(self._email_queue) - self._state.step_count
        return obs, clamp_score(reward_obj.total), done, info

    @property
    def state(self) -> TriageState:
        state = self._state.model_copy()
        state.last_reward = clamp_score(state.last_reward)
        state.cumulative_score = clamp_score(state.cumulative_score)
        return state

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _build_obs(self, feedback: str, reward: float, done: bool) -> EmailObservation:
        avg = (
            self._cumulative_score / self._state.emails_processed
            if self._state.emails_processed > 0
            else 0.0
        )
        return EmailObservation(
            email_id=self._current_email["email_id"],
            subject=self._current_email["subject"],
            body=self._current_email["body"],
            sender=self._current_email["sender"],
            timestamp=self._current_email["timestamp"],
            task_id=self._state.task_id,
            task_description=TASK_DESCRIPTIONS[self._state.task_id],
            step_feedback=feedback,
            reward=clamp_score(round(reward, 4)),
            done=done,
            score=clamp_score(round(avg, 4)),
        )
