"""
models.py - Typed Pydantic models for the Email Triage OpenEnv environment.

All action, observation, reward, and state types are defined here.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Action Model
# -----------------------------------------------------------------------------

UrgencyLevel = Literal["urgent", "normal", "low"]
DepartmentType = Literal["billing", "technical", "returns", "general"]


class TriageAction(BaseModel):
    """
    The action an agent takes on a support email.

    Fields:
        urgency     : Required for all tasks. One of: urgent | normal | low
        department  : Required for task_route and task_respond.
                      One of: billing | technical | returns | general
        response    : Required for task_respond. The draft reply text.
        reasoning   : Optional chain-of-thought text (not scored, aids debugging).
    """

    urgency: UrgencyLevel = Field(
        ...,
        description="Urgency classification of the email: urgent, normal, or low",
    )
    department: Optional[DepartmentType] = Field(
        None,
        description="Department to route the email to (required for task_route and task_respond)",
    )
    response: Optional[str] = Field(
        None,
        description="Draft reply to the customer (required for task_respond)",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Optional explanation of the agent's decision (not scored)",
    )


# -----------------------------------------------------------------------------
# Observation Model
# -----------------------------------------------------------------------------


class EmailObservation(BaseModel):
    """
    The observation returned after reset() or step().

    Fields:
        email_id    : Unique identifier for the current email.
        subject     : Email subject line.
        body        : Email body text.
        sender      : Sender's email address.
        timestamp   : ISO 8601 timestamp the email was received.
        task_id     : Which task is currently active.
        task_description : Human-readable task goal.
        step_feedback   : Human-readable feedback on the last action (empty on reset).
        reward       : Reward received for the last action (0.0 on reset).
        done         : Whether the episode has ended.
        score        : Running cumulative score for the episode (0.0-1.0).
    """

    email_id: str = Field(..., description="Unique ID for this email")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    sender: str = Field(..., description="Sender email address")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    task_id: str = Field(..., description="Active task ID")
    task_description: str = Field(..., description="What the agent needs to accomplish")
    step_feedback: str = Field("", description="Feedback on the previous action")
    reward: float = Field(0.0, description="Reward for the last action")
    done: bool = Field(False, description="True if the episode is complete")
    score: float = Field(0.0, description="Cumulative episode score (0.0-1.0)")


# -----------------------------------------------------------------------------
# Reward Model
# -----------------------------------------------------------------------------


class TriageReward(BaseModel):
    """
    Detailed breakdown of the reward for the last action.

    Fields:
        total          : Overall reward [0.0-1.0]
        urgency_score  : Partial score for urgency classification [0.0-0.33]
        routing_score  : Partial score for department routing [0.0-0.33]
        response_score : Partial score for response quality [0.0-0.34]
        penalties      : Any penalties applied (e.g. empty response, wrong format)
        details        : Human-readable explanation
    """
    total: float = Field(..., description="Total reward strictly between 0 and 1")
    urgency_score: float = Field(0.0, description="Score for urgency classification")
    routing_score: float = Field(0.0, description="Score for department routing")
    response_score: float = Field(0.0, description="Score for response quality")
    penalties: float = Field(0.0, description="Penalties applied (negative contribution)")
    details: str = Field("", description="Human-readable reward explanation")


# -----------------------------------------------------------------------------
# State Model
# -----------------------------------------------------------------------------


class TriageState(BaseModel):
    """
    Internal environment state (returned by state() endpoint).
    """

    episode_id: str = Field(..., description="Current episode UUID")
    task_id: str = Field(..., description="Active task ID")
    step_count: int = Field(0, description="Steps taken in this episode")
    current_email_id: str = Field("", description="ID of the email currently being triaged")
    cumulative_score: float = Field(0.0, description="Running score for this episode")
    last_reward: float = Field(0.0, description="Reward from the most recent step")
    done: bool = Field(False, description="Whether the episode has ended")
    emails_processed: int = Field(0, description="Total emails triaged this episode")
    max_steps: int = Field(10, description="Maximum steps allowed per episode")
