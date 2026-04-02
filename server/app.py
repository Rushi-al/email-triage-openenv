"""
app.py - FastAPI server exposing the OpenEnv HTTP interface.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .environment import EmailTriageEnvironment, VALID_TASK_IDS
from .models import TriageAction, EmailObservation, TriageState
from .graders import grade, TASK_DESCRIPTIONS
from .email_dataset import EMAILS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "A real-world customer support email triage environment. "
        "Agents learn to classify urgency, route to departments, and draft replies."
    ),
    version="1.0.0",
)

env = EmailTriageEnvironment()


# -----------------------------------------------------------------------------
# Request schemas
# -----------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_classify"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    urgency: str
    department: Optional[str] = None
    response: Optional[str] = None
    reasoning: Optional[str] = None


class GraderRequest(BaseModel):
    task_id: str
    email_id: str
    urgency: str
    department: Optional[str] = None
    response: Optional[str] = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "email_triage_env", "version": "1.0.0"}


@app.post("/reset", response_model=EmailObservation)
async def reset(req: ResetRequest):
    """Start a new episode. task_id: task_classify | task_route | task_respond"""
    if req.task_id not in VALID_TASK_IDS:
        raise HTTPException(400, f"Invalid task_id '{req.task_id}'. Must be one of {VALID_TASK_IDS}")
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        return obs
    except Exception as e:
        logger.exception("Error in /reset")
        raise HTTPException(500, str(e))


@app.post("/step")
async def step(req: StepRequest):
    """Submit a triage action for the current email."""
    try:
        action = TriageAction(
            urgency=req.urgency,
            department=req.department,
            response=req.response,
            reasoning=req.reasoning,
        )
    except Exception as e:
        raise HTTPException(422, f"Invalid action: {e}")
    try:
        obs, reward, done, info = env.step(action)
        return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        logger.exception("Error in /step")
        raise HTTPException(500, str(e))


@app.get("/state", response_model=TriageState)
async def state():
    """Return current internal environment state."""
    return env.state


@app.get("/tasks")
async def tasks():
    """List all tasks with their action schema."""
    return {
        "tasks": [
            {
                "id": "task_classify",
                "name": "Email Urgency Classification",
                "difficulty": "easy",
                "description": TASK_DESCRIPTIONS["task_classify"],
                "action_schema": {
                    "urgency": "required - urgent | normal | low",
                    "department": "NOT required",
                    "response": "NOT required",
                    "reasoning": "optional",
                },
            },
            {
                "id": "task_route",
                "name": "Department Routing + Classification",
                "difficulty": "medium",
                "description": TASK_DESCRIPTIONS["task_route"],
                "action_schema": {
                    "urgency": "required - urgent | normal | low",
                    "department": "required - billing | technical | returns | general",
                    "response": "NOT required",
                    "reasoning": "optional",
                },
            },
            {
                "id": "task_respond",
                "name": "Full Triage: Classify + Route + Draft Response",
                "difficulty": "hard",
                "description": TASK_DESCRIPTIONS["task_respond"],
                "action_schema": {
                    "urgency": "required - urgent | normal | low",
                    "department": "required - billing | technical | returns | general",
                    "response": "required - draft reply string",
                    "reasoning": "optional",
                },
            },
        ],
        "valid_urgency_values": ["urgent", "normal", "low"],
        "valid_department_values": ["billing", "technical", "returns", "general"],
    }


@app.post("/grader")
async def grader(req: GraderRequest):
    """Score a single action against a specific email (no full episode needed)."""
    if req.task_id not in VALID_TASK_IDS:
        raise HTTPException(400, f"Invalid task_id. Must be one of {VALID_TASK_IDS}")
    email_data = next((e for e in EMAILS if e["email_id"] == req.email_id), None)
    if not email_data:
        raise HTTPException(404, f"Email '{req.email_id}' not found")
    try:
        action = TriageAction(urgency=req.urgency, department=req.department, response=req.response)
    except Exception as e:
        raise HTTPException(422, f"Invalid action: {e}")
    reward = grade(req.task_id, action, email_data["ground_truth"])
    return {
        "email_id": req.email_id,
        "task_id": req.task_id,
        "action": action.model_dump(),
        "ground_truth": {
            "urgency": email_data["ground_truth"]["urgency"],
            "department": email_data["ground_truth"]["department"],
        },
        "reward": reward.model_dump(),
    }


@app.get("/baseline")
async def baseline():
    """Run the keyword-heuristic baseline and return scores for all 3 tasks."""
    from .baseline_agent import run_baseline
    try:
        scores = run_baseline()
        return {
            "baseline_agent": "keyword_heuristic",
            "scores": scores,
            "note": "Run inference_script.py with OPENAI_API_KEY for LLM baseline scores.",
        }
    except Exception as e:
        logger.exception("Error in /baseline")
        raise HTTPException(500, str(e))
