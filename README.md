---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - rl
  - agent
  - email
  - nlp
  - real-world
---

# 📧 Email Triage OpenEnv

A real-world OpenEnv-compatible environment where AI agents learn to triage a business support inbox: classify urgency, route to departments, and draft professional replies.

Domain: Customer support email operations — a task millions of humans do daily.

## Why Email Triage?

Email triage is genuinely complex for AI agents:
- Requires natural language understanding across domains (billing, technical, returns)
- Tests priority judgment (what's truly urgent vs. just loud?)
- Demands professional communication (Task 3)
- Has clear ground-truth labels enabling deterministic, reproducible scoring

## Environment Specification

| Property | Value |
|---|---|
| OpenEnv Version | 1.0 |
| Action Type | `TriageAction` (Pydantic model) |
| Observation Type | `EmailObservation` (Pydantic model) |
| Reward | Dense — partial credit per step (0.0–1.0) |
| Episode Length | 10 emails (shuffled from dataset of 10) |
| Tasks | 3 (easy → medium → hard) |
| Graders | Deterministic keyword + rule-based |

## Tasks

### Task 1 — `task_classify` (Easy)
Goal: Classify the urgency of each email.

Action fields required:
- `urgency`: `urgent` | `normal` | `low`

Grader: Full credit (1.0) for correct urgency, 0.0 for wrong.
Small penalty (-0.05) if unnecessary fields are included.

Expected score: Frontier LLMs ~0.85+, keyword baseline ~0.70

### Task 2 — `task_route` (Medium)
Goal: Classify urgency AND route to the correct department.

Action fields required:
- `urgency`: `urgent` | `normal` | `low`
- `department`: `billing` | `technical` | `returns` | `general`

Grader: 0.5 for urgency + 0.5 for department. Penalty if response is included.

Expected score: Frontier LLMs ~0.70+, keyword baseline ~0.55

### Task 3 — `task_respond` (Hard)
Goal: Classify urgency, route to department, AND draft a professional customer reply.

Action fields required:
- `urgency`: `urgent` | `normal` | `low`
- `department`: `billing` | `technical` | `returns` | `general`
- `response`: string — your draft reply to the customer

Grader breakdown (total = 1.0):
- Urgency correct → +0.25
- Department correct → +0.25
- Response provided → +0.10
- Response ≥50 words → +0.05
- Response ≥100 words → +0.05
- ≥1 required keyword present → +0.10
- ≥3 required keywords present → +0.10
- All required keywords present → +0.10
- Professional closing (Best regards, Sincerely, etc.) → +0.05
- No unfilled placeholders → +0.05

Expected score: Frontier LLMs ~0.65+, keyword baseline ~0.45

## Action Space

```python
class TriageAction(BaseModel):
    urgency: Literal["urgent", "normal", "low"]           # required always
    department: Optional[Literal["billing", "technical",
                                 "returns", "general"]]   # required for task 2+3
    response: Optional[str]                               # required for task 3
    reasoning: Optional[str]                              # optional, not scored
```

## Observation Space

```python
class EmailObservation(BaseModel):
    email_id: str           # e.g. "E001"
    subject: str            # email subject line
    body: str               # full email body
    sender: str             # sender email address
    timestamp: str          # ISO 8601
    task_id: str            # active task
    task_description: str   # what the agent must do
    step_feedback: str      # human-readable feedback on last action
    reward: float           # reward for last step (0.0 on reset)
    done: bool              # True when episode ends
    score: float            # running cumulative average score
```

## Reward Function Design

Rewards are dense — the agent receives a signal after every email, not just at the end. This enables:
- Gradient flow across the entire trajectory
- Clear partial credit for multi-component tasks
- Interpretable per-step debugging

Penalties discourage providing unnecessary fields (verbosity) and missing the response entirely on Task 3.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "healthy"}` |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Submit triage action |
| `GET` | `/state` | Internal episode state |
| `GET` | `/tasks` | All tasks + action schema |
| `POST` | `/grader` | Score one action offline |
| `GET` | `/baseline` | Run rule-based baseline, return scores |
| `GET` | `/docs` | Interactive Swagger UI |

## Quick Start

### 1. Local via Docker

```bash
# Build
docker build -t email-triage-openenv .

# Run
docker run -p 7860:7860 email-triage-openenv

# Health check
curl http://localhost:7860/health
```

### 2. Local via Python (no Docker)

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run an episode with curl

```bash
# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_classify", "seed": 42}'

# Submit action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"urgency": "urgent"}'

# Check state
curl http://localhost:7860/state
```

### 4. Python client

```python
from client import EmailTriageClient

with EmailTriageClient("http://localhost:7860") as c:
    obs = c.reset("task_respond", seed=42)
    
    while not obs.get("done"):
        print(f"Email: {obs['subject']}")
        
        result = c.step(
            urgency="urgent",
            department="billing",
            response="Dear Customer, I apologize for the issue. "
                     "Our billing team will resolve this within 24 hours. "
                     "Best regards, Support Team"
        )
        obs = result["observation"]
        print(f"  Reward: {result['reward']:.3f}")
```

### 5. Run the LLM inference baseline

```bash
export OPENAI_API_KEY=sk-...
export ENV_BASE_URL=http://localhost:7860
python baseline.py

# Or against your HF Space:
export ENV_BASE_URL=https://YOUR-USERNAME-email-triage-openenv.hf.space
python baseline.py
```

## Baseline Scores

| Task | Heuristic Baseline | LLM (Qwen2.5-72B) | Difficulty |
|---|---|---|---|
| `task_classify` | 0.70 | 0.82 | Easy |
| `task_route` | 0.74 | 0.78 | Medium |
| `task_respond` | 0.62 | 0.80 | Hard |

Episode length: 20 emails per episode (dataset of 20 realistic support emails).

## Project Structure

```
email-triage-env/
├── server/
│   ├── __init__.py
│   ├── app.py              # FastAPI server (all endpoints)
│   ├── environment.py      # Core env: reset() / step() / state()
│   ├── models.py           # Pydantic: TriageAction, EmailObservation, TriageState
│   ├── email_dataset.py    # 10 realistic emails with ground-truth labels
│   ├── graders.py          # Deterministic graders for all 3 tasks
│   └── baseline_agent.py   # Keyword heuristic agent (for /baseline endpoint)
├── client.py               # Python HTTP client
├── baseline.py             # LLM baseline (OpenAI API)
├── openenv.yaml            # OpenEnv metadata manifest
├── requirements.txt
├── Dockerfile
└── README.md
```

## Running Tests

```bash
pip install -r requirements.txt
pytest -q
```

## Deploying to Hugging Face Spaces

```bash
hf auth login
hf repos create email-triage-openenv --type space --space-sdk docker
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/email-triage-openenv
git push -u hf main
```

The Space will automatically build and run the Dockerfile. Port 7860 is used by default.

## License

Apache 2.0
