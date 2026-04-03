"""
inference.py — Official inference script for Email Triage OpenEnv.

Environment variables:
    API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace API token
    ENV_BASE_URL   Environment URL (default: http://localhost:7860)
"""

import os
import sys
import json
import time
import requests
from typing import Optional
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "email_triage_env"
MAX_STEPS    = 12
SEED         = 42

TASK_IDS = ["task_classify", "task_route", "task_respond"]

SYSTEM_PROMPT = """You are an expert customer support email triage agent.
Respond ONLY with a valid JSON object — no prose, no markdown.

For task_classify:  {"urgency": "urgent"|"normal"|"low"}
For task_route:     {"urgency": "...", "department": "billing"|"technical"|"returns"|"general"}
For task_respond:   {"urgency": "...", "department": "...", "response": "<your draft reply>"}

Urgency: urgent=time-critical/financial loss, normal=standard issue, low=informational/feedback
Department: billing=charges/invoices, technical=API/bugs/login, returns=wrong items/damage, general=other
"""

# ── Logging helpers (MANDATORY FORMAT) ───────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_str = error if error else "null"
    done_str  = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

# ── Environment helpers ───────────────────────────────────────────────────
def env_post(path: str, payload: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def env_get(path: str) -> dict:
    r = requests.get(f"{ENV_BASE_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

# ── LLM call ─────────────────────────────────────────────────────────────
def call_llm(client: OpenAI, obs: dict) -> dict:
    prompt = (
        f"TASK: {obs['task_description']}\n\n"
        f"FROM: {obs['sender']}\n"
        f"SUBJECT: {obs['subject']}\n"
        f"DATE: {obs['timestamp']}\n\n"
        f"BODY:\n{obs['body']}\n\n"
        "Respond with JSON only."
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                return {"urgency": "normal"}
            time.sleep(1 + attempt)
    return {"urgency": "normal"}

# ── Run one task episode ──────────────────────────────────────────────────
def run_task(client: OpenAI, task_id: str) -> dict:
    log_start(task_id, BENCHMARK, MODEL_NAME)

    obs     = env_post("/reset", {"task_id": task_id, "seed": SEED})
    rewards = []
    steps   = 0
    error   = None

    try:
        while not obs.get("done", False) and steps < MAX_STEPS:
            action_dict = call_llm(client, obs)
            action_dict.setdefault("urgency", "normal")

            action_str = json.dumps(action_dict, separators=(",", ":"))

            step_result = env_post("/step", {
                "urgency":    action_dict.get("urgency"),
                "department": action_dict.get("department"),
                "response":   action_dict.get("response"),
            })

            reward = step_result.get("reward", 0.0)
            done   = step_result.get("done", False)
            error  = step_result.get("info", {}).get("error")
            steps += 1
            rewards.append(reward)

            log_step(steps, action_str, reward, done, error)

            obs = step_result.get("observation", {})
            if done:
                break

    except Exception as e:
        error = str(e)
        log_step(steps + 1, "null", 0.0, True, error)

    avg_score = sum(rewards) / len(rewards) if rewards else 0.0
    success   = avg_score >= 0.5
    log_end(success, steps, rewards)

    return {
        "task_id":   task_id,
        "avg_score": round(avg_score, 4),
        "steps":     steps,
        "success":   success,
        "rewards":   rewards,
    }

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.", flush=True)
        sys.exit(1)

    # Health check
    try:
        health = env_get("/health")
        print(f"Environment healthy: {health}", flush=True)
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {ENV_BASE_URL}: {e}", flush=True)
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    all_results = {}
    for task_id in TASK_IDS:
        result = run_task(client, task_id)
        all_results[task_id] = result

    # Save scores
    output = {
        "model":    MODEL_NAME,
        "seed":     SEED,
        "env_url":  ENV_BASE_URL,
        "results":  all_results,
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n========================================", flush=True)
    print(f"  RESULTS  (model={MODEL_NAME})", flush=True)
    print("========================================", flush=True)
    diff = {"task_classify": "easy", "task_route": "medium", "task_respond": "hard"}
    for tid, r in all_results.items():
        print(f"  {tid:25s} [{diff[tid]:6s}]  score={r['avg_score']:.4f}  steps={r['steps']}", flush=True)
    print("========================================", flush=True)
    print(f"Saved to baseline_scores.json", flush=True)

if __name__ == "__main__":
    main()