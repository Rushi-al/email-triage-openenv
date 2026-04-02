#!/usr/bin/env python3
"""
baseline.py — Official baseline inference script for Email Triage OpenEnv.

Runs an OpenAI-compatible LLM agent against all 3 tasks and prints a
reproducible score table. Uses seed=42 for deterministic episode ordering.

Prerequisites:
    pip install openai requests

Usage:
    export OPENAI_API_KEY=sk-...
    export ENV_BASE_URL=http://localhost:7860   # or your HF Space URL
    python baseline.py

    # Override model:
    MODEL=gpt-4o python baseline.py

    # Run only specific tasks:
    python baseline.py --tasks task_classify task_route
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baseline")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
SEED = 42

SYSTEM_PROMPT = """You are an expert customer support email triage agent.
Read the support email carefully and respond with a valid JSON object ONLY.
No prose, no markdown fences, just a JSON object.

For task_classify:    {"urgency": "urgent"|"normal"|"low"}
For task_route:       {"urgency": "urgent"|"normal"|"low", "department": "billing"|"technical"|"returns"|"general"}
For task_respond:     {"urgency": "...", "department": "...", "response": "<your draft reply>"}

Urgency rules:
  urgent  — time-critical, financial loss, production down, very distressed customer
  normal  — standard issue, needs timely response
  low     — informational, feature requests, general questions

Department rules:
  billing    — charges, invoices, subscriptions, refunds, payments
  technical  — API, code, login, bugs, server errors, integrations
  returns    — returns, wrong items, damaged goods, replacements
  general    — product questions, feedback, feature requests, other
"""


def call_api(method: str, path: str, payload: Optional[Dict] = None) -> Dict:
    url = f"{ENV_BASE_URL}{path}"
    r = (requests.post(url, json=payload, timeout=30) if method == "POST"
         else requests.get(url, timeout=30))
    r.raise_for_status()
    return r.json()


def call_llm(prompt: str) -> Dict:
    try:
        from openai import OpenAI
    except ImportError:
        log.error("Install openai: pip install openai")
        sys.exit(1)

    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    for attempt in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                seed=SEED,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            log.warning(f"LLM attempt {attempt+1} failed: {e}")
            time.sleep(1 + attempt)
    return {"urgency": "normal"}


def run_task(task_id: str) -> Dict[str, Any]:
    log.info(f"▶ Task: {task_id}")
    obs = call_api("POST", "/reset", {"task_id": task_id, "seed": SEED})
    total_reward, steps = 0.0, 0

    while not obs.get("done", False):
        prompt = (
            f"TASK: {obs['task_description']}\n\n"
            f"FROM: {obs['sender']}\n"
            f"SUBJECT: {obs['subject']}\n"
            f"DATE: {obs['timestamp']}\n\n"
            f"BODY:\n{obs['body']}\n\n"
            "Respond with JSON only."
        )
        action = call_llm(prompt)
        action.setdefault("urgency", "normal")

        result = call_api("POST", "/step", {
            "urgency": action.get("urgency"),
            "department": action.get("department"),
            "response": action.get("response"),
        })

        reward = result.get("reward", 0.0)
        total_reward += reward
        steps += 1
        log.info(
            f"  [{steps:2d}] email={obs.get('email_id'):5s}  "
            f"urgency={action.get('urgency'):8s}  "
            f"dept={str(action.get('department')):12s}  "
            f"reward={reward:.3f}"
        )
        obs = result.get("observation", {})
        if result.get("done"):
            break

    avg = total_reward / steps if steps > 0 else 0.0
    log.info(f"  ✓ {task_id}: avg={avg:.4f}  steps={steps}")
    return {"task_id": task_id, "avg_score": round(avg, 4), "total_reward": round(total_reward, 4), "steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["task_classify", "task_route", "task_respond"])
    parser.add_argument("--output", default="baseline_scores.json")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY is not set. Export it and retry.")
        sys.exit(1)

    log.info(f"Checking {ENV_BASE_URL}/health ...")
    try:
        h = call_api("GET", "/health")
        log.info(f"  Environment: {h}")
    except Exception as e:
        log.error(f"Cannot reach environment at {ENV_BASE_URL}: {e}")
        sys.exit(1)

    results = {t: run_task(t) for t in args.tasks}

    width = 60
    print("\n" + "=" * width)
    print(f"  EMAIL TRIAGE OPENENV — BASELINE  (model={MODEL})")
    print("=" * width)
    diff_map = {"task_classify": "easy", "task_route": "medium", "task_respond": "hard"}
    for tid, r in results.items():
        print(f"  {tid:25s}  [{diff_map.get(tid,'?'):6s}]  "
              f"score={r['avg_score']:.4f}  steps={r['steps']}")
    print("=" * width)

    out = {"model": MODEL, "seed": SEED, "env_url": ENV_BASE_URL, "results": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
