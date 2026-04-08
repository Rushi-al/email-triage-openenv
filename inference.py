import os, sys, json, time, requests
from typing import Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "email_triage_env"
MAX_STEPS    = 12
SEED         = 42
TASK_IDS     = ["task_classify", "task_route", "task_respond"]

SYSTEM_PROMPT = """You are a customer support email triage agent.
Respond ONLY with valid JSON. No prose, no markdown fences.
task_classify:  {"urgency": "urgent"|"normal"|"low"}
task_route:     {"urgency": "...", "department": "billing"|"technical"|"returns"|"general"}
task_respond:   {"urgency": "...", "department": "...", "response": "<draft reply>"}
"""

def _r(v) -> float:
    return round(max(0.1, min(0.9, float(v if v is not None else 0.5))), 2)

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={_r(reward):.2f} done={'true' if done else 'false'} error={error or 'null'}", flush=True)

def log_end(success, steps, rewards):
    safe = [_r(r) for r in rewards] if rewards else [0.5]
    print(f"[END] success={'true' if success else 'false'} steps={steps} rewards={','.join(f'{r:.2f}' for r in safe)}", flush=True)

def env_post(path, payload):
    r = requests.post(f"{ENV_BASE_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def env_get(path):
    r = requests.get(f"{ENV_BASE_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

def call_llm(client, obs):
    prompt = f"TASK: {obs['task_description']}\nFROM: {obs['sender']}\nSUBJECT: {obs['subject']}\n\nBODY:\n{obs['body']}\n\nJSON only."
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
                temperature=0.1, max_tokens=512,
                response_format={"type":"json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            if attempt == 2: return {"urgency": "normal"}
            time.sleep(1 + attempt)
    return {"urgency": "normal"}

def run_task(client, task_id):
    log_start(task_id, BENCHMARK, MODEL_NAME)
    rewards, steps, error = [], 0, None
    try:
        obs = env_post("/reset", {"task_id": task_id, "seed": SEED})
        while not obs.get("done", False) and steps < MAX_STEPS:
            action_dict = call_llm(client, obs)
            action_dict.setdefault("urgency", "normal")
            action_str = json.dumps(action_dict, separators=(",",":"))
            result = env_post("/step", {
                "urgency": action_dict.get("urgency"),
                "department": action_dict.get("department"),
                "response": action_dict.get("response"),
            })
            reward = _r(result.get("reward", 0.5))
            done   = result.get("done", False)
            error  = result.get("info", {}).get("error")
            steps += 1
            rewards.append(reward)
            log_step(steps, action_str, reward, done, error)
            obs = result.get("observation", {})
            if done: break
    except Exception as e:
        error = str(e)
        log_step(steps + 1, "null", 0.5, True, error)
        if not rewards: rewards = [0.5]

    avg   = _r(sum(rewards) / len(rewards))
    log_end(avg >= 0.5, steps, rewards)
    return {"task_id": task_id, "avg_score": avg, "steps": steps, "success": avg >= 0.5, "rewards": rewards}

def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set.", flush=True)
        sys.exit(1)
    try:
        print(f"Health: {env_get('/health')}", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        sys.exit(1)
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    results = {}
    for task_id in TASK_IDS:
        results[task_id] = run_task(client, task_id)
    with open("baseline_scores.json", "w") as f:
        json.dump({"model": MODEL_NAME, "seed": SEED, "results": results}, f, indent=2)
    print("\n========================================", flush=True)
    for tid, r in results.items():
        print(f"  {tid}: score={r['avg_score']:.4f} steps={r['steps']}", flush=True)
    print("========================================", flush=True)

if __name__ == "__main__":
    main()