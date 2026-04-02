"""
client.py — Python client for Email Triage OpenEnv.

Simple synchronous HTTP client that mirrors the server API exactly.
No extra dependencies beyond `requests`.

Usage:
    from client import EmailTriageClient

    client = EmailTriageClient("http://localhost:7860")
    obs    = client.reset("task_classify", seed=42)
    print(obs["subject"])

    result = client.step(urgency="urgent")
    print(result["reward"], result["done"])

    # Or use as context manager
    with EmailTriageClient("http://localhost:7860") as c:
        c.reset("task_respond")
        while True:
            r = c.step(urgency="normal", department="billing",
                       response="Dear customer, ...")
            if r["done"]:
                break
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import requests


class EmailTriageClient:
    """
    Synchronous HTTP client for the Email Triage OpenEnv environment.
    Matches the server's /reset, /step, /state, /tasks, /grader, /baseline endpoints.
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ── OpenEnv core API ──────────────────────────────────────────────────

    def reset(
        self,
        task_id: str = "task_classify",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start a new episode. Returns the first EmailObservation dict."""
        r = self._session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(
        self,
        urgency: str,
        department: Optional[str] = None,
        response: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a triage action.

        Returns dict with keys:
            observation (dict), reward (float), done (bool), info (dict)
        """
        payload = {
            "urgency": urgency,
            "department": department,
            "response": response,
            "reasoning": reasoning,
        }
        r = self._session.post(f"{self.base_url}/step", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        """Return the current TriageState dict."""
        r = self._session.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()

    # ── Extra endpoints ───────────────────────────────────────────────────

    def tasks(self) -> Dict[str, Any]:
        """List all tasks with their action schemas."""
        r = self._session.get(f"{self.base_url}/tasks", timeout=10)
        r.raise_for_status()
        return r.json()

    def grader(
        self,
        task_id: str,
        email_id: str,
        urgency: str,
        department: Optional[str] = None,
        response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score a single action against a specific email (offline)."""
        r = self._session.post(
            f"{self.base_url}/grader",
            json={
                "task_id": task_id,
                "email_id": email_id,
                "urgency": urgency,
                "department": department,
                "response": response,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def baseline(self) -> Dict[str, Any]:
        """Trigger and return the rule-based baseline scores."""
        r = self._session.get(f"{self.base_url}/baseline", timeout=120)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        r = self._session.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    # ── Context manager ───────────────────────────────────────────────────

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "EmailTriageClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
