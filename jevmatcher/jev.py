"""Thin client for TypeSafe's System One endpoint (docs: https://docs.typesafe.ai/api).

Follows the public API reference (checked 2026-09-29): a Choice question is
{type, instructions, criteria: {option: description|null}} and the answer carries `choice`,
`probabilities`, `confidence`. Confirm against the live API before trusting timings or calibration.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
INPUT_USD_PER_TOKEN = 0.042 / 1e6  # vendor-quoted early-access price; unverified


def choice_question(instructions: Any, criteria: dict[str, Any]) -> dict:
    if len(criteria) > 255:
        raise ValueError(f"Choice supports at most 255 options, got {len(criteria)}")
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul_question(instructions: Any, criteria: dict[str, Any] | None = None) -> dict:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    return q


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class JevResponse:
    model: str
    answers: dict[str, Any]  # ChoiceAnswer | float (noul)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * INPUT_USD_PER_TOKEN


def parse_response(body: dict, latency_s: float = 0.0) -> JevResponse:
    answers: dict[str, Any] = {}
    for qid, a in body["answers"].items():
        if a["type"] == "choice":
            answers[qid] = ChoiceAnswer(a["choice"], a["probabilities"], a["confidence"])
        elif a["type"] == "noul":
            answers[qid] = float(a["noul"])
        else:
            answers[qid] = a
    usage = body.get("usage", {})
    return JevResponse(
        model=body["model"],
        answers=answers,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        latency_s=latency_s,
    )


class Jev(Protocol):
    def ask(self, state: Any, questions: dict[str, dict]) -> JevResponse: ...


@dataclass
class JevClient:
    api_key: str = field(default_factory=lambda: os.environ["TYPESAFE_API_KEY"])
    model: str = "jev-latest"
    endpoint: str = ENDPOINT
    timeout: float = 30.0
    max_retries: int = 3
    _http: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # One reused client: connection reuse matters for latency measurements.
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout
        )

    def ask(self, state: Any, questions: dict[str, dict]) -> JevResponse:
        payload = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._http.post(self.endpoint, json=payload)
                dt = time.perf_counter() - t0
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(float(r.headers.get("retry-after", 2**attempt * 0.5)))
                    continue
                r.raise_for_status()
                return parse_response(r.json(), dt)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                time.sleep(2**attempt * 0.5)
        raise RuntimeError("unreachable")
