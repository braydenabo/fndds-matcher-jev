"""Offline stand-in for Jev so the app and tests never touch the real API or key.

Not a model: option probabilities come from rapidfuzz similarity between the food string and
each option, softmaxed. Token counts are estimated (~4 chars/token) and latency is simulated,
so cost/latency numbers from this are illustrative only, never a benchmark of Jev.
"""
from __future__ import annotations

import json
import math
import random
import time
from typing import Any

from rapidfuzz import fuzz

from .jev import ChoiceAnswer, JevResponse
from .normalize import normalize

NONE_OPTION = "none of these"


class MockJev:
    def __init__(self, temperature: float = 9.0, simulate_latency: bool = True, seed: int = 0):
        self.temperature, self.simulate_latency = temperature, simulate_latency
        self._rng = random.Random(seed)

    @staticmethod
    def _est_tokens(obj: Any) -> int:
        return max(1, len(json.dumps(obj)) // 4)

    def _score(self, food: str, option: str, extra: str = "") -> float:
        f, o = normalize(food), normalize(option)
        base = 0.5 * fuzz.token_set_ratio(f, o) + 0.5 * fuzz.WRatio(f, o)
        if extra:  # the option's description (brands, aliases) can only help
            base = max(base, 0.9 * fuzz.token_set_ratio(f, normalize(f"{option} {extra}")))
        return base

    def ask(self, state: Any, questions: dict[str, dict]) -> JevResponse:
        t0 = time.perf_counter()
        in_tokens = self._est_tokens({"state": state, "questions": questions})
        answers: dict[str, Any] = {}
        for qid, q in questions.items():
            ins = q["instructions"]
            food = ins["food_normalized"] if "food_normalized" in ins else ins["food"]
            if q["type"] == "noul":
                answers[qid] = self._score(food, ins["candidate"]) / 100
                continue
            opts = list(q["criteria"])
            scores = {o: self._score(food, o, q['criteria'][o] or '') for o in opts if o != NONE_OPTION}
            best = max(scores.values(), default=0.0)
            scores[NONE_OPTION] = max(0.0, 62.0 - best) + 20.0  # wins only when nothing fits
            z = {o: math.exp((s - max(scores.values())) / self.temperature) for o, s in scores.items()}
            tot = sum(z.values())
            probs = {o: z[o] / tot for o in opts}
            top = max(probs, key=probs.get)
            n = len(opts)
            conf = max(0.0, min(1.0, (n * probs[top] - 1) / (n - 1)))
            answers[qid] = ChoiceAnswer(top, probs, conf)
        out_tokens = 8 + 6 * len(answers)
        if self.simulate_latency:  # ~vendor-quoted 70-500 ms; scales with input size
            time.sleep(0.06 + in_tokens / 60000 + self._rng.random() * 0.04)
        return JevResponse("mock-jev (simulated)", answers, in_tokens, out_tokens, time.perf_counter() - t0)
