"""Stages 1-6: normalize -> retrieve -> Jev Choice -> (verify) -> decide."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from .fndds import FnddsIndex, Food
from .jev import ChoiceAnswer, Jev, choice_question, noul_question
from .normalize import normalize
from .retrieve import Retriever

NONE_OPTION = "none of these"
PROMPT_VERSION = "v1"
TASK = "Map a free-text food name to the closest USDA FNDDS food description."
CHOICE_INSTRUCTIONS = (
    "Which FNDDS food description best matches `food`, given `context`? "
    f"Choose '{NONE_OPTION}' if no candidate is a good match."
)


@dataclass(frozen=True)
class Thresholds:
    t_high: float = 0.6  # min top-1 probability to accept
    margin: float = 0.2  # min p_top1 - p_top2 to accept
    t_verify: float | None = None  # min Noul probability when verify is on


@dataclass
class Alternative:
    code: str
    description: str
    p: float


@dataclass
class MatchResult:
    status: str  # accepted | review | no_match
    code: str | None
    description: str | None
    confidence: float
    alternatives: list[Alternative]
    matcher: dict[str, Any]
    timings_s: dict[str, float] = field(default_factory=dict)
    candidate_codes: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    verify_p: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    probabilities: list[dict[str, Any]] = field(default_factory=list)  # top options incl. none, averaged


def decide(ranked: list[tuple[str, float]], th: Thresholds, verify_p: float | None = None) -> str:
    """ranked: (option, p) best first, including NONE_OPTION. Returns the status."""
    if not ranked or ranked[0][0] == NONE_OPTION:
        return "no_match"
    p1 = ranked[0][1]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    if p1 < th.t_high or p1 - p2 < th.margin:
        return "review"
    if th.t_verify is not None and verify_p is not None and verify_p < th.t_verify:
        return "review"
    return "accepted"


def _instructions(question: str, food: str, normalized: str, context: dict | None, **extra: Any) -> dict:
    ins: dict[str, Any] = {"question": question, "food": food}
    if normalized != food.lower().strip():
        ins["food_normalized"] = normalized
    if context:
        ins["context"] = context
    ins.update(extra)
    return ins


class FnddsMatcher:
    def __init__(
        self,
        index: FnddsIndex,
        retriever: Retriever,
        jev: Jev,
        k: int = 30,
        shuffles: int = 1,
        verify: bool = False,
        thresholds: Thresholds = Thresholds(),
        use_additional_as_criteria: bool = True,
        instructions: str | None = None,
        allow_none: bool = True,
        seed: int = 0,
        jev_model_label: str = "jev-latest",
    ):
        self.index, self.retriever, self.jev = index, retriever, jev
        self.k, self.shuffles, self.verify = k, shuffles, verify
        self.thresholds = thresholds
        self.use_additional = use_additional_as_criteria
        self.allow_none = allow_none
        base = instructions or CHOICE_INSTRUCTIONS
        if not allow_none and base == CHOICE_INSTRUCTIONS:
            base = CHOICE_INSTRUCTIONS.split(" Choose ")[0]
        self._choice_instructions = base
        self._rng = random.Random(seed)
        self._label = jev_model_label

    # -- stages -------------------------------------------------------------
    def candidates(self, food: str) -> list[Food]:
        return [self.index.foods[i] for i, _ in self.retriever.search(normalize(food), self.k)]

    def _criteria(self, foods: list[Food], order: list[int]) -> dict[str, Any]:
        crit: dict[str, Any] = {}
        for i in order:
            f = foods[i]
            extra = "; ".join(f.additional + f.common_names) if self.use_additional else ""
            crit[f.description] = extra or None
        if self.allow_none:
            crit[NONE_OPTION] = "No candidate is a good match for the food."
        return crit

    def _orders(self, n: int) -> list[list[int]]:
        orders = [list(range(n))]
        for _ in range(self.shuffles - 1):
            o = list(range(n))
            self._rng.shuffle(o)
            orders.append(o)
        return orders

    def _questions(self, food: str, norm: str, context: dict | None, foods: list[Food], prefix: str = "") -> dict:
        ins = _instructions(self._choice_instructions, food, norm, context)
        return {
            f"{prefix}match_{j}": choice_question(ins, self._criteria(foods, order))
            for j, order in enumerate(self._orders(len(foods)))
        }

    def _collect(self, answers: dict, foods: list[Food], prefix: str = "") -> list[tuple[str, float]]:
        totals: dict[str, float] = {}
        n = 0
        for j in range(self.shuffles):
            a: ChoiceAnswer = answers[f"{prefix}match_{j}"]
            for opt, p in a.probabilities.items():
                totals[opt] = totals.get(opt, 0.0) + p
            n += 1
        return sorted(((o, p / n) for o, p in totals.items()), key=lambda kv: -kv[1])

    def _result(
        self, ranked, verify_p, foods, model, timings, cost, candidate_codes, tokens=(0, 0)
    ) -> MatchResult:
        status = decide(ranked, self.thresholds, verify_p)
        top = ranked[0][0]
        alts = [
            Alternative(self.index.by_description[o].code, o, p)
            for o, p in ranked[:5]
            if o != NONE_OPTION
        ]
        best = None if top == NONE_OPTION else self.index.by_description[top]
        return MatchResult(
            status=status,
            code=best.code if best and status == "accepted" else None,
            description=best.description if best and status == "accepted" else None,
            confidence=ranked[0][1],
            alternatives=alts,
            matcher={
                "jev_model": model or self._label,
                "fndds_release": self.index.release,
                "retriever": type(self.retriever).__name__,
                "k": self.k,
                "shuffles": self.shuffles,
                "verify": self.verify,
                "prompt_version": PROMPT_VERSION,
                "thresholds": vars(self.thresholds),
            },
            timings_s=timings,
            candidate_codes=candidate_codes,
            cost_usd=cost,
            verify_p=verify_p,
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            probabilities=[
                {
                    "option": o,
                    "code": None if o == NONE_OPTION else self.index.by_description[o].code,
                    "p": p,
                }
                for o, p in ranked[:12]
            ],
        )

    # -- public API ---------------------------------------------------------
    def match(self, food: str, context: dict | None = None) -> MatchResult:
        return self.match_many([(food, context)])[0]

    def match_many(self, foods: list[tuple[str, dict | None]]) -> list[MatchResult]:
        """One Jev request carrying one Choice question (x shuffles) per food."""
        t0 = time.perf_counter()
        norms = [normalize(f) for f, _ in foods]
        t1 = time.perf_counter()
        cands = [self.candidates(f) for f, _ in foods]
        t2 = time.perf_counter()

        questions: dict[str, dict] = {}
        for n, ((food, ctx), norm, fs) in enumerate(zip(foods, norms, cands)):
            questions.update(self._questions(food, norm, ctx, fs, prefix=f"f{n}_"))
        resp = self.jev.ask({"task": TASK}, questions)
        t3 = time.perf_counter()

        rankings = [self._collect(resp.answers, fs, prefix=f"f{n}_") for n, fs in enumerate(cands)]
        verify_ps: list[float | None] = [None] * len(foods)
        cost, model = resp.cost_usd, resp.model
        tok_in, tok_out = resp.input_tokens, resp.output_tokens
        if self.verify:
            vq = {}
            for n, ((food, ctx), norm, ranked) in enumerate(zip(foods, norms, rankings)):
                if ranked[0][0] != NONE_OPTION:
                    vq[f"f{n}_verify"] = noul_question(
                        _instructions(
                            "Is `candidate` an acceptable FNDDS match for `food`?",
                            food, norm, ctx, candidate=ranked[0][0],
                        )
                    )
            if vq:
                vr = self.jev.ask({"task": TASK}, vq)
                cost += vr.cost_usd
                tok_in += vr.input_tokens
                tok_out += vr.output_tokens
                for n in range(len(foods)):
                    verify_ps[n] = vr.answers.get(f"f{n}_verify")
        t4 = time.perf_counter()

        share = 1 / len(foods)
        timings = {
            "normalize": (t1 - t0) * share,
            "retrieve": (t2 - t1) * share,
            "jev_match": (t3 - t2) * share,  # batch wall-clock, split evenly
            "jev_verify": (t4 - t3) * share,
        }
        return [
            self._result(r, v, fs, model, dict(timings), cost * share,
                         [f.code for f in fs], (round(tok_in * share), round(tok_out * share)))
            for r, v, fs in zip(rankings, verify_ps, cands)
        ]
