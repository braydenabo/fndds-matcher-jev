"""Demo web app: type a food, see retrieval candidates and the Jev decision with tokens/cost/latency.

Live by default: calls the real Jev API and needs TYPESAFE_API_KEY. Set JEV_MOCK=1 to use the
offline simulator instead (no key, no cost; tokens and latency are estimates).
Run: .venv/bin/uvicorn jevmatcher.web:create_app --factory --port 8000
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .fndds import FnddsIndex
from .matcher import FnddsMatcher, Thresholds
from .mockjev import MockJev
from .normalize import normalize
from .retrieve import HybridRetriever, default_retriever

ROOT = Path(__file__).resolve().parent.parent
LABELS = {"FuzzyRetriever": "Fuzzy", "TfidfRetriever": "TF-IDF", "HeadNounRetriever": "Head-noun", "EmbeddingRetriever": "bge-small"}


class MatchRequest(BaseModel):
    food: str = Field(min_length=1, max_length=200)
    context: str = Field(default="", max_length=300)
    k: int = Field(default=30, ge=5, le=100)
    shuffles: int = Field(default=1, ge=1, le=3)
    verify: bool = False


def create_app(live: bool | None = None, index_path: Path | None = None, retriever=None) -> FastAPI:
    live = not os.environ.get("JEV_MOCK") if live is None else live
    if live and not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY is not set. Export it, or set JEV_MOCK=1 for the simulator.")
    index = FnddsIndex.load(index_path or ROOT / "data" / "fndds_index.json")
    retr: HybridRetriever = retriever or default_retriever(index)
    if live:
        from .jev import JevClient

        jev = JevClient()
    else:
        jev = MockJev()
    retr.search("warm up", 5)  # load the embedding model now so the first request is not slow
    app = FastAPI(title="jevMatcher demo")

    @app.get("/")
    def home():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/api/info")
    def info():
        return {"mode": "live" if live else "mock", "release": index.release, "foods": len(index)}

    @app.post("/api/match")
    def match(req: MatchRequest):
        t_start = time.perf_counter()
        matcher = FnddsMatcher(
            index, retr, jev, k=req.k, shuffles=req.shuffles, verify=req.verify,
            thresholds=Thresholds(t_verify=0.5 if req.verify else None),
        )
        ctx = {"note": req.context.strip()} if req.context.strip() else None
        result = matcher.match(req.food, ctx)

        # Retrieval view: fused list plus each member's own rank for the same query.
        q = normalize(req.food)
        fused = retr.search(q, req.k)
        member_ranks, lanes = {}, {}
        for m in retr.members:
            name = LABELS.get(type(m).__name__, type(m).__name__)
            hits = m.search(q, req.k)
            member_ranks[name] = {i: r + 1 for r, (i, _) in enumerate(hits)}
            lanes[name] = [
                {"rank": r + 1, "code": index.foods[i].code, "description": index.foods[i].description}
                for r, (i, _) in enumerate(hits[:4])
            ]
        prob = {p["code"]: p["p"] for p in result.probabilities if p["code"]}
        candidates = []
        for rank, (i, score) in enumerate(fused, 1):
            f = index.foods[i]
            candidates.append({
                "rank": rank, "code": f.code, "description": f.description,
                "additional": list(f.additional)[:6], "fused_score": round(score, 5),
                "methods": {name: ranks.get(i) for name, ranks in member_ranks.items()},
                "p": prob.get(f.code),
            })
        return {
            "query": req.food, "normalized": q,
            "mode": "live" if live else "mock",
            "candidates": candidates,
            "lanes": lanes,
            "result": asdict(result),
            "wall_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }

    return app

