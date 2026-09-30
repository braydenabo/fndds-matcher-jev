"""Stage 2: top-K candidate retrieval. Recall@K here is the ceiling for the whole system."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

from .fndds import FnddsIndex
from .normalize import normalize


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (food index, score) pairs, best first."""


class FuzzyRetriever:
    """Lexical: rapidfuzz token-set + partial ratio over main and additional descriptions."""

    def __init__(self, index: FnddsIndex):
        self._texts = [normalize(f.search_text) for f in index.foods]
        self._titles = [normalize(f.description) for f in index.foods]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = normalize(query)
        scores = np.zeros(len(self._texts))
        for texts, weight in ((self._titles, 1.0), (self._texts, 0.6)):
            for scorer in (fuzz.token_set_ratio, fuzz.partial_ratio):
                res = process.extract(q, texts, scorer=scorer, limit=k * 2)
                for _, score, i in res:
                    scores[i] = max(scores[i], weight * score)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


_STOP = {"and", "or", "with", "of", "the", "a", "in", "on", "nfs", "ns", "as", "to"}


def _tokens(text: str) -> set[str]:
    return {_stem(t) for t in normalize(text).split() if t not in _STOP}


class HeadNounRetriever:
    """Scores the query against each entry's head (text before the first comma).

    Generic strings ("egg", "carrots") should land on the general entry ("Egg, whole, ...")
    rather than on long descriptions that merely contain the word ("Roll, egg bread").
    Score = query coverage of the head x sqrt(head coverage by the query).
    """

    def __init__(self, index: FnddsIndex):
        self._heads = [_tokens(f.description.split(",")[0]) for f in index.foods]
        self._full = [_tokens(f.description) for f in index.foods]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = _tokens(query)
        if not q:
            return []
        scores = np.zeros(len(self._heads))
        for i, (head, full) in enumerate(zip(self._heads, self._full)):
            hit = len(q & head)
            if hit:
                scores[i] = (hit / len(q)) * (hit / len(head)) ** 0.5
            elif q & full:  # query words only in the qualifiers, e.g. "fried"
                scores[i] = 0.2 * len(q & full) / len(q)
        # prefer shorter (more general) descriptions among ties
        scores -= 1e-4 * np.array([len(f) for f in self._full])
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


class TfidfRetriever:
    """Char n-gram TF-IDF: tolerant of typos and word-order changes; needs no model download."""

    def __init__(self, index: FnddsIndex):
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._m = self._vec.fit_transform([normalize(f.search_text) for f in index.foods])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        sims = (self._m @ self._vec.transform([normalize(query)]).T).toarray().ravel()
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top if sims[i] > 0]


class EmbeddingRetriever:
    """Dense retrieval. `embed` maps list[str] -> (n, d) float array (e.g. sentence-transformers)."""

    def __init__(self, index: FnddsIndex, embed):
        self._embed = embed
        m = np.asarray(embed([f.description for f in index.foods]), dtype=np.float32)
        self._m = m / np.linalg.norm(m, axis=1, keepdims=True)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = np.asarray(self._embed([query]), dtype=np.float32)[0]
        sims = self._m @ (q / np.linalg.norm(q))
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


class HybridRetriever:
    """Union of member retrievers, fused with reciprocal-rank fusion, deduped."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60):
        self._rs = retrievers
        self._rrf_k = rrf_k

    @property
    def members(self) -> list[Retriever]:
        return self._rs

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for r in self._rs:
            for rank, (i, _) in enumerate(r.search(query, k)):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self._rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda kv: -kv[1])[:k]


class FastEmbedder:
    """fastembed (ONNX, no torch) wrapper with an on-disk cache for the FNDDS side.

    Callable: list[str] -> (n, d) array. Batches identical to `docs` come from the cache,
    so index build is a one-time cost per (model, release).
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: str = "data/cache", tag: str = ""):
        from pathlib import Path

        self.model_name, self._model = model, None
        self._cache = Path(cache_dir)
        self._tag = tag

    def _m(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    def __call__(self, texts: list[str]) -> np.ndarray:
        import hashlib

        key = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]
        path = self._cache / f"{self.model_name.split('/')[-1]}-{self._tag}-{key}.npy"
        if len(texts) > 100 and path.exists():
            return np.load(path)
        out = np.array(list(self._m().embed(texts)), dtype=np.float32)
        if len(texts) > 100:
            self._cache.mkdir(parents=True, exist_ok=True)
            np.save(path, out)
        return out


def default_retriever(index: FnddsIndex, embed="bge-small", dense: bool = True, embed_model: str | None = None) -> HybridRetriever:
    """fuzzy + char TF-IDF + head-noun + dense (bge-small). Chosen by Recall@30 on the labeled set;
    see README. Pass dense=False to skip the model download, or embed=<callable> to swap models."""
    rs: list[Retriever] = [FuzzyRetriever(index), TfidfRetriever(index), HeadNounRetriever(index)]
    if dense:
        if embed_model:  # any fastembed model name, e.g. "thenlper/gte-large"
            fn = FastEmbedder(embed_model, tag=index.release)
        else:
            fn = FastEmbedder("BAAI/bge-small-en-v1.5", tag=index.release) if embed == "bge-small" else embed
        rs.append(EmbeddingRetriever(index, fn))
    return HybridRetriever(rs)
