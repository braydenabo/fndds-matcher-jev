"""Rule-based normalization. Dictionary first; no LLM until error analysis says it is needed."""
from __future__ import annotations

import re

ABBREVIATIONS = {
    "pb": "peanut butter",
    "pbj": "peanut butter and jelly",
    "pb&j": "peanut butter and jelly",
    "oj": "orange juice",
    "bbq": "barbecue",
    "w/": "with",
    "w/o": "without",
    "mac": "macaroni",
    "veg": "vegetable",
    "veggies": "vegetables",
    "choc": "chocolate",
    "cheez": "cheese",
    "chix": "chicken",
    "sammie": "sandwich",
    "sandwhich": "sandwich",
}

# Common misspellings seen in logged food strings; extend from error analysis.
TYPOS = {
    "crakers": "crackers",
    "cracker": "cracker",
    "chiken": "chicken",
    "brocoli": "broccoli",
    "spagetti": "spaghetti",
    "cheeze": "cheese",
    "yogurt": "yogurt",
    "yoghurt": "yogurt",
    "omlette": "omelet",
    "omelette": "omelet",
}

_PUNCT = re.compile(r"[^\w\s&/%]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    s = text.lower().strip()
    s = _PUNCT.sub(" ", s)
    tokens = []
    for tok in _WS.split(s):
        if not tok:
            continue
        tok = ABBREVIATIONS.get(tok, tok)
        tok = TYPOS.get(tok, tok)
        tokens.append(tok)
    return _WS.sub(" ", " ".join(tokens)).strip()
