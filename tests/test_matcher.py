from pathlib import Path

import numpy as np
import pytest

from jevmatcher.evaluate import coverage_precision, expected_calibration_error, load_labels
from jevmatcher.fndds import Food, FnddsIndex
from jevmatcher.jev import ChoiceAnswer, JevResponse, choice_question, parse_response
from jevmatcher.matcher import NONE_OPTION, FnddsMatcher, Thresholds, decide
from jevmatcher.normalize import normalize
from jevmatcher.retrieve import FuzzyRetriever, HybridRetriever, TfidfRetriever

FOODS = [
    Food("54328100", "Crackers, sandwich, peanut butter filled", ("Lance",), nutrients={"kcal": 494.0}),
    Food("54319000", "Crackers, saltine", nutrients={"kcal": 420.0}),
    Food("42202000", "Peanut butter", nutrients={"kcal": 598.0}),
    Food("53105275", "Cake or cupcake, chocolate, no icing", nutrients={"kcal": 370.0}),
]
INDEX = FnddsIndex("test", FOODS)


class FakeJev:
    """Puts `p` on the option whose text contains `pick`, spreading the rest evenly."""

    def __init__(self, pick: str, p: float = 0.9, noul: float = 0.9):
        self.pick, self.p, self.noul, self.calls = pick, p, noul, []

    def ask(self, state, questions):
        self.calls.append(questions)
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = self.noul
                continue
            opts = list(q["criteria"])
            hit = next((o for o in opts if self.pick in o), NONE_OPTION)
            rest = (1 - self.p) / (len(opts) - 1)
            probs = {o: (self.p if o == hit else rest) for o in opts}
            answers[qid] = ChoiceAnswer(hit, probs, self.p)
        return JevResponse("fake-1", answers, input_tokens=1000)


def matcher(jev, **kw):
    return FnddsMatcher(INDEX, HybridRetriever([FuzzyRetriever(INDEX), TfidfRetriever(INDEX)]), jev, k=4, **kw)


def test_normalize_expands_abbreviations_and_typos():
    assert normalize("Lance PB Crakers!") == "lance peanut butter crackers"


def test_accepts_confident_match_and_returns_provenance():
    r = matcher(FakeJev("sandwich, peanut")).match("peanut butter crackers", {"note": "Lance"})
    assert (r.status, r.code) == ("accepted", "54328100")
    assert r.matcher["fndds_release"] == "test" and r.matcher["prompt_version"]
    assert r.alternatives[0].code == "54328100"


def test_none_of_these_is_no_match():
    r = matcher(FakeJev("zzz")).match("chocolate cake")
    assert r.status == "no_match" and r.code is None


def test_low_margin_goes_to_review():
    r = matcher(FakeJev("saltine", p=0.4)).match("crackers")
    assert r.status == "review" and r.code is None and r.alternatives


def test_decide_rules():
    th = Thresholds(t_high=0.6, margin=0.2, t_verify=0.5)
    assert decide([("a", 0.7), ("b", 0.1)], th) == "accepted"
    assert decide([("a", 0.7), ("b", 0.6)], th) == "review"
    assert decide([("a", 0.7), ("b", 0.1)], th, verify_p=0.2) == "review"
    assert decide([(NONE_OPTION, 0.9), ("a", 0.1)], th) == "no_match"


def test_shuffles_ask_permuted_questions_and_average():
    jev = FakeJev("saltine", p=0.9)
    m = matcher(jev, shuffles=3)
    r = m.match("crackers")
    qs = jev.calls[0]
    assert len(qs) == 3
    orders = {tuple(q["criteria"]) for q in qs.values()}
    assert len(orders) > 1
    assert r.code == "54319000" and r.confidence == pytest.approx(0.9)


def test_verify_can_downgrade():
    m = matcher(FakeJev("saltine", noul=0.1), verify=True, thresholds=Thresholds(t_verify=0.5))
    assert m.match("crackers").status == "review"


def test_match_many_is_one_request():
    jev = FakeJev("Peanut butter")
    out = matcher(jev).match_many([("peanut butter", None), ("pb", None)])
    assert len(out) == 2 and len(jev.calls) == 1 and len(jev.calls[0]) == 2


def test_choice_option_cap():
    with pytest.raises(ValueError):
        choice_question("q", {str(i): None for i in range(256)})


def test_parse_response_matches_documented_shape():
    body = {
        "model": "jev-1.13.0",
        "answers": {
            "a": {"type": "choice", "choice": "x", "probabilities": {"x": 0.8, "y": 0.2}, "confidence": 0.6},
            "b": {"type": "noul", "noul": 0.95},
        },
        "usage": {"input_tokens": 300, "output_tokens": 20},
    }
    r = parse_response(body)
    assert r.answers["a"].choice == "x" and r.answers["b"] == 0.95
    assert r.cost_usd == pytest.approx(300 * 0.042 / 1e6)


def test_calibration_and_coverage():
    conf, ok = np.array([0.9] * 10), np.array([1.0] * 9 + [0.0])
    assert expected_calibration_error(conf, ok) == pytest.approx(0.0)
    assert coverage_precision(conf, ok, [0.5, 0.95])[1]["coverage"] == 0.0


def test_load_labels_skips_non_numeric(tmp_path: Path):
    p = tmp_path / "l.csv"
    p.write_text("food,code,tier\npeanut butter,42202000,easy\nmystery,n/a,hard\ncrackers,54319000.0,easy\n")
    assert [(l.food, l.code) for l in load_labels(p)] == [("peanut butter", "42202000"), ("crackers", "54319000")]


def test_head_noun_prefers_general_entry():
    from jevmatcher.retrieve import HeadNounRetriever

    idx = FnddsIndex("t", [Food("1", "Roll, egg bread"), Food("2", "Egg omelet or scrambled egg, NS as to fat"), Food("3", "Egg, whole, boiled")])
    top = HeadNounRetriever(idx).search("eggs", 3)
    assert [idx.foods[i].code for i, _ in top][0] == "3"


def test_mock_jev_and_web_app(tmp_path):
    from fastapi.testclient import TestClient

    from jevmatcher.mockjev import MockJev
    from jevmatcher.web import create_app

    p = tmp_path / "idx.json"
    INDEX.save(p)
    retr = HybridRetriever([FuzzyRetriever(INDEX), TfidfRetriever(INDEX)])
    c = TestClient(create_app(live=False, index_path=p, retriever=retr))
    assert c.get("/api/info").json()["mode"] == "mock"
    d = c.post("/api/match", json={"food": "peanut butter crackers", "k": 5, "verify": True}).json()
    r = d["result"]
    assert d["candidates"] and r["input_tokens"] > 0 and r["cost_usd"] > 0
    assert r["timings_s"]["jev_match"] > 0 and r["probabilities"][0]["p"] >= r["probabilities"][-1]["p"]
    assert c.post("/api/match", json={"food": "", "k": 5}).status_code == 422
    assert MockJev(simulate_latency=False).ask("s", {"q": {"type": "noul", "instructions": {"food": "a", "candidate": "a"}}}).answers["q"] == 1.0


def test_dotenv_loads_without_overriding(tmp_path, monkeypatch):
    from jevmatcher.env import load_dotenv

    f = tmp_path / ".env"
    f.write_text("# c\nA_KEY='quoted'\nexport B_KEY=plain\nC_KEY=\"dq\"\nKEEP=fromfile\n")
    monkeypatch.setenv("KEEP", "fromenv")
    for k in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert load_dotenv(f) == [f]
    import os

    assert (os.environ["A_KEY"], os.environ["B_KEY"], os.environ["C_KEY"], os.environ["KEEP"]) == ("quoted", "plain", "dq", "fromenv")
    monkeypatch.delenv("A_KEY"); monkeypatch.delenv("B_KEY"); monkeypatch.delenv("C_KEY")
