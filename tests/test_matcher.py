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


def test_split_is_deterministic_and_keyed_on_normalized_food():
    from jevmatcher.evaluate import split_of

    assert split_of("Peanut Butter!") == split_of("peanut butter")
    sides = [split_of(f"food {i}") for i in range(400)]
    assert 150 < sides.count("fit") < 250 and set(sides) == {"fit", "test"}


def test_nutrient_close():
    from jevmatcher.evaluate import nutrient_close

    idx = FnddsIndex("t", [
        Food("1", "A", nutrients={"kcal": 100, "protein_g": 5, "carb_g": 10, "fat_g": 2}),
        Food("2", "B", nutrients={"kcal": 108, "protein_g": 5.5, "carb_g": 11, "fat_g": 2.5}),
        Food("3", "C", nutrients={"kcal": 300, "protein_g": 5, "carb_g": 10, "fat_g": 2}),
    ])
    assert nutrient_close(idx, "1", "1") and nutrient_close(idx, "2", "1")
    assert not nutrient_close(idx, "3", "1") and not nutrient_close(idx, None, "1")


def test_fit_thresholds_meets_target_and_reports_none_when_unreachable():
    from jevmatcher.evaluate import fit_thresholds, score_thresholds

    def it(p1, ok):
        return {"p1": p1, "p2": 0.0, "top_is_none": False, "exact": ok, "near": ok}

    items = [it(0.95, True)] * 6 + [it(0.7, True)] * 2 + [it(0.7, False)] * 4 + [it(0.4, False)] * 3
    best = fit_thresholds(items, 0.9)
    assert best["precision"] >= 0.9 and best["t_high"] > 0.7 and best["accepted"] == 6
    assert score_thresholds(items, best["t_high"], best["margin"])["coverage"] == pytest.approx(6 / 15)
    assert fit_thresholds([it(0.9, False)] * 3, 0.9) is None


def test_evaluate_system_items_and_summary():
    from jevmatcher.evaluate import Label, evaluate_system

    m = matcher(FakeJev("sandwich, peanut"))
    out = evaluate_system(INDEX, m, [Label("peanut butter crackers", "54328100"), Label("saltine", "54319000")])
    items = out.pop("items")
    assert [i["exact"] for i in items] == [True, False]
    assert {"split", "true_description", "pred_code", "p1", "p2", "probabilities", "retrieval_top1", "near"} <= set(items[0])
    assert out["n"] == 2 and out["top1"] == 0.5 and "retrieval_only_top1" in out and "by_split" in out


def test_eval_out_creates_missing_directory(tmp_path, monkeypatch):
    from jevmatcher import cli
    from jevmatcher.evaluate import Label
    from jevmatcher.mockjev import MockJev

    idx_path = tmp_path / "idx.json"
    INDEX.save(idx_path)
    lab = tmp_path / "labels.csv"
    lab.write_text("food,code\npeanut butter,42202000\n")
    out = tmp_path / "nested" / "dir" / "run.json"
    monkeypatch.setattr("jevmatcher.retrieve.default_retriever", lambda idx: HybridRetriever([FuzzyRetriever(idx), TfidfRetriever(idx)]))
    cli.main(["--index", str(idx_path), "eval", str(lab), "--mock", "--out", str(out)])
    assert out.exists() and '"items"' in out.read_text()


def test_parallel_evaluation_matches_sequential():
    from jevmatcher.evaluate import Label, evaluate_system

    labels = [Label("peanut butter crackers", "54328100"), Label("saltine", "54319000"), Label("peanut butter", "42202000")] * 4
    seq = evaluate_system(INDEX, matcher(FakeJev("sandwich, peanut")), labels, batch=2, workers=1)
    par = evaluate_system(INDEX, matcher(FakeJev("sandwich, peanut")), labels, batch=2, workers=4)
    key = lambda out: [(i["food"], i["pred_code"], i["status"]) for i in out["items"]]  # noqa: E731
    assert key(seq) == key(par) and seq["top1"] == par["top1"]
