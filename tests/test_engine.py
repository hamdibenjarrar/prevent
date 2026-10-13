import json
from datetime import date

from prevent import Engine, Node, ScriptedModel, score_node


def scripted():
    branch = json.dumps(["the project ships and finds 100 users", "the project stalls for a year", "a partner joins"])
    good = json.dumps({"plausibility": 0.8, "value": 0.9, "reversibility": 0.7, "risk": 0.2})
    bad = json.dumps({"plausibility": 0.6, "value": 0.2, "reversibility": 0.3, "risk": 0.8})
    lock = json.dumps({"summary": "A small product with real users", "assumptions": ["ships", "users come", "money lasts"],
                       "moves_now": ["cut scope", "talk to 10 people", "set a launch date"]})
    # per year: one branch call, then one score call per branch (3), repeated; lock calls at the end
    answers = []
    for _ in range(3):
        answers += [branch, good, bad, good]
    answers += [lock] * 50
    return ScriptedModel(answers)


def test_run_keeps_best_and_carries_moves():
    eng = Engine(scripted(), branches=3, beam=2, today=date(2026, 10, 8))
    res = eng.run("Hamdi builds Alpha", years=2, keep=2)
    assert res.horizon_years == 2 and res.explored > 0
    assert len(res.futures) == 2
    assert res.futures[0].score >= res.futures[1].score
    assert res.futures[0].year == 2028
    assert res.futures[0].moves_now and res.futures[0].assumptions
    assert res.futures[0].path[0].startswith("2027:")
    md = res.to_markdown()
    assert "Moves now:" in md and "Prevent: 2 years" in md
    json.loads(res.to_json())


def test_score_prefers_value_and_punishes_irreversible_risk():
    safe = Node(2027, "x", plausibility=0.8, value=0.8, reversibility=0.9, risk=0.2)
    reckless = Node(2027, "y", plausibility=0.8, value=0.8, reversibility=0.1, risk=0.9)
    assert score_node(safe) > score_node(reckless)


def test_bad_model_output_does_not_crash():
    eng = Engine(ScriptedModel(["not json at all"]), branches=2, beam=2, today=date(2026, 1, 1))
    res = eng.run("anything", years=3, keep=8)
    assert res.futures == [] and res.explored == 0
