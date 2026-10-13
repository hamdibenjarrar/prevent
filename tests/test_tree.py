import json
from datetime import date

import pytest

from prevent import TEMPLATE, TreeModel, check_tree, run_tree
from prevent.cli import main


def tree():
    def n(i, parent, text, p, v, r, k, **extra):
        return {"id": i, "parent": parent, "text": text, "plausibility": p, "value": v, "reversibility": r, "risk": k, **extra}

    return {
        "state": "A one person studio decides whether to open source its core tool.",
        "nodes": [
            n("a", None, "It opens the tool and 40 developers try it in the first year.", 0.6, 0.8, 0.8, 0.2),
            n("b", None, "It keeps the tool closed and sells two licences.", 0.5, 0.5, 0.6, 0.3),
            n("c", None, "It opens the tool and a bigger company copies it.", 0.3, 0.2, 0.3, 0.7),
            n("a1", "a", "Three companies pay for support.", 0.5, 0.85, 0.8, 0.2,
              summary="Open core with paid support", assumptions=["people adopt it"], moves_now=["write the docs"]),
            n("a2", "a", "Nobody pays and the maintainer burns out.", 0.4, 0.2, 0.4, 0.6),
            n("b1", "b", "Licence sales stay at two.", 0.5, 0.35, 0.6, 0.3),
            n("b2", "b", "A partner resells it in the region.", 0.4, 0.7, 0.7, 0.3),
            n("c1", "c", "The studio pivots to services.", 0.5, 0.45, 0.7, 0.3),
            n("c2", "c", "The studio closes.", 0.3, 0.05, 0.2, 0.9),
        ],
    }


def test_tree_runs_through_the_real_engine():
    res = run_tree(tree(), keep=3, beam=3, today=date(2026, 10, 13))
    assert res.horizon_years == 2 and res.explored == 9
    assert len(res.futures) == 3
    assert res.futures[0].summary == "Open core with paid support"
    assert res.futures[0].moves_now == ["write the docs"]
    assert [f.score for f in res.futures] == sorted([f.score for f in res.futures], reverse=True)
    assert res.futures[0].path[0].startswith("2027:") and res.futures[0].year == 2028


def test_check_tree_catches_errors_and_flattery():
    bad = tree()
    bad["nodes"][0]["value"] = 1.5
    bad["nodes"][3]["parent"] = "zz"
    errors, _ = check_tree(bad)
    assert any("value" in e for e in errors) and any("does not exist" in e for e in errors)
    hopeful = tree()
    for node in hopeful["nodes"]:
        node["value"] = 0.9
    _, warnings = check_tree(hopeful)
    assert any("no bad branch" in w for w in warnings)
    with pytest.raises(ValueError):
        run_tree(bad)


def test_template_is_a_valid_tree():
    errors, _ = check_tree(TEMPLATE)
    assert errors == []
    assert TreeModel(TEMPLATE).children[None]


def test_cli_tree_and_template(tmp_path, capsys):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(tree()), encoding="utf-8")
    assert main(["--tree", str(path), "--keep", "2", "--out", str(tmp_path / "f")]) == 0
    md = (tmp_path / "f.md").read_text(encoding="utf-8")
    assert "Open core with paid support" in md
    capsys.readouterr()
    assert main(["--template"]) == 0
    assert check_tree(json.loads(capsys.readouterr().out))[0] == []
