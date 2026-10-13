"""Tree mode: an agent is the model.

An agent such as Claude, Alpha or a human analyst writes every branch and every score into one JSON file.
Prevent's own engine then does the beam search, the geometric mean ranking and the final cut, exactly as it
does with a live model. The agent supplies judgment, the engine supplies arithmetic that cannot be talked into
a better ranking.

    {
      "state": "what is true today",
      "nodes": [
        {"id": "a", "parent": null, "text": "one year later...", "plausibility": 0.6, "value": 0.8,
         "reversibility": 0.7, "risk": 0.3},
        {"id": "a1", "parent": "a", "text": "two years later...", "plausibility": 0.5, "value": 0.8,
         "reversibility": 0.8, "risk": 0.2, "summary": "name of this future",
         "assumptions": ["..."], "moves_now": ["..."]}
      ]
    }

A node with "parent": null branches from today. Depth decides the year. Texts must be unique.
"""
from __future__ import annotations

import json
import re
from datetime import date

from .engine import Engine, Result

AXES = ("plausibility", "value", "reversibility", "risk")

TEMPLATE = {
    "state": "Describe today in two or three plain sentences: who, what, the decision on the table.",
    "nodes": [
        {"id": "a", "parent": None, "text": "A good outcome one year later, concrete, with numbers.",
         "plausibility": 0.6, "value": 0.8, "reversibility": 0.7, "risk": 0.3},
        {"id": "b", "parent": None, "text": "A neutral outcome one year later.",
         "plausibility": 0.5, "value": 0.5, "reversibility": 0.7, "risk": 0.3},
        {"id": "c", "parent": None, "text": "A bad outcome one year later, and why it went wrong.",
         "plausibility": 0.4, "value": 0.2, "reversibility": 0.4, "risk": 0.7},
        {"id": "a1", "parent": "a", "text": "What follows from a, two years later.",
         "plausibility": 0.5, "value": 0.8, "reversibility": 0.8, "risk": 0.2,
         "summary": "One sentence naming this future",
         "assumptions": ["what must hold", "what must hold", "what must hold"],
         "moves_now": ["action this month", "action this month", "action this month"]},
    ],
}


def check_tree(tree: dict) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors stop a run. Warnings point at a tree that flatters itself."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(tree, dict) or not str(tree.get("state", "")).strip():
        errors.append("the tree needs a non empty 'state'")
    nodes = tree.get("nodes") if isinstance(tree, dict) else None
    if not isinstance(nodes, list) or not nodes:
        return errors + ["the tree needs a non empty 'nodes' list"], warnings
    ids, texts = set(), set()
    for n in nodes:
        nid, text = n.get("id"), str(n.get("text", "")).strip()
        if not nid or nid in ids:
            errors.append(f"node id missing or repeated: {nid!r}")
        ids.add(nid)
        if not text or text in texts:
            errors.append(f"node {nid!r}: text missing or repeated (texts must be unique)")
        texts.add(text)
        for axis in AXES:
            v = n.get(axis)
            if not isinstance(v, (int, float)) or not 0 <= v <= 1:
                errors.append(f"node {nid!r}: {axis} must be a number from 0 to 1")
    for n in nodes:
        if n.get("parent") is not None and n.get("parent") not in ids:
            errors.append(f"node {n.get('id')!r}: parent {n.get('parent')!r} does not exist")
    if errors:
        return errors, warnings
    kids: dict = {}
    for n in nodes:
        kids.setdefault(n.get("parent"), []).append(n)
    for parent, group in kids.items():
        where = "today" if parent is None else f"node {parent!r}"
        if len(group) < 2:
            warnings.append(f"{where} has only one branch. A forecast needs alternatives.")
        if all(n["value"] >= 0.4 for n in group):
            warnings.append(f"{where} has no bad branch (value under 0.4). Add one, or the ranking only compares hopes.")
    return errors, warnings


class TreeModel:
    """Answers the engine's three prompts from a tree an agent wrote. No network, no randomness."""

    def __init__(self, tree: dict):
        self.tree = tree
        self.by_text = {str(n["text"]).strip(): n for n in tree["nodes"]}
        by_id = {n["id"]: n for n in tree["nodes"]}
        self.children: dict = {}
        for n in tree["nodes"]:
            key = None if n.get("parent") is None else str(by_id[n["parent"]]["text"]).strip()
            self.children.setdefault(key, []).append(str(n["text"]).strip())

    def complete(self, prompt: str) -> str:
        if "Rate this future state" in prompt:
            m = re.search(r"State in year \d+: (.*?)\nIt followed from", prompt, re.DOTALL)
            n = self.by_text.get(m.group(1).strip()) if m else None
            return json.dumps({a: n[a] for a in AXES}) if n else "{}"
        if prompt.startswith("Here is one future"):
            block = prompt.split("Path:\n", 1)[1].split("\n\nWrite JSON", 1)[0].strip()
            last = block.splitlines()[-1].split(": ", 1)[-1].strip()
            n = self.by_text.get(last, {})
            return json.dumps({"summary": n.get("summary", ""), "assumptions": n.get("assumptions", []),
                               "moves_now": n.get("moves_now", [])})
        chain = prompt.split("The chain of events so far:\n", 1)[-1].split("\n\nWrite", 1)[0].strip()
        parent = None if chain == "(none yet)" else chain.splitlines()[-1].split(": ", 1)[-1].strip()
        return json.dumps(self.children.get(parent, []))


def depth(tree: dict) -> int:
    by_id = {n["id"]: n for n in tree["nodes"]}

    def d(n):
        return 1 if n.get("parent") is None else 1 + d(by_id[n["parent"]])

    return max(d(n) for n in tree["nodes"])


def run_tree(tree: dict, keep: int = 8, beam: int = 8, today: date | None = None) -> Result:
    errors, _ = check_tree(tree)
    if errors:
        raise ValueError("; ".join(errors))
    model = TreeModel(tree)
    branches = max(len(v) for v in model.children.values())
    return Engine(model, branches=branches, beam=beam, today=today).run(str(tree["state"]).strip(), years=depth(tree), keep=keep)
