"""Prevent: the layer after the model.

Take a state of the world (a plan, a draft, a decision, any model's output), branch it forward
year by year, score every branch, and keep the best eight futures. Each future carries the path
that leads to it, the assumptions it rests on, and the moves that keep it reachable from today.

Prevent is model agnostic. Anything with a ``complete(prompt) -> str`` method is a model: an
Ollama model, an OpenAI compatible endpoint, Alpha's own router, or a fixed script for tests.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Protocol


class Model(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class Node:
    year: int
    text: str
    score: float = 0.0
    plausibility: float = 0.5
    value: float = 0.5
    reversibility: float = 0.5
    risk: float = 0.5
    parent: int | None = None
    depth: int = 0


@dataclass
class Future:
    rank: int
    score: float
    year: int
    summary: str
    path: list[str]
    assumptions: list[str]
    moves_now: list[str]
    scores: dict = field(default_factory=dict)


@dataclass
class Result:
    state: str
    horizon_years: int
    generated_on: str
    futures: list[Future]
    explored: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        out = [f"# Prevent: {self.horizon_years} years from {self.generated_on}", "", f"State: {self.state}", "",
               f"Explored {self.explored} branches, kept {len(self.futures)}.", ""]
        for f in self.futures:
            out += [f"## {f.rank}. {f.summary} (score {f.score:.2f}, year {f.year})", ""]
            out += ["Path:"] + [f"- {p}" for p in f.path] + [""]
            out += ["Assumptions:"] + [f"- {a}" for a in f.assumptions] + [""]
            out += ["Moves now:"] + [f"- {m}" for m in f.moves_now] + [""]
        return "\n".join(out)


BRANCH_PROMPT = """You are a careful forecaster. Today is {today}. The current state is:

{state}

The chain of events so far:
{path}

Write {n} different, concrete things that could be true one year later, in year {year}.
Each must follow from the chain above. Mix good, neutral and bad outcomes. Be specific:
numbers, names of what was built, what failed and why. No prophecy, no vagueness.
Reply with a JSON list of {n} strings and nothing else."""

SCORE_PROMPT = """Rate this future state on four axes from 0.0 to 1.0.

State in year {year}: {text}
It followed from: {parent}

plausibility: how likely it is given the chain.
value: how good it is for the person or system in the state.
reversibility: how easy it is to recover if it goes wrong.
risk: how much could be lost (1.0 means everything).

Reply with JSON only: {{"plausibility": x, "value": x, "reversibility": x, "risk": x}}"""

LOCK_PROMPT = """Here is one future and the path that leads to it.

Path:
{path}

Write JSON only: {{"summary": one sentence naming this future, "assumptions": [3 things that must
hold for it to happen], "moves_now": [3 concrete actions to take this month that keep it reachable]}}"""


def _json(text: str, default):
    text = text.strip()
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return default


def _clamp(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.5


def score_node(n: Node) -> float:
    """One number for ranking. Value matters most, plausibility second, risk subtracts, reversibility
    softens risk. Kept simple on purpose so anyone can read why a future ranked where it did."""
    return round(0.45 * n.value + 0.35 * n.plausibility + 0.20 * n.reversibility - 0.25 * n.risk * (1 - n.reversibility), 4)


class Engine:
    def __init__(self, model: Model, branches: int = 4, beam: int = 8, today: date | None = None):
        self.model = model
        self.branches = max(2, branches)
        self.beam = max(1, beam)
        self.today = today or date.today()

    def run(self, state: str, years: int = 8, keep: int = 8) -> Result:
        """Beam search over years. Each year every kept node branches ``branches`` ways, every branch
        is scored, the best ``beam`` survive. Returns the ``keep`` best full trajectories."""
        nodes: list[Node] = [Node(year=self.today.year, text=state, score=1.0, depth=0)]
        frontier = [0]
        explored = 0
        for step in range(1, years + 1):
            year = self.today.year + step
            new_ids: list[int] = []
            for pid in frontier:
                parent = nodes[pid]
                for text in self._branch(state, self._path(nodes, pid), year):
                    n = self._score(Node(year=year, text=text, parent=pid, depth=step), parent)
                    nodes.append(n)
                    new_ids.append(len(nodes) - 1)
                    explored += 1
            if not new_ids:
                break
            new_ids.sort(key=lambda i: self._trajectory_score(nodes, i), reverse=True)
            frontier = new_ids[: self.beam]
        frontier = [i for i in frontier if nodes[i].parent is not None]  # the root is not a future
        frontier.sort(key=lambda i: self._trajectory_score(nodes, i), reverse=True)
        futures = []
        for rank, i in enumerate(frontier[:keep], 1):
            path = self._path(nodes, i)
            lock = _json(self.model.complete(LOCK_PROMPT.format(path="\n".join(path))), {})
            n = nodes[i]
            futures.append(Future(
                rank=rank, score=round(self._trajectory_score(nodes, i), 4), year=n.year,
                summary=str(lock.get("summary") or n.text[:140]),
                path=path, assumptions=[str(a) for a in lock.get("assumptions", [])][:5],
                moves_now=[str(m) for m in lock.get("moves_now", [])][:5],
                scores={"plausibility": n.plausibility, "value": n.value, "reversibility": n.reversibility, "risk": n.risk},
            ))
        return Result(state=state, horizon_years=years, generated_on=self.today.isoformat(), futures=futures, explored=explored)

    def _branch(self, state: str, path: list[str], year: int) -> list[str]:
        raw = self.model.complete(BRANCH_PROMPT.format(today=self.today.isoformat(), state=state, path="\n".join(path) or "(none yet)", n=self.branches, year=year))
        items = _json(raw, [])
        out = [str(x).strip() for x in items if str(x).strip()] if isinstance(items, list) else []
        return out[: self.branches]

    def _score(self, n: Node, parent: Node) -> Node:
        s = _json(self.model.complete(SCORE_PROMPT.format(year=n.year, text=n.text, parent=parent.text)), {})
        n.plausibility, n.value = _clamp(s.get("plausibility", 0.5)), _clamp(s.get("value", 0.5))
        n.reversibility, n.risk = _clamp(s.get("reversibility", 0.5)), _clamp(s.get("risk", 0.5))
        n.score = score_node(n)
        return n

    @staticmethod
    def _path(nodes: list[Node], i: int) -> list[str]:
        out = []
        while i is not None:
            n = nodes[i]
            out.append(f"{n.year}: {n.text}")
            i = n.parent
        return list(reversed(out))[1:]  # drop the root state, it is printed separately

    @staticmethod
    def _trajectory_score(nodes: list[Node], i: int) -> float:
        """Geometric mean of the scores along the path: one implausible year sinks the whole future."""
        vals, cur = [], i
        while cur is not None and nodes[cur].parent is not None:
            vals.append(max(1e-6, nodes[cur].score))
            cur = nodes[cur].parent
        return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
