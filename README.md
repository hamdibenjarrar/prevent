# Prevent

**The layer after the model.** Give Prevent any output, a plan, a draft, a decision, and it runs it
years forward, branch by branch, scores every path, and hands back the best eight futures with the
moves that keep each one reachable from today.

[![ci](https://github.com/hamdibenjarrar/prevent/actions/workflows/ci.yml/badge.svg)](https://github.com/hamdibenjarrar/prevent/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![model agnostic](https://img.shields.io/badge/model-agnostic-orange)](#models)
[![stars](https://img.shields.io/github/stars/hamdibenjarrar/prevent?style=flat)](https://github.com/hamdibenjarrar/prevent/stargazers)

Prevent is not a prophet. A future is a scored scenario with its assumptions written down. What it
gives you is the thing most agents skip: before you act on an output, you see where it leads.

## 10 second start

```bash
pip install git+https://github.com/hamdibenjarrar/prevent
ollama pull qwen2.5:7b   # or any local model, Prevent does not care which
prevent "I launch a one person AI studio in Tunis with no funding" --years 8 --out futures
```

`futures.md` reads like this:

```
## 1. A small studio that pays for itself from year two (score 0.71, year 2034)
Path:
- 2027: first three retainer clients from the Libyan NGO network, 1 800 USD a month
- 2028: ...
Assumptions:
- the retainers renew at least twice
Moves now:
- cut the offer to three fixed price packages
- talk to 10 past clients this month
```

## How it works

1. **Branch.** For each kept state, the model writes N concrete things that could be true one year
   later (good, neutral and bad; numbers and names, no vagueness).
2. **Score.** Each branch gets four numbers from 0 to 1: plausibility, value, reversibility, risk.
   One line of arithmetic turns them into a score, written so anyone can read why a future ranked
   where it did (`prevent/engine.py`, `score_node`).
3. **Beam.** The best K survive the year. A trajectory's score is the geometric mean of its years,
   so one implausible year sinks a whole future.
4. **Lock.** For the final eight, the model names the future, lists what must hold, and writes the
   three moves to make this month that keep it reachable.

`Engine(model, branches=4, beam=8).run(state, years=8, keep=8)` is the whole API.

## Models

Anything with `complete(prompt) -> str` is a model. Built in:

| spec | door |
|---|---|
| `ollama:qwen2.5:7b` | local Ollama (default). Any model you have pulled. |
| `openai:gpt-4o-mini` | OpenAI, key from `OPENAI_API_KEY` |
| `groq:llama-3.3-70b-versatile` | Groq, key from `GROQ_API_KEY` |
| `openrouter:MODEL` | OpenRouter, key from `OPENROUTER_API_KEY` |
| `url:MODEL@https://host/v1` | any OpenAI compatible endpoint |

```python
from prevent import Engine, OllamaModel

res = Engine(OllamaModel("qwen2.5:7b")).run("my plan", years=8)
for f in res.futures:
    print(f.rank, f.score, f.summary, f.moves_now)
```

Keys are read from the environment only and never printed.

## Use it inside an agent

Prevent was built as a layer for Alpha, a personal AI operating system (private for now). Alpha runs Prevent on its own plans before it ships a week of work, and the best
future becomes the plan. Drop it after any model call the same way:

```python
plan = your_model(prompt)
futures = Engine(your_model_adapter).run(plan, years=8, keep=8)
best = futures.futures[0]
act_on(best.moves_now)
```

## Let an agent be the model

New in 0.2.0. An agent such as Claude, Alpha or a human analyst writes every branch and every score into one
JSON file. Prevent's engine then does the beam search, the geometric mean ranking and the final cut, exactly as
it does with a live model. The agent brings judgment. The engine brings arithmetic that cannot be talked into a
better ranking. No model download, no key, no network.

```bash
prevent --template > tree.json      # a starting tree
prevent --tree tree.json --keep 3   # rank it
```

```json
{
  "state": "What is true today, and the decision on the table.",
  "nodes": [
    {"id": "a", "parent": null, "text": "A good outcome one year later.", "plausibility": 0.6, "value": 0.8, "reversibility": 0.7, "risk": 0.3},
    {"id": "c", "parent": null, "text": "A bad outcome one year later.", "plausibility": 0.4, "value": 0.2, "reversibility": 0.4, "risk": 0.7},
    {"id": "a1", "parent": "a", "text": "What follows from a, two years later.", "plausibility": 0.5, "value": 0.8, "reversibility": 0.8, "risk": 0.2,
     "summary": "One sentence naming this future", "assumptions": ["what must hold"], "moves_now": ["action this month"]}
  ]
}
```

`check_tree` refuses broken trees and warns when a tree flatters itself: a parent with only one branch, or a
set of branches with no bad outcome. Write the scores before you look at any ranking, or the ranking is just
your hopes read back to you.

```python
from prevent import run_tree
res = run_tree(tree, keep=3)
print(res.to_markdown())
```

## Honest scope

- It costs model calls: branches x beam x years, plus scoring. With the defaults and a local 7B model
  a run is a few minutes. Lower `--branches` and `--beam` for a quick look.
- The scores are the model's judgement, made legible. Treat them as a second opinion, not a verdict.
- No network calls except to the model you chose.

## Roadmap

- Pluggable scorers (a rule based one for finance, one for product launches)
- Diff two runs: what changed in the futures when you change one move
- HTML report

MIT. Built by Hamdi Ben Jarrar, MYNTHISM, Tunis.
