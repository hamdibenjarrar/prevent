"""prevent "your state" --model ollama:qwen2.5:7b --years 8 --keep 8 --out futures
prevent --tree tree.json --out futures        an agent wrote the branches and scores
prevent --template > tree.json                start a tree"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import Engine
from .models import from_spec
from .tree import TEMPLATE, check_tree, run_tree


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prevent", description="Run a state years forward and keep the best futures.")
    p.add_argument("state", nargs="?", help="The plan, draft or decision to run forward. Use - to read stdin.")
    p.add_argument("--model", default="ollama:qwen2.5:7b", help="ollama:NAME, openai:NAME, groq:NAME, openrouter:NAME or url:NAME@BASE")
    p.add_argument("--tree", help="A JSON tree written by an agent. The agent is the model; the engine ranks.")
    p.add_argument("--template", action="store_true", help="Print a starting tree and exit.")
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--keep", type=int, default=8)
    p.add_argument("--branches", type=int, default=4)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--out", default=None, help="Write OUT.json and OUT.md")
    a = p.parse_args(argv)

    if a.template:
        print(json.dumps(TEMPLATE, indent=2))
        return 0
    if a.tree:
        tree = json.loads(Path(a.tree).read_text(encoding="utf-8"))
        errors, warnings = check_tree(tree)
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
        if errors:
            for e in errors:
                print(f"error: {e}", file=sys.stderr)
            return 2
        res = run_tree(tree, keep=a.keep, beam=a.beam)
    else:
        if not a.state:
            p.error("give a state, or --tree FILE, or --template")
        state = sys.stdin.read().strip() if a.state == "-" else a.state
        res = Engine(from_spec(a.model), branches=a.branches, beam=a.beam).run(state, years=a.years, keep=a.keep)

    if a.out:
        Path(a.out + ".json").write_text(res.to_json(), encoding="utf-8")
        Path(a.out + ".md").write_text(res.to_markdown(), encoding="utf-8")
        print(f"wrote {a.out}.json and {a.out}.md ({res.explored} branches explored)")
    else:
        print(res.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
