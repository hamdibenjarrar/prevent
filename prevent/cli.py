"""prevent "your state" --model ollama:qwen2.5:7b --years 8 --keep 8 --out futures"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import Engine
from .models import from_spec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prevent", description="Run a state 8 years forward and keep the best 8 futures.")
    p.add_argument("state", help="The plan, draft or decision to run forward. Use - to read stdin.")
    p.add_argument("--model", default="ollama:qwen2.5:7b", help="ollama:NAME, openai:NAME, groq:NAME, openrouter:NAME or url:NAME@BASE")
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--keep", type=int, default=8)
    p.add_argument("--branches", type=int, default=4)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--out", default=None, help="Write OUT.json and OUT.md")
    a = p.parse_args(argv)
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
