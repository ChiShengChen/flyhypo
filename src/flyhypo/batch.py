"""Batch generation: run the full pipeline over many types → outputs/<type>.{json,md}.

Feeds the evaluation harness (flyhypo-eval). One Gemini call per type.

    uv run flyhypo-batch EPG MBON01 KCg-m
    uv run flyhypo-batch                # all types in eval/gold/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .cli import _slug, render_markdown


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    types = list(argv)
    if not types:
        gold = Path("eval/gold")
        types = [json.loads(p.read_text())["type"] for p in sorted(gold.glob("*.json"))]
    if not types:
        print("no types given and no eval/gold/*.json found")
        return 1

    from . import connectome, literature, synthesize

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for t in types:
        print(f"[batch] {t} …", flush=True)
        try:
            fp = connectome.build_fingerprint(t)
            if not fp.found:
                # Don't skip — run the degraded pipeline (graceful low-confidence
                # output with suggestions), so negative/obscure types are evaluable.
                print(f"  not found → degraded run (suggestions: {', '.join(fp.suggestions[:4])})")
            lit = literature.fetch_literature(fp)
            result = synthesize.synthesize(fp, lit)
            slug = _slug(t)
            (out / f"{slug}.json").write_text(result.model_dump_json(indent=2, by_alias=True))
            (out / f"{slug}.md").write_text(render_markdown(result))
            print(f"  wrote outputs/{slug}.json — {len(result.functional_roles)} roles, "
                  f"{len(result.hypotheses)} hypotheses; NT={fp.predicted_nt or '?'}"
                  f"{' ('+fp.predicted_nt_source+')' if fp.predicted_nt_source else ''}")
            ok += 1
        except Exception as e:  # keep going through the batch
            print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"\n[batch] {ok}/{len(types)} generated. Now: uv run flyhypo-eval")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
