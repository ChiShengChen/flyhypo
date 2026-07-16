"""Evaluation harness: grade the tool's output against a small gold set.

Not a substitute for expert review — it is a *coarse, deterministic* regression
signal so changes to prompts/verification can be compared objectively. It reads
`eval/gold/*.json` (curated known biology) and the tool's `outputs/<type>.json`
(produced by a prior `flyhypo <type>` run), then scores, per type:

  - function recall    — gold functions matched by a produced role/hypothesis
                         (keyword overlap; approximate by construction)
  - function precision — produced roles that map to some gold function
  - partner coverage   — gold key partner families present in the fingerprint
  - citations valid    — every cited id is a well-formed, present id

    uv run flyhypo-eval                 # all gold types with an outputs/ file
    uv run flyhypo-eval EPG MBON01

Run `flyhypo <type>` first to generate outputs/<type>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GOLD_DIR = Path("eval/gold")
OUT_DIR = Path("outputs")


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "celltype"


_CONF_RANK = {"speculative": 0, "low": 1, "medium": 2, "high": 3}


def _matches(text: str, keywords: list[str], need: int = 1) -> bool:
    t = text.lower()
    return sum(1 for k in keywords if k.lower() in t) >= need


def score_negative(gold: dict, report: dict) -> dict:
    """For obscure / invalid types: check graceful degradation rather than recall.

    Confirms the tool stays honest — capped confidence, no citations it can't
    ground, and (for invalid types) not-found with suggestions and no structure."""
    fp = report.get("fingerprint", {})
    exp = gold.get("expect", {})
    roles, hyps = report.get("functional_roles", []), report.get("hypotheses", [])
    confs = [x.get("confidence", "speculative") for x in roles + hyps]

    lit_ids = {h.get("id") for h in report.get("literature", [])}
    cited = [c for x in roles for c in x.get("references", [])] + \
            [c for h in hyps for c in h.get("supporting_literature", [])]
    fabricated = [c for c in cited if c and c != "n/a" and c not in lit_ids]

    checks: dict[str, bool] = {}
    if "found" in exp:
        checks["found"] = bool(fp.get("resolved")) == exp["found"]
    if exp.get("suggestions"):
        checks["suggest"] = bool(fp.get("suggestions"))
    if exp.get("no_fabricated_citations"):
        checks["no_fab"] = not fabricated
    if "max_confidence" in exp:
        cap = _CONF_RANK[exp["max_confidence"]]
        checks["max_conf"] = all(_CONF_RANK.get(c, 0) <= cap for c in confs)
    return {"type": gold["type"], "checks": checks, "ok": all(checks.values()),
            "fabricated": fabricated}


def score_type(gold: dict, report: dict) -> dict:
    roles = report.get("functional_roles", [])
    hyps = report.get("hypotheses", [])
    produced_texts = [r.get("function", "") for r in roles] + \
                     [h.get("statement", "") for h in hyps]

    gold_funcs = gold.get("known_functions", [])
    matched = [g for g in gold_funcs
               if any(_matches(pt, g["keywords"]) for pt in produced_texts)]
    recall = round(len(matched) / len(gold_funcs), 2) if gold_funcs else 0.0

    role_hits = [r for r in roles
                 if any(_matches(r.get("function", ""), g["keywords"]) for g in gold_funcs)]
    precision = round(len(role_hits) / len(roles), 2) if roles else 0.0

    fp = report.get("fingerprint", {})
    partners = {p.get("type", "") for p in (fp.get("upstream", []) + fp.get("downstream", []))}
    key = gold.get("key_partners", [])
    covered = [k for k in key if any(k.lower() in (pt or "").lower() for pt in partners)]
    partner_cov = round(len(covered) / len(key), 2) if key else 0.0

    all_refs = {rf for r in roles for rf in r.get("references", [])} | \
               {rf for h in hyps for rf in h.get("supporting_literature", [])}
    cites_ok = all(rf and rf != "n/a" for rf in all_refs)

    # NT check (catches wrong-field / mis-attributed predictions the keyword
    # scoring can't): compare the fingerprint's predicted NT to the gold value.
    exp_nt = (gold.get("expected_nt") or "").lower()
    got_nt = (fp.get("predicted_nt") or "").lower()
    nt_ok = None if not exp_nt else (got_nt == exp_nt)

    return {
        "type": gold["type"], "recall": recall, "precision": precision,
        "n_gold": len(gold_funcs), "n_matched": len(matched), "n_roles": len(roles),
        "partner_cov": partner_cov,
        "missing": [g["name"] for g in gold_funcs if g not in matched],
        "cites_ok": cites_ok, "nt_ok": nt_ok, "got_nt": got_nt or "?", "exp_nt": exp_nt,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    golds = {}
    for p in sorted(GOLD_DIR.glob("*.json")):
        g = json.loads(p.read_text())
        golds[g["type"]] = g
    wanted = argv or list(golds)

    rows, negs = [], []
    for t in wanted:
        gold = golds.get(t)
        if not gold:
            print(f"! no gold for {t} (have: {', '.join(golds) or 'none'})")
            continue
        rpath = OUT_DIR / f"{_slug(t)}.json"
        if not rpath.exists():
            print(f"! no outputs/{_slug(t)}.json — run `flyhypo {t}` first")
            continue
        report = json.loads(rpath.read_text())
        (negs if gold.get("negative") else rows).append(
            (score_negative if gold.get("negative") else score_type)(gold, report))

    if not rows and not negs:
        print("nothing scored.")
        return 1

    def _nt(r):
        return "—" if r["nt_ok"] is None else ("ok" if r["nt_ok"] else f"BAD:{r['got_nt']}")

    if rows:
        print(f"\n{'type':<10} {'recall':>7} {'prec':>6} {'partners':>9} {'cites':>6} {'nt':>10}  missing")
        print("-" * 84)
        for r in rows:
            print(f"{r['type']:<10} {r['recall']:>7.2f} {r['precision']:>6.2f} "
                  f"{r['partner_cov']:>9.2f} {'ok' if r['cites_ok'] else 'BAD':>6} {_nt(r):>10}  "
                  f"{'; '.join(r['missing']) or '—'}")
        n = len(rows)
        print("-" * 72)
        print(f"{'MEAN':<10} {sum(r['recall'] for r in rows)/n:>7.2f} "
              f"{sum(r['precision'] for r in rows)/n:>6.2f} "
              f"{sum(r['partner_cov'] for r in rows)/n:>9.2f}")

    if negs:
        keys = ["found", "suggest", "no_fab", "max_conf"]
        print(f"\ngraceful degradation (obscure / invalid types)")
        print(f"{'type':<10} " + " ".join(f"{k:>9}" for k in keys) + f" {'OK':>6}")
        print("-" * 60)
        for r in negs:
            cells = " ".join(
                ("—" if k not in r["checks"] else ("ok" if r["checks"][k] else "FAIL")).rjust(9)
                for k in keys)
            print(f"{r['type']:<10} {cells} {'PASS' if r['ok'] else 'FAIL':>6}")

    print("\n(coarse keyword scoring — a regression signal, not expert review.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
