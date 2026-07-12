"""Independent verification of generated claims — the anti-hallucination layer.

Replaces the same-model self-verify pass with the `paper-evidence` stack, so a
literature-grounded claim is checked by machinery and a *different* model family, not by
Gemini grading itself:

  * verbatim re-grep      — the claim's `quote` must be found in the cited abstract
  * mis-attribution guard — the focus cell type must be NAMED in that quote (focus_named)
  * cross-family judge    — a DeepSeek judge (≠ Gemini) confirms the quote supports the claim
  * citation reality      — every cited PMID/DOI/arXiv id must resolve and not be RETRACTED
  * evidence grading      — Supported/Tension/Conflict/Unknown + a confidence tier; literature
                            is abstract-only, so confidence caps at 'medium' by construction.

A claim whose quote is fabricated / mis-attributed / unsupported is DROPPED; a cited id that
doesn't resolve (or is retracted) is stripped. Confidence is re-tiered from the deterministic
grade, not the model's opinion. Connectivity-only claims are passed through (their grounding is
the fingerprint number, verified elsewhere).

Requires `paper-evidence` (https://github.com/ChiShengChen/paper-evidence).
"""
from __future__ import annotations

import re
from typing import Any

from paper_evidence import citation, evidence_state, quote_gate

from .schema import Confidence, HypothesisAnalysis, LiteratureHit

_ARXIV = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
# evidence_state tier -> flyhypo Confidence (abstract-only caps at 'medium')
_TIER = {"High": "high", "Med": "medium", "Low": "low", "Unknown": "speculative"}


def default_judge() -> Any:
    """A cross-family faithfulness judge (a provider != Gemini, e.g. DeepSeek) or None.
    None -> verbatim + citation + grading still run, just without the LLM support check."""
    return quote_gate.make_judge(avoid="gemini")


def _cite_args(cite_id: str, title: str | None) -> dict[str, Any]:
    """Route a citation id (PMID / DOI / arXiv) to verify_citation kwargs."""
    s = str(cite_id).strip()
    if s.isdigit():
        return {"pmid": s, "title": title}
    if _ARXIV.match(s):
        return {"arxiv": s, "title": title}
    m = re.search(r"10\.\d{4,}/\S+", s)
    if m:
        return {"doi": m.group(0), "title": title}
    return {"title": title}


def verify_claim(*, claim_text: str, quote: str, references: list[str],
                 evidence_type: str, subject_names: list[str],
                 lit_by_id: dict[str, LiteratureHit], judge: Any = None,
                 check_citations: bool = True) -> dict[str, Any]:
    """Verify one literature-grounded claim. Returns a verdict dict (does not mutate)."""
    reasons: list[str] = []
    src = " ".join((lit_by_id[i].snippet or "") for i in references if i in lit_by_id)
    is_lit = evidence_type in ("literature", "both")

    verbatim_ok: bool | None = None
    subject_named: bool | None = None
    faithful: bool | None = None
    if is_lit and quote:
        if not src:
            verbatim_ok = False
            reasons.append("cited paper(s) not in retrieved evidence — quote unverifiable")
        elif not quote_gate.verify_quote(quote, src)["ok"]:
            verbatim_ok = False
            reasons.append("quote not found in cited abstract(s)")
        else:
            verbatim_ok = True
            subject_named = quote_gate.focus_named(subject_names, quote)
            if not subject_named:
                reasons.append("focus cell type not named in the quote (mis-attribution)")
            elif judge is not None:
                fj = quote_gate.judge_claim_support(judge, claim_text, quote)
                faithful = fj["supported"] if fj["judged"] else None
                if faithful is False:
                    reasons.append("quote does not support the claim (cross-family judge)")
    elif is_lit and not quote:
        reasons.append("no verbatim quote provided for a literature claim")

    citations = []
    if check_citations:
        for cid in references:
            hit = lit_by_id.get(cid)
            cr = citation.verify_citation(**_cite_args(cid, hit.title if hit else None))
            citations.append({"id": cid, "status": cr["status"], "retracted": cr["retracted"]})
    bad_ids = [c["id"] for c in citations if c["status"] != "verified"]
    if bad_ids:
        reasons.append("unresolved/retracted citation(s): " + ", ".join(bad_ids))

    # deterministic grade (abstract-only for literature; None quote -> silence -> Unknown)
    if is_lit:
        state = evidence_state.classify(verbatim_ok=bool(verbatim_ok), faithful=faithful,
                                        subject_named=subject_named, abstract_only=True)
    else:
        state = {"state": "Supported", "confidence": "Med", "overclaim_risk": False,
                 "reasons": ["connectivity-grounded (not literature-verified here)"]}

    # drop literature claims whose quote is missing / fabricated / mis-attributed / unsupported
    drop = is_lit and (verbatim_ok is False or subject_named is False or faithful is False
                       or (quote == "" and not references))
    return {"drop": drop, "quote_ok": verbatim_ok, "subject_named": subject_named,
            "faithful": faithful, "citations": citations, "bad_ids": bad_ids,
            "evidence_state": state, "confidence": _TIER.get(state["confidence"], "speculative"),
            "reasons": reasons}


def verify_analysis(analysis: HypothesisAnalysis, lit: list[LiteratureHit],
                    subject_names: list[str], *, judge: Any = None,
                    check_citations: bool = True) -> dict[str, Any]:
    """Verify every role + hypothesis IN PLACE: drop fabricated/mis-attributed claims, strip
    bad citations, and re-tier confidence from the deterministic grade. Returns a summary."""
    lit_by_id = {h.id: h for h in lit}
    dropped: list[str] = []
    verdicts: list[dict] = []

    def _process(items, label, claim_of, refs_of, set_refs, set_conf):
        keep = []
        for j, it in enumerate(items, 1):
            v = verify_claim(claim_text=claim_of(it), quote=getattr(it, "quote", "") or "",
                             references=refs_of(it), evidence_type=getattr(it, "evidence_type", "literature"),
                             subject_names=subject_names, lit_by_id=lit_by_id, judge=judge,
                             check_citations=check_citations)
            verdicts.append({"kind": label, "index": j, **{k: v[k] for k in
                             ("drop", "quote_ok", "subject_named", "faithful", "reasons")}})
            if v["drop"]:
                dropped.append(f"{label}{j} ({'; '.join(v['reasons']) or 'unverified'})")
                continue
            set_refs(it, [i for i in refs_of(it) if i not in v["bad_ids"]])   # strip bad cites
            set_conf(it, v["confidence"])                                     # deterministic tier
            keep.append(it)
        return keep

    analysis.functional_roles = _process(
        analysis.functional_roles, "role", lambda r: r.function, lambda r: r.references,
        lambda r, x: setattr(r, "references", x), lambda r, c: setattr(r, "confidence", c))
    analysis.hypotheses = _process(
        analysis.hypotheses, "H", lambda h: h.statement, lambda h: h.supporting_literature,
        lambda h, x: setattr(h, "supporting_literature", x), lambda h, c: setattr(h, "confidence", c))

    notes = ("[verify] paper-evidence: verbatim + numbers-in-context + mis-attribution + "
             f"{'cross-family judge + ' if judge is not None else ''}citation/retraction + "
             "abstract-only grading.")
    if dropped:
        notes += f"\n[verify] dropped {len(dropped)} unverified claim(s): " + "; ".join(dropped)
    return {"notes": notes, "dropped": dropped, "verdicts": verdicts}
