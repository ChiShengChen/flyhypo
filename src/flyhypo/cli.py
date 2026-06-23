"""flyhypo CLI: structure + literature → grounded functional hypothesis.

    flyhypo <cell_type> [--dataset hemibrain:v1.2.1] [--top-k 15] [--out outputs/]

Writes outputs/<cell_type>.json and outputs/<cell_type>.md and prints a summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .schema import Hypothesis, StructuralFingerprint


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "celltype"


def render_markdown(h: Hypothesis) -> str:
    fp = h.fingerprint
    L: list[str] = []
    L.append(f"# Functional hypothesis: `{h.cell_type}`")
    L.append("")
    L.append(f"**Dataset:** {h.dataset}  ")
    L.append(
        "**Status:** "
        + (
            f"resolved {len(fp.resolved)} cell(s)"
            if fp.found
            else "⚠️ type NOT found — degraded, low-confidence output"
        )
    )
    L.append("")
    L.append(
        "> A connectome gives connectivity, not synapse sign, effective strength, "
        "or neuromodulation. Everything below is a **hypothesis for "
        "experimentalists**, not a stated fact."
    )
    L.append("")

    if not fp.found and fp.suggestions:
        L.append(f"**Did you mean:** {', '.join(fp.suggestions)}")
        L.append("")

    # --- fingerprint ---------------------------------------------------- #
    L.append("## Structural fingerprint")
    L.append(
        "> **Reading the numbers:** ROI weights are **synaptic *site* counts** "
        "(pre/post sites summed over the cells); partner `w` is the **pairwise "
        "synapse count** (synapses between neuron pairs, summed over the type). "
        "Both are structural proxies — not functional strength."
    )
    L.append("")
    L.append(f"- **Predicted neurotransmitter:** {fp.predicted_nt or 'unknown'}")
    if fp.input_rois:
        rois = ", ".join(f"{r.roi} ({r.weight})" for r in fp.input_rois)
        L.append(f"- **Top input ROIs (postsynaptic sites):** {rois}")
    if fp.output_rois:
        rois = ", ".join(f"{r.roi} ({r.weight})" for r in fp.output_rois)
        L.append(f"- **Top output ROIs (presynaptic sites):** {rois}")
    if fp.upstream:
        L.append("")
        L.append("**Top upstream partners** "
                 "(type — n cells — w = pairwise synapse count — NT — class):")
        for p in fp.upstream:
            L.append(
                f"- {p.type or '?'} — n={p.n_cells} — w={p.total_weight} — "
                f"{p.predicted_nt or '?'} — {p.neuron_class or '?'}"
            )
    if fp.downstream:
        L.append("")
        L.append("**Top downstream partners** "
                 "(type — n cells — w = pairwise synapse count — NT — class):")
        for p in fp.downstream:
            L.append(
                f"- {p.type or '?'} — n={p.n_cells} — w={p.total_weight} — "
                f"{p.predicted_nt or '?'} — {p.neuron_class or '?'}"
            )
    L.append("")

    # --- literature ----------------------------------------------------- #
    L.append("## Literature used")
    if h.literature:
        for hit in h.literature:
            yr = f" ({hit.year})" if hit.year else ""
            L.append(f"- **[{hit.source}:{hit.id}]**{yr} {hit.title}  ")
            L.append(f"  _why:_ {hit.relevance}")
    else:
        L.append("- _No literature retrieved._")
    L.append("")

    # --- hypotheses ----------------------------------------------------- #
    L.append("## Hypotheses (tiered)")
    for i, hyp in enumerate(h.hypotheses, 1):
        L.append(f"### H{i} — confidence: **{hyp.confidence}**")
        L.append(f"{hyp.statement}")
        L.append("")
        L.append(f"_Rationale:_ {hyp.rationale}")
        if hyp.supporting_structure:
            L.append("")
            L.append("_Supporting structure:_")
            for s in hyp.supporting_structure:
                L.append(f"  - {s}")
        if hyp.supporting_literature:
            L.append(f"_Supporting literature:_ {', '.join(hyp.supporting_literature)}")
        L.append("")

    # --- not supported -------------------------------------------------- #
    L.append("## Not supported by connectivity")
    for s in h.not_supported_by_connectivity:
        L.append(f"- {s}")
    L.append("")

    # --- experiments ---------------------------------------------------- #
    L.append("## Proposed experiments")
    for e in h.proposed_experiments:
        L.append(f"- **{e.hypothesis_ref}** — _{e.method}_: {e.expected_result}")
    L.append("")

    # --- caveats + verification ---------------------------------------- #
    if h.caveats:
        L.append("## Caveats")
        for c in h.caveats:
            L.append(f"- {c}")
        L.append("")
    L.append("## Verification notes")
    L.append(h.verification_notes or "_(none)_")
    L.append("")
    return "\n".join(L)


def _print_summary(h: Hypothesis) -> None:
    fp = h.fingerprint
    print(f"\n=== flyhypo: {h.cell_type} ({h.dataset}) ===")
    if not fp.found:
        print("  type NOT found.", end=" ")
        if fp.suggestions:
            print(f"suggestions: {', '.join(fp.suggestions)}")
        else:
            print("no suggestions.")
    else:
        print(f"  resolved {len(fp.resolved)} cell(s); NT={fp.predicted_nt or '?'}")
        print(f"  {len(fp.upstream)} upstream / {len(fp.downstream)} downstream partner types")
    print(f"  literature hits: {len(h.literature)}")
    print(f"  hypotheses: {len(h.hypotheses)} "
          f"(confidences: {[x.confidence for x in h.hypotheses]})")
    print(f"  proposed experiments: {len(h.proposed_experiments)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flyhypo", description=__doc__)
    ap.add_argument("cell_type", help="fly cell type, e.g. EPG, MBON01, SA1")
    ap.add_argument("--dataset", default="hemibrain:v1.2.1")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--out", default="outputs/")
    ap.add_argument("--no-cache", action="store_true", help="bypass on-disk cache")
    ap.add_argument(
        "--fingerprint-only",
        action="store_true",
        help="stop after the structural fingerprint (step 1; no API keys for LLM)",
    )
    args = ap.parse_args(argv)

    # Imported lazily so --help works without credentials/network.
    from . import connectome

    fp = connectome.build_fingerprint(
        args.cell_type, args.dataset, args.top_k, use_cache=not args.no_cache
    )

    if args.fingerprint_only:
        print(fp.model_dump_json(indent=2, by_alias=True))
        return 0 if fp.found else 2

    from . import literature, synthesize

    lit = literature.fetch_literature(fp, use_cache=not args.no_cache)
    result = synthesize.synthesize(fp, lit)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(args.cell_type)
    (out_dir / f"{slug}.json").write_text(
        result.model_dump_json(indent=2, by_alias=True)
    )
    (out_dir / f"{slug}.md").write_text(render_markdown(result))

    _print_summary(result)
    print(f"\nwrote {out_dir / (slug + '.json')} and {out_dir / (slug + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
