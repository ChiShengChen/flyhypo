"""flyhypo CLI: structure + literature → grounded functional hypothesis.

    flyhypo <cell_type> [--dataset hemibrain:v1.2.1] [--top-k 15] [--out outputs/]

Writes outputs/<cell_type>.json and outputs/<cell_type>.md and prints a summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .schema import HierarchyReport, Hypothesis, StructuralFingerprint


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

    # --- functional roles (headline answer) ----------------------------- #
    if h.functional_roles:
        L.append("## Functional roles (what this neuron is involved in)")
        L.append("_Each role is grounded in a paper id and/or a specific "
                 "connectivity number. Roles with neither are not listed._")
        L.append("")
        for r in h.functional_roles:
            L.append(f"### {r.function}")
            L.append(f"- _evidence:_ **{r.evidence_type}** · confidence: **{r.confidence}**")
            if r.references:
                L.append(f"- _references:_ {', '.join(r.references)}")
            if r.connectivity_basis:
                L.append("- _connectivity basis:_")
                for s in r.connectivity_basis:
                    L.append(f"  - {s}")
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
    if fp.is_neuron:
        L.append(
            f"- **Single neuron:** bodyId `{fp.neuron_bodyId}` — instance "
            f"`{fp.neuron_instance or '?'}` — type `{fp.neuron_type or '?'}`"
            + (f" (1 of {fp.n_in_type} cells of this type)" if fp.n_in_type else "")
        )
        if fp.sub_rois:
            sub = ", ".join(f"{r.roi} ({r.weight})" for r in fp.sub_rois)
            L.append(f"- **Topographic position (sub-compartments):** {sub}")
        L.append(
            "- _Single-cell scope: structure is this individual cell; its "
            "**function is inherited from its type**, localized to the position "
            "above. See caveats (n=1, no single-cell literature)._"
        )
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
    if h.reasoning_summary:
        L.append("## Model reasoning (summary)")
        L.append("_The model's own thought summary during synthesis — provided for "
                 "transparency; it is not authoritative evidence._")
        L.append("")
        L.append(h.reasoning_summary)
        L.append("")
    return "\n".join(L)


_LEVEL_TITLES = {
    "region": "REGION", "subregion": "SUBREGION", "umbrella": "UMBRELLA (system)",
    "cell_type": "CELL TYPE", "neuron": "NEURON",
}


def render_hierarchy_markdown(r: HierarchyReport) -> str:
    L: list[str] = []
    L.append(f"# Hierarchical functional analysis: `{r.query}`")
    L.append("")
    L.append(f"**Dataset:** {r.dataset}  ")
    L.append(
        f"**Resolved:** region `{r.region or '?'}` ▸ subregion `{r.subregion or '?'}` "
        f"▸ type `{r.cell_type or '?'}`"
        + (f" ▸ neuron `{r.neuron_bodyId}`" if r.neuron_bodyId else "")
    )
    L.append("")
    L.append(
        "> Functions are reported at each level (coarse → fine). Each role is "
        "grounded in a paper id and/or a connectivity number. A connectome gives "
        "connectivity, not synapse sign/strength/neuromodulation — these are "
        "**hypotheses for experimentalists**."
    )
    L.append("")
    for lvl in r.levels:
        L.append(f"## {_LEVEL_TITLES.get(lvl.level, lvl.level.upper())}: {lvl.label}")
        if lvl.note:
            L.append(f"_{lvl.note}_")
        for role in lvl.functional_roles:
            L.append(f"### {role.function}")
            L.append(f"- _evidence:_ **{role.evidence_type}** · confidence: **{role.confidence}**")
            if role.references:
                L.append(f"- _references:_ {', '.join(role.references)}")
            if role.connectivity_basis:
                L.append("- _connectivity basis:_")
                for s in role.connectivity_basis:
                    L.append(f"  - {s}")
            L.append("")
        L.append("")
    L.append("## Literature used")
    for hit in r.literature:
        yr = f" ({hit.year})" if hit.year else ""
        L.append(f"- **[{hit.source}:{hit.id}]**{yr} {hit.title}")
    L.append("")
    L.append("## Caveats")
    for c in r.caveats:
        L.append(f"- {c}")
    L.append("")
    L.append("## Verification notes")
    L.append(r.verification_notes or "_(none)_")
    L.append("")
    if r.reasoning_summary:
        L.append("## Model reasoning (summary)")
        L.append(r.reasoning_summary)
        L.append("")
    return "\n".join(L)


def _print_hierarchy_summary(r: HierarchyReport) -> None:
    print(f"\n=== flyhypo hierarchy: {r.query} ({r.dataset}) ===")
    print(f"  region={r.region} subregion={r.subregion} type={r.cell_type}"
          + (f" neuron={r.neuron_bodyId}" if r.neuron_bodyId else ""))
    for lvl in r.levels:
        print(f"  [{lvl.level}] {lvl.label}: {len(lvl.functional_roles)} role(s)")
    print(f"  literature hits: {len(r.literature)}")


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
    ap.add_argument("cell_type", nargs="?", help="fly cell type, e.g. EPG, MBON01, SA1")
    ap.add_argument(
        "--neuron", type=int, metavar="BODYID",
        help="single-neuron mode: build a structural fingerprint for one bodyId "
             "(function is inherited from its type; see caveats)",
    )
    ap.add_argument("--dataset", default="hemibrain:v1.2.1")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--out", default="outputs/")
    ap.add_argument("--no-cache", action="store_true", help="bypass on-disk cache")
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the LLM verification pass (faster, fewer tokens; citation "
             "hygiene still applied)",
    )
    ap.add_argument(
        "--hierarchy",
        action="store_true",
        help="analyze every level: region ▸ subregion ▸ umbrella(system) ▸ "
             "cell_type ▸ neuron, each with functional roles + refs",
    )
    ap.add_argument(
        "--fingerprint-only",
        action="store_true",
        help="stop after the structural fingerprint (step 1; no API keys for LLM)",
    )
    args = ap.parse_args(argv)
    if not args.cell_type and args.neuron is None:
        ap.error("provide a cell_type, or --neuron BODYID for single-neuron mode")

    # Imported lazily so --help works without credentials/network.
    from . import connectome

    # --- multi-level hierarchy mode ------------------------------------- #
    if args.hierarchy:
        from . import hierarchy, literature, synthesize

        context, type_fp, _ = hierarchy.build_context(
            args.cell_type, args.dataset, args.top_k, args.neuron,
            use_cache=not args.no_cache,
        )
        anchor = type_fp if (type_fp and type_fp.found) else None
        if anchor is None:
            print("Could not resolve a cell type for the hierarchy "
                  f"(query={args.cell_type or args.neuron}).")
            if type_fp and type_fp.suggestions:
                print("suggestions:", ", ".join(type_fp.suggestions))
            return 2
        lit = literature.fetch_literature(anchor, use_cache=not args.no_cache)
        report = synthesize.synthesize_hierarchy(context, lit, verify=not args.no_verify)

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = _slug((f"bodyId_{args.neuron}" if args.neuron else args.cell_type) + "_hierarchy")
        (out_dir / f"{slug}.json").write_text(report.model_dump_json(indent=2, by_alias=True))
        (out_dir / f"{slug}.md").write_text(render_hierarchy_markdown(report))
        _print_hierarchy_summary(report)
        print(f"\nwrote {out_dir / (slug + '.json')} and {out_dir / (slug + '.md')}")
        return 0

    if args.neuron is not None:
        fp = connectome.build_neuron_fingerprint(
            args.neuron, args.dataset, args.top_k, use_cache=not args.no_cache
        )
        label = f"bodyId_{args.neuron}"
    else:
        fp = connectome.build_fingerprint(
            args.cell_type, args.dataset, args.top_k, use_cache=not args.no_cache
        )
        label = args.cell_type

    if args.fingerprint_only:
        print(fp.model_dump_json(indent=2, by_alias=True))
        return 0 if fp.found else 2

    from . import literature, synthesize

    lit = literature.fetch_literature(fp, use_cache=not args.no_cache)
    result = synthesize.synthesize(fp, lit, verify=not args.no_verify)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(label)
    (out_dir / f"{slug}.json").write_text(
        result.model_dump_json(indent=2, by_alias=True)
    )
    (out_dir / f"{slug}.md").write_text(render_markdown(result))

    _print_summary(result)
    print(f"\nwrote {out_dir / (slug + '.json')} and {out_dir / (slug + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
