"""Cross-dataset replication: does a type's connectivity motif hold up in
another connectome (a different specimen, even a different sex)?

This is the first piece of the formerly out-of-scope "cross-dataset replication"
TODO. It stays purely STRUCTURAL — replication is about whether the wiring motif
recurs across individuals, which connectivity can answer directly (no LLM needed).

Data sources are other neuPrint datasets on the same server (e.g. male-cns, banc,
hemibrain:v1.1). FlyWire/Codex remains a separate later adapter (static CSV + CAVE,
per the guardrails) — see TODO in README; we do NOT scrape Codex.
"""

from __future__ import annotations

from . import connectome
from .schema import (
    DatasetSummary,
    PartnerAgreement,
    ReplicationReport,
    StructuralFingerprint,
)

# Sensible default comparison datasets that contain the central brain.
DEFAULT_OTHERS = ["male-cns:v1.0", "banc:v888"]


def _summary(fp: StructuralFingerprint) -> DatasetSummary:
    return DatasetSummary(
        dataset=fp.dataset,
        found=fp.found,
        n_cells=len(fp.resolved),
        predicted_nt=fp.predicted_nt,
        upstream=fp.upstream,
        downstream=fp.downstream,
        suggestions=fp.suggestions,
    )


def _type_set(partners) -> set[str]:
    return {p.type for p in partners if p.type}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return round(len(a & b) / len(a | b), 3)


def _agreements(summaries: list[DatasetSummary], direction: str):
    """Build per-partner-type agreement across datasets for one direction."""
    by_type: dict[str, dict[str, int]] = {}
    for s in summaries:
        partners = s.upstream if direction == "upstream" else s.downstream
        for p in partners:
            if not p.type:
                continue
            by_type.setdefault(p.type, {})[s.dataset] = p.total_weight
    out = []
    for ptype, weights in by_type.items():
        out.append(PartnerAgreement(
            partner_type=ptype, direction=direction,
            weights=weights, n_datasets=len(weights),
        ))
    out.sort(key=lambda a: (a.n_datasets, sum(a.weights.values())), reverse=True)
    return out


def replicate(
    cell_type: str,
    base_dataset: str = connectome.DEFAULT_DATASET,
    others: list[str] | None = None,
    top_k: int = 15,
    *,
    use_cache: bool = True,
) -> ReplicationReport:
    others = others or [d for d in DEFAULT_OTHERS if d != base_dataset]
    datasets = [base_dataset] + others

    summaries: list[DatasetSummary] = []
    fps: dict[str, StructuralFingerprint] = {}
    for ds in datasets:
        fp = connectome.build_fingerprint(cell_type, ds, top_k, use_cache=use_cache)
        fps[ds] = fp
        summaries.append(_summary(fp))

    base = fps[base_dataset]
    up_agree, down_agree = {}, {}
    for ds in others:
        other = fps[ds]
        up_agree[ds] = _jaccard(_type_set(base.upstream), _type_set(other.upstream))
        down_agree[ds] = _jaccard(_type_set(base.downstream), _type_set(other.downstream))

    found_datasets = [s.dataset for s in summaries if s.found]
    all_agree = _agreements(summaries, "upstream") + _agreements(summaries, "downstream")
    replicated = [a for a in all_agree if a.n_datasets >= 2]
    divergent = [a for a in all_agree if a.n_datasets == 1 and len(found_datasets) >= 2]

    notes = (
        f"Type '{cell_type}' resolved in {len(found_datasets)}/{len(datasets)} "
        f"datasets: {', '.join(found_datasets) or 'none'}. Agreement is the Jaccard "
        f"overlap of top-{top_k} partner-type sets vs the base ({base_dataset}). "
        "A motif replicating across specimens (and sexes) is stronger structural "
        "evidence; partner weights still vary across individuals."
    )
    return ReplicationReport(
        cell_type=cell_type,
        base_dataset=base_dataset,
        datasets=datasets,
        summaries=summaries,
        upstream_agreement=up_agree,
        downstream_agreement=down_agree,
        replicated_partners=replicated,
        divergent_partners=divergent,
        notes=notes,
    )
