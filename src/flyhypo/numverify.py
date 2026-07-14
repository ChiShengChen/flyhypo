"""Deterministic connectivity-number verification — the structural counterpart
to the literature verify layer.

verify.py checks literature claims (verbatim quote, mis-attribution, citations)
and deliberately passes connectivity claims through, noting their numbers are
"verified elsewhere". This module IS that elsewhere: every synapse-scale number a
claim cites in its connectivity_basis / supporting_structure must actually appear
in the structural fingerprint (or the cross-dataset replication evidence). A claim
citing a number that is nowhere in the evidence is likely a model error, so it is
flagged and its confidence downgraded one tier (never dropped — a single wrong
number shouldn't nuke an otherwise-real structural role; downgrade + surface it).

Conservative by design: only integers ≥ 100 that are not plausible years are
checked, so incidental small counts ("top 8", "1 of 46") and citation years don't
cause false positives.
"""

from __future__ import annotations

import re

from .schema import HierarchyAnalysis, HypothesisAnalysis, StructuralFingerprint

_CONF_ORDER = ["speculative", "low", "medium", "high"]
_NUM = re.compile(r"\d[\d,]*")


def _downgrade(conf: str) -> str:
    i = _CONF_ORDER.index(conf) if conf in _CONF_ORDER else 0
    return _CONF_ORDER[max(0, i - 1)]


def _replication_numbers(replication: dict | None) -> set[int]:
    nums: set[int] = set()
    for a in (replication or {}).get("replicated_partner_types", []):
        for w in (a.get("weights") or {}).values():
            try:
                nums.add(int(w))
            except (TypeError, ValueError):
                pass
    return nums


def fingerprint_numbers(fp: StructuralFingerprint, replication: dict | None = None) -> set[int]:
    """Every number a connectivity claim could legitimately cite."""
    nums: set[int] = {len(fp.resolved)}
    for coll in (fp.input_rois, fp.output_rois, fp.sub_rois):
        for r in coll:
            nums.add(int(r.weight))
    for coll in (fp.upstream, fp.downstream):
        for p in coll:
            nums.add(int(p.total_weight))
            nums.add(int(p.n_cells))
    if fp.n_in_type:
        nums.add(int(fp.n_in_type))
    return nums | _replication_numbers(replication)


def context_numbers(context: dict, replication: dict | None = None) -> set[int]:
    """Same, but from a hierarchy context dict (trimmed fingerprint as tuples)."""
    nums: set[int] = set()
    tf = context.get("type_fingerprint") or {}
    for key in ("input_rois", "output_rois", "sub_rois"):
        for _roi, w in tf.get(key, []) or []:
            nums.add(int(w))
    for key in ("upstream", "downstream"):
        for row in tf.get(key, []) or []:
            # (type, n_cells, total_weight)
            for v in row[1:]:
                try:
                    nums.add(int(v))
                except (TypeError, ValueError):
                    pass
    for key in ("region_dominant_types", "subregion_dominant_types"):
        for d in context.get(key, []) or []:
            nums.add(int(d.get("n_cells", 0)))
    neuron = context.get("neuron") or {}
    for _roi, w in neuron.get("sub_rois", []) or []:
        nums.add(int(w))
    if neuron.get("n_in_type"):
        nums.add(int(neuron["n_in_type"]))
    return nums | _replication_numbers(replication)


def _unverified(text: str, valid: set[int]) -> set[int]:
    out: set[int] = set()
    for m in _NUM.findall(text):
        n = int(m.replace(",", ""))
        if n < 100 or 1990 <= n <= 2099:  # ignore small counts and citation years
            continue
        if n not in valid:
            out.add(n)
    return out


def _scan(items, label: str, field: str, valid: set[int]) -> list[str]:
    flags: list[str] = []
    for j, it in enumerate(items, 1):
        bad: set[int] = set()
        for s in getattr(it, field, None) or []:
            bad |= _unverified(s, valid)
        if bad:
            old = it.confidence
            it.confidence = _downgrade(it.confidence)
            flags.append(f"{label}{j} {sorted(bad)} ({old}→{it.confidence})")
    return flags


def check_analysis(analysis: HypothesisAnalysis, fp: StructuralFingerprint,
                   replication: dict | None = None) -> str:
    """Flag+downgrade roles/hypotheses citing numbers absent from the fingerprint."""
    valid = fingerprint_numbers(fp, replication)
    flags = (_scan(analysis.functional_roles, "role", "connectivity_basis", valid)
             + _scan(analysis.hypotheses, "H", "supporting_structure", valid))
    if not flags:
        return ""
    return ("[verify] connectivity number(s) not found in the fingerprint "
            "(possible error) — affected claims downgraded: " + "; ".join(flags))


def check_hierarchy(analysis: HierarchyAnalysis, context: dict,
                    replication: dict | None = None) -> str:
    valid = context_numbers(context, replication)
    flags: list[str] = []
    for lvl in analysis.levels:
        flags += _scan(lvl.functional_roles, f"{lvl.level}:role", "connectivity_basis", valid)
    if not flags:
        return ""
    return ("[verify] connectivity number(s) not found in the evidence "
            "(possible error) — affected roles downgraded: " + "; ".join(flags))
