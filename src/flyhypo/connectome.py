"""Connectome layer: a pure, typed wrapper around neuprint-python.

Given a cell-type query it returns a :class:`StructuralFingerprint`. On a
"type not found" it fuzzy-matches against the dataset's type list and returns
suggestions instead of crashing.

This module is deliberately self-contained (its only flyhypo imports are the
schema and the cache) so it could later be lifted into a standalone MCP server.

What a connectome tells you: who connects to whom, and roughly how strongly.
What it does NOT tell you: synapse sign, effective/intrinsic strength, or
neuromodulation. Predicted neurotransmitter is a *prediction*, not ground truth.
"""

from __future__ import annotations

import difflib
import os
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria, fetch_neurons, fetch_simple_connections
from neuprint.client import default_client, set_default_client

from . import cache
from .schema import Partner, ResolvedInstance, RoiWeight, StructuralFingerprint

load_dotenv()

DEFAULT_SERVER = os.environ.get("NEUPRINT_SERVER", "neuprint.janelia.org")
DEFAULT_DATASET = "hemibrain:v1.2.1"

# Cap how many resolved bodyIds we serialise into the fingerprint (a well-known
# type like EPG has dozens of cells; we don't need them all for a hypothesis).
MAX_RESOLVED = 50


class NeuprintAuthError(RuntimeError):
    pass


def get_client(dataset: str = DEFAULT_DATASET) -> Client:
    """Build (and memoise) a neuPrint client. Token comes from the env."""
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise NeuprintAuthError(
            "NEUPRINT_APPLICATION_CREDENTIALS is not set. Get a token from your "
            "account page at https://neuprint.janelia.org (top-right menu → "
            "'Account') and put it in .env (see .env.example)."
        )
    try:
        existing = default_client()
        if existing and existing.dataset == dataset:
            return existing
    except Exception:
        pass
    client = Client(DEFAULT_SERVER, dataset=dataset, token=token)
    set_default_client(client)
    return client


def _safe_col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def list_all_types(client: Client, dataset: str) -> list[str]:
    """Distinct, non-null neuron types in the dataset (cached on disk)."""
    cached = cache.get("types", dataset)
    if cached is not None:
        return cached
    cypher = (
        "MATCH (n:Neuron) WHERE n.type IS NOT NULL "
        "RETURN DISTINCT n.type AS type ORDER BY type"
    )
    df = client.fetch_custom(cypher)
    types = [t for t in df["type"].tolist() if t]
    cache.put("types", dataset, types)
    return types


def _aggregate_rois(roi_counts: pd.DataFrame, primary: set[str], kind: str,
                    top: int = 8) -> list[RoiWeight]:
    """Aggregate per-ROI synapse weights across all resolved cells.

    ``kind='post'`` → input ROIs (postsynaptic sites = where this cell receives).
    ``kind='pre'``  → output ROIs (presynaptic sites = where this cell sends).
    Restricted to primary (non-overlapping) ROIs to avoid double-counting the
    neuPrint ROI hierarchy.
    """
    if roi_counts is None or roi_counts.empty or kind not in roi_counts.columns:
        return []
    df = roi_counts[roi_counts["roi"].isin(primary)]
    grouped = (
        df.groupby("roi")[kind].sum().sort_values(ascending=False).head(top)
    )
    return [
        RoiWeight(roi=str(roi), weight=int(w))
        for roi, w in grouped.items()
        if w > 0
    ]


def _partners(conns: pd.DataFrame, partner_side: str, top_k: int) -> list[Partner]:
    """Aggregate a fetch_simple_connections frame by partner type.

    ``partner_side`` is 'pre' (for upstream partners) or 'post' (downstream).
    """
    if conns is None or conns.empty:
        return []
    type_col = f"type_{partner_side}"
    body_col = f"bodyId_{partner_side}"
    nt_col = f"predictedNt_{partner_side}"
    class_col = f"class_{partner_side}"

    out: list[Partner] = []
    for ptype, grp in conns.groupby(type_col, dropna=False):
        nt = None
        if nt_col in grp.columns:
            nts = [x for x in grp[nt_col].tolist() if isinstance(x, str) and x]
            nt = Counter(nts).most_common(1)[0][0] if nts else None
        cls = None
        if class_col in grp.columns:
            classes = [x for x in grp[class_col].tolist() if isinstance(x, str) and x]
            cls = Counter(classes).most_common(1)[0][0] if classes else None
        out.append(
            Partner(
                type=None if pd.isna(ptype) else str(ptype),
                n_cells=int(grp[body_col].nunique()),
                total_weight=int(grp["weight"].sum()),
                predicted_nt=nt,
                **{"class": cls},
            )
        )
    out.sort(key=lambda p: p.total_weight, reverse=True)
    return out[:top_k]


def _connections(criteria_kwargs: dict, *, direction: str, client: Client) -> pd.DataFrame:
    """fetch_simple_connections with extra neuron properties, degrading safely.

    direction='upstream'   → partners that synapse ONTO the target type.
    direction='downstream' → partners the target type synapses ONTO.
    """
    nc = NeuronCriteria(**criteria_kwargs)
    rich = ["type", "instance", "predictedNt", "class"]
    for props in (rich, ["type", "instance"]):
        try:
            if direction == "upstream":
                return fetch_simple_connections(
                    downstream_criteria=nc, properties=props, client=client
                )
            return fetch_simple_connections(
                upstream_criteria=nc, properties=props, client=client
            )
        except Exception:
            continue
    return pd.DataFrame()


def build_fingerprint(
    cell_type: str,
    dataset: str = DEFAULT_DATASET,
    top_k: int = 15,
    *,
    use_cache: bool = True,
) -> StructuralFingerprint:
    cache_key = f"{dataset}|{cell_type}|{top_k}"
    if use_cache:
        cached = cache.get("fingerprint", cache_key)
        if cached is not None:
            return StructuralFingerprint.model_validate(cached)

    client = get_client(dataset)

    neurons, roi_counts = fetch_neurons(
        NeuronCriteria(type=cell_type, regex=False, client=client), client=client
    )

    # --- not found → fuzzy-suggest -------------------------------------- #
    if neurons is None or neurons.empty:
        all_types = list_all_types(client, dataset)
        suggestions = difflib.get_close_matches(cell_type, all_types, n=8, cutoff=0.4)
        if not suggestions:  # substring fallback
            low = cell_type.lower()
            suggestions = [t for t in all_types if low in t.lower()][:8]
        fp = StructuralFingerprint(
            cell_type_query=cell_type,
            dataset=dataset,
            suggestions=suggestions,
            notes=(
                f"No neurons of type '{cell_type}' found in {dataset}. "
                f"Showing {len(suggestions)} nearest type name(s)."
            ),
        )
        if use_cache:
            cache.put("fingerprint", cache_key, fp.model_dump(by_alias=True))
        return fp

    # --- resolved instances --------------------------------------------- #
    resolved = [
        ResolvedInstance(
            bodyId=int(r.bodyId),
            type=None if pd.isna(r.type) else str(r.type),
            instance=None if pd.isna(getattr(r, "instance", None)) else str(r.instance),
        )
        for r in neurons.head(MAX_RESOLVED).itertuples()
    ]

    # --- predicted neurotransmitter (consensus across cells) ------------ #
    predicted_nt = None
    nt_series = _safe_col(neurons, "predictedNt", "consensusNt", "celltypePredictedNt")
    nt_note = ""
    if nt_series is not None:
        vals = [x for x in nt_series.tolist() if isinstance(x, str) and x]
        if vals:
            predicted_nt = Counter(vals).most_common(1)[0][0]
    else:
        # hemibrain:v1.2.1 stores no NT/class properties on neuron nodes.
        nt_note = (
            " Neurotransmitter and class are not provided by this dataset "
            "(null ≠ unknown-from-our-query)."
        )

    # --- ROIs (primary only) -------------------------------------------- #
    from neuprint import fetch_primary_rois

    try:
        primary = set(fetch_primary_rois(client=client))
    except Exception:
        primary = set(roi_counts["roi"].unique()) if roi_counts is not None else set()
    input_rois = _aggregate_rois(roi_counts, primary, "post")
    output_rois = _aggregate_rois(roi_counts, primary, "pre")

    # --- partners -------------------------------------------------------- #
    crit = {"type": cell_type, "regex": False}
    upstream = _partners(
        _connections(crit, direction="upstream", client=client), "pre", top_k
    )
    downstream = _partners(
        _connections(crit, direction="downstream", client=client), "post", top_k
    )

    fp = StructuralFingerprint(
        cell_type_query=cell_type,
        dataset=dataset,
        resolved=resolved,
        predicted_nt=predicted_nt,
        input_rois=input_rois,
        output_rois=output_rois,
        upstream=upstream,
        downstream=downstream,
        notes=f"Resolved {len(resolved)} cell(s) of type '{cell_type}'." + nt_note,
    )
    if use_cache:
        cache.put("fingerprint", cache_key, fp.model_dump(by_alias=True))
    return fp
