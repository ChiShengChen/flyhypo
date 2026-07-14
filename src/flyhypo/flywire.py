"""FlyWire (FAFB) adapter — a 4th replication dataset, from the static Codex export.

FlyWire is NOT on neuPrint. Per the guardrails, the sanctioned path is the
official **static CSV download** (Codex) + CAVE — we do NOT scrape Codex. Point
FLYWIRE_DATA_DIR at a directory holding the two public exports:

  - classification.csv[.gz]  — a `root_id` column + a type column
                               (`hemibrain_type` preferred, else `cell_type`/`type`)
  - connections.csv[.gz]     — `pre_root_id`, `post_root_id`, `syn_count`

Then `--replicate` auto-includes FlyWire (FAFB ♀) alongside male-cns / banc, so a
motif can be checked against yet another specimen. Matching is by `hemibrain_type`
so a hemibrain query (e.g. EPG) lines up. Without the data, FlyWire is skipped.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import pandas as pd

from .schema import Partner, ResolvedInstance, StructuralFingerprint


def data_dir(ddir: str | None = None) -> str | None:
    return ddir or os.environ.get("FLYWIRE_DATA_DIR")


def _find(d: Path, stem: str) -> Path | None:
    for name in (f"{stem}.csv", f"{stem}.csv.gz"):
        if (d / name).exists():
            return d / name
    return None


def flywire_available(ddir: str | None = None) -> bool:
    d = data_dir(ddir)
    if not d:
        return False
    dp = Path(d)
    return _find(dp, "classification") is not None and _find(dp, "connections") is not None


@functools.lru_cache(maxsize=4)
def _load(ddir: str):
    dp = Path(ddir)
    cls = pd.read_csv(_find(dp, "classification"))
    con = pd.read_csv(_find(dp, "connections"))
    type_col = next((c for c in ("hemibrain_type", "cell_type", "type") if c in cls.columns), None)
    if type_col is None:
        raise ValueError("classification.csv needs a hemibrain_type / cell_type / type column")
    id2type = dict(zip(cls["root_id"], cls[type_col]))
    return id2type, con, type_col


def _partners(con: pd.DataFrame, ids: set, id2type: dict, side: str, top_k: int) -> list[Partner]:
    """side='pre'  → upstream (partners synapsing ONTO ids);
       side='post' → downstream (partners ids synapse ONTO)."""
    match_col = "post_root_id" if side == "pre" else "pre_root_id"
    partner_col = "pre_root_id" if side == "pre" else "post_root_id"
    sub = con[con[match_col].isin(ids)].copy()
    if sub.empty:
        return []
    sub["ptype"] = sub[partner_col].map(id2type)
    agg = sub.groupby("ptype", dropna=True).agg(
        total_weight=("syn_count", "sum"), n_cells=(partner_col, "nunique"))
    agg = agg.sort_values("total_weight", ascending=False).head(top_k)
    return [Partner(type=str(t), n_cells=int(r.n_cells), total_weight=int(r.total_weight),
                    **{"class": None}) for t, r in agg.iterrows()]


def flywire_fingerprint(cell_type: str, top_k: int = 15,
                        ddir: str | None = None) -> StructuralFingerprint:
    """Structural fingerprint for `cell_type` from the FlyWire export (partners only)."""
    d = data_dir(ddir)
    if not d or not flywire_available(d):
        return StructuralFingerprint(cell_type_query=cell_type, dataset="flywire",
                                     notes="FlyWire data not configured (set FLYWIRE_DATA_DIR).")
    id2type, con, _ = _load(d)
    ids = {rid for rid, t in id2type.items() if t == cell_type}
    if not ids:
        return StructuralFingerprint(cell_type_query=cell_type, dataset="flywire",
                                     notes=f"type '{cell_type}' not found in the FlyWire export.")
    return StructuralFingerprint(
        cell_type_query=cell_type, dataset="flywire",
        resolved=[ResolvedInstance(bodyId=int(i), type=cell_type) for i in list(ids)[:50]],
        upstream=_partners(con, ids, id2type, "pre", top_k),
        downstream=_partners(con, ids, id2type, "post", top_k),
        notes=f"FlyWire (FAFB) — {len(ids)} cell(s) of type '{cell_type}' from the static export.",
    )
