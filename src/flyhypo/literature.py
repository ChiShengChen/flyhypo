"""Literature layer: retrieve functional evidence via paper-search-mcp.

We call the package's search classes directly (fastest working integration;
PubMed and bioRxiv need no API key). Queries are built automatically from the
structural fingerprint: the cell-type name, its main neuropils, and its top
partner types. We keep ONLY abstracts/metadata — never full text.
"""

from __future__ import annotations

from . import cache
from .schema import LiteratureHit, StructuralFingerprint

# Generic neuropil-abbreviation glossary (region metadata, NOT cell-specific
# biology) used only to widen literature search recall. neuPrint ROI codes like
# "EB" are poor search terms; their full names are what papers actually use.
NEUROPIL_GLOSSARY = {
    "EB": "ellipsoid body",
    "PB": "protocerebral bridge",
    "FB": "fan-shaped body",
    "NO": "noduli",
    "BU": "bulb",
    "LAL": "lateral accessory lobe",
    "MB": "mushroom body",
    "CA": "calyx",
    "AL": "antennal lobe",
    "LH": "lateral horn",
    "AOTU": "anterior optic tubercle",
    "PED": "mushroom body peduncle",
}


def _expand_roi(roi: str) -> str:
    """Map a (possibly side-suffixed) ROI code to a search-friendly name."""
    base = roi.split("(")[0].strip().rstrip("LR").strip("_").strip()
    for code, name in NEUROPIL_GLOSSARY.items():
        if base == code or base.startswith(code):
            return name
    return roi


def build_queries(fp: StructuralFingerprint) -> list[tuple[str, str]]:
    """Return (query, why-relevant) pairs derived from the fingerprint."""
    queries: list[tuple[str, str]] = []
    # In single-neuron mode the query string is "bodyId:NNN" (useless for search);
    # there is no literature about an individual cell, so search by its TYPE.
    ct = fp.neuron_type or fp.cell_type_query

    # 1. The type itself.
    queries.append((f'{ct} Drosophila neuron', f"exact cell type '{ct}'"))

    # 2. Type + its dominant neuropil(s).
    rois = (fp.input_rois[:2] + fp.output_rois[:2])
    neuropils = []
    for rw in rois:
        name = _expand_roi(rw.roi)
        if name not in neuropils:
            neuropils.append(name)
    for name in neuropils[:2]:
        queries.append(
            (f'Drosophila {name} {ct}', f"{ct} in its main neuropil ({name})")
        )

    # 3. Top partner types (functional context from connectivity).
    partners = [p.type for p in (fp.upstream[:3] + fp.downstream[:3]) if p.type]
    seen = set()
    for ptype in partners:
        if ptype in seen or ptype == ct:
            continue
        seen.add(ptype)
        queries.append(
            (f'Drosophila {ct} {ptype}', f"connectivity partner type {ptype}")
        )

    # 4. Region-only fallback (helps when the exact type is unstudied).
    if neuropils:
        queries.append(
            (f'Drosophila {neuropils[0]} function circuit',
             f"region-level fallback ({neuropils[0]})")
        )
    return queries[:8]


import os

# bioRxiv's searcher in this package returns recent preprints largely regardless
# of the query (observed: mammalian papers for fly queries), which pollutes
# results. PubMed keyword search is reliable and on-topic, so it is the default.
# Set FLYHYPO_USE_BIORXIV=1 to also include bioRxiv (best-effort).
USE_BIORXIV = os.environ.get("FLYHYPO_USE_BIORXIV", "") == "1"


def _search_one(query: str, max_results: int) -> list:
    """Run one query against PubMed (and optionally bioRxiv), tolerating failures."""
    papers = []
    try:
        from paper_search_mcp.academic_platforms.pubmed import PubMedSearcher

        papers += PubMedSearcher().search(query, max_results=max_results) or []
    except Exception:
        pass
    if USE_BIORXIV:
        try:
            from paper_search_mcp.academic_platforms.biorxiv import BioRxivSearcher

            papers += BioRxivSearcher().search(query, max_results=3) or []
        except Exception:
            pass
    return papers


def _year_of(paper) -> int | None:
    d = getattr(paper, "published_date", None)
    if d is None:
        return None
    try:
        return int(getattr(d, "year", str(d)[:4]))
    except (ValueError, TypeError):
        return None


def fetch_literature(
    fp: StructuralFingerprint, max_hits: int = 8, *, use_cache: bool = True
) -> list[LiteratureHit]:
    queries = build_queries(fp)
    hits: dict[str, LiteratureHit] = {}  # dedup by id (or title)

    # Saturation-aware retrieval: stop once k consecutive queries add nothing new
    # (recall.saturated), instead of only the fixed max_hits*2 cap. Coverage-honest.
    try:
        from paper_evidence.recall import new_count, saturated
    except Exception:  # noqa: BLE001 — degrade to the fixed cap if paper-evidence absent
        new_count = saturated = None
    new_counts: list[int] = []

    for query, why in queries:
        if len(hits) >= max_hits * 2:
            break
        if saturated and saturated(new_counts, k=3):
            break
        before = len(hits)
        cache_key = query
        papers_raw = cache.get("lit", cache_key) if use_cache else None
        if papers_raw is None:
            papers = _search_one(query, max_results=5)
            papers_raw = []
            for p in papers:
                papers_raw.append(
                    {
                        "id": getattr(p, "doi", "") or getattr(p, "paper_id", ""),
                        "title": getattr(p, "title", "") or "",
                        "source": getattr(p, "source", "") or "",
                        "year": _year_of(p),
                        "abstract": getattr(p, "abstract", "") or "",
                    }
                )
            if use_cache:
                cache.put("lit", cache_key, papers_raw)

        for pr in papers_raw:
            pid = pr["id"] or pr["title"]
            if not pid or pid in hits or not pr["title"]:
                continue
            abstract = pr["abstract"]
            snippet = (abstract[:500] + "…") if len(abstract) > 500 else abstract
            hits[pid] = LiteratureHit(
                title=pr["title"],
                source=pr["source"] or "pubmed",
                id=pr["id"] or "n/a",
                year=pr["year"],
                snippet=snippet or "(no abstract available)",
                relevance=why,
            )
        new_counts.append(len(hits) - before)

    values = list(hits.values())

    # Optional semantic re-ranking (FLYHYPO_SEMANTIC=1): order hits by embedding
    # similarity to the cell's functional question, so a relevant paper that shares
    # no keywords with the query still floats up. Falls back silently if unavailable.
    if os.environ.get("FLYHYPO_SEMANTIC") == "1" and values:
        try:
            from paper_evidence.semantic import rank_chunks, embed_texts
            ct = fp.neuron_type or fp.cell_type_query
            question = f"functional role of {ct} neurons in the Drosophila brain"
            docs = [f"{h.title}. {h.snippet}" for h in values]
            order = rank_chunks(question, docs, embed_texts, k=len(docs))
            values = [values[i] for i, _, _ in order]
        except Exception:  # noqa: BLE001 — embeddings optional
            pass

    # Rank PubMed (keyword-relevant) ahead of other sources, then preserve
    # order (semantic if enabled, else query order — exact cell type first).
    ranked = sorted(values, key=lambda h: 0 if h.source == "pubmed" else 1)
    return ranked[:max_hits]


def deep_read_hit(hit: LiteratureHit, *, contact_email: str | None = None) -> str | None:
    """EXPERIMENTAL, opt-in — fetch a hit's full text (arXiv / OA PDF URL) so verification
    has more than the abstract to re-grep against, lifting the abstract-only confidence cap
    for that paper. Returns the full text, or None if not open-access / not fetchable.

    This deliberately departs from flyhypo's default "abstracts only" policy, so it is NOT
    wired into the pipeline — call it explicitly (e.g. behind FLYHYPO_DEEP_READ) when you
    want full-text grounding for the open-access subset. Requires paper-evidence[pdf].
    """
    try:
        from paper_evidence.deepread import load_text
    except Exception:  # noqa: BLE001
        return None
    ident = str(hit.id or "")
    kwargs: dict = {"contact_email": contact_email}
    import re as _re
    if _re.match(r"^\d{4}\.\d{4,5}", ident):
        kwargs["arxiv"] = ident
    elif ident.lower().startswith("http"):
        kwargs["url"] = ident
    else:
        return None  # PMIDs / bare DOIs need an OA resolver — out of scope for this helper
    try:
        return load_text(**kwargs)
    except Exception:  # noqa: BLE001
        return None
