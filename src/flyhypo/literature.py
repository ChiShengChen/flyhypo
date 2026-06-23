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
    ct = fp.cell_type_query

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

    for query, why in queries:
        if len(hits) >= max_hits * 2:
            break
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

    # Rank PubMed (keyword-relevant) ahead of other sources, then preserve
    # insertion order — which follows query order (exact cell type first), so
    # the most on-topic hits stay on top. We deliberately do NOT sort by year.
    ranked = sorted(
        hits.values(), key=lambda h: 0 if h.source == "pubmed" else 1
    )
    return ranked[:max_hits]
