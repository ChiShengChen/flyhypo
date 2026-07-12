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

    # Optional citation snowball (FLYHYPO_SNOWBALL=1): expand from the top hits via
    # references + citations, catching relevant papers no keyword query guessed.
    if os.environ.get("FLYHYPO_SNOWBALL") == "1" and hits:
        seeds = [h.id for h in list(hits.values())[:3] if h.id and h.id != "n/a"]
        for rec in snowball(seeds, max_per_seed=15):
            pid = rec["id"] or rec["title"]
            if not pid or pid in hits:
                continue
            abstract = rec["abstract"]
            snippet = (abstract[:500] + "…") if len(abstract) > 500 else abstract
            hits[pid] = LiteratureHit(
                title=rec["title"], source="s2-snowball", id=rec["id"] or "n/a",
                year=rec.get("year"), snippet=snippet or "(no abstract available)",
                relevance="snowballed (references/citations of a top hit)")

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


# --------------------------------------------------------------------------- #
# Citation snowball (Semantic Scholar) — recall a query never guessed.
# Self-contained (stdlib urllib), so flyhypo needs no external skill. Set
# SEMANTIC_SCHOLAR_API_KEY to avoid 429s; degrades gracefully to [] on failure.
# --------------------------------------------------------------------------- #
_S2 = "https://api.semanticscholar.org/graph/v1"


def _seed_ref(hit_id: str) -> str | None:
    """A cited hit id -> a Semantic Scholar paper reference (PMID:/DOI:/ARXIV:)."""
    import re
    s = str(hit_id or "").strip()
    if s.isdigit():
        return f"PMID:{s}"
    if re.match(r"^\d{4}\.\d{4,5}", s):
        return f"ARXIV:{s.split('v')[0]}"
    m = re.search(r"10\.\d{4,}/\S+", s)
    if m:
        return f"DOI:{m.group(0)}"
    return None


def _s2_get(path: str, params: dict):
    import json
    import urllib.parse
    import urllib.request
    url = f"{_S2}/{path}?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "flyhypo/0.1"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        headers["x-api-key"] = key
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 — 429 / offline -> no snowball, not a crash
        return None


def _edge_records(seed_ref: str, direction: str, limit: int) -> list[dict]:
    node = "references" if direction == "refs" else "citations"
    inner = "citedPaper" if direction == "refs" else "citingPaper"
    js = _s2_get(f"paper/{seed_ref}/{node}",
                 {"fields": "title,abstract,year,externalIds", "limit": min(limit, 100)})
    out = []
    for d in (js or {}).get("data", []):
        p = d.get(inner) or {}
        if not p.get("title"):
            continue
        ext = p.get("externalIds") or {}
        pid = ext.get("DOI") or (str(ext["PubMed"]) if ext.get("PubMed") else "") or p.get("paperId", "")
        out.append({"id": pid, "title": p["title"], "abstract": p.get("abstract") or "",
                    "year": p.get("year")})
    return out


def snowball(seed_ids: list[str], *, direction: str = "both",
             max_per_seed: int = 15) -> list[dict]:
    """References + citations of the seed papers (dedup), via Semantic Scholar."""
    dirs = ["refs", "cites"] if direction == "both" else [direction]
    out, seen = [], set()
    for sid in seed_ids:
        ref = _seed_ref(sid)
        if not ref:
            continue
        for d in dirs:
            for rec in _edge_records(ref, d, max_per_seed):
                key = rec["id"] or rec["title"]
                if key and key not in seen:
                    seen.add(key)
                    out.append(rec)
    return out


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
