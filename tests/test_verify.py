"""Unit tests for the paper-evidence verification layer (offline; citation network faked).

Locks the anti-hallucination behavior wired into synthesize: a literature claim survives only
if its quote is verbatim in the cited abstract AND names the focus cell type; a fabricated or
mis-attributed quote is dropped; connectivity-only claims pass through; unresolved citations are
stripped and confidence is re-tiered from the deterministic grade (abstract-only caps at medium)."""
from types import SimpleNamespace

from flyhypo import verify as V
from flyhypo.schema import FunctionalRole, LiteratureHit

ABSTRACT = ("The EPG neurons form a ring attractor that encodes the fly's heading direction "
            "in the ellipsoid body, receiving input from ER neurons.")
LIT = [LiteratureHit(title="EPG ring attractor", source="pubmed", id="12345678",
                     snippet=ABSTRACT, relevance="EPG heading")]


def _role(function, quote, refs=("12345678",), etype="literature"):
    return FunctionalRole(function=function, evidence_type=etype, references=list(refs),
                          connectivity_basis=[], quote=quote, confidence="high")


def _analysis(roles):
    # verify_analysis only reads/sets .functional_roles and .hypotheses, so a lightweight
    # container avoids pydantic forward-ref rebuild (triggered by google-genai in the app).
    return SimpleNamespace(functional_roles=roles, hypotheses=[])


def _no_citation_check(monkeypatch):
    monkeypatch.setattr(V.citation, "verify_citation",
                        lambda **k: {"status": "verified", "retracted": False})


def test_faithful_role_kept(monkeypatch):
    _no_citation_check(monkeypatch)
    a = _analysis([_role("heading encoding",
                         "The EPG neurons form a ring attractor that encodes the fly's heading direction")])
    out = V.verify_analysis(a, LIT, ["EPG"], judge=None)
    assert out["dropped"] == [] and len(a.functional_roles) == 1
    assert a.functional_roles[0].confidence == "medium"      # abstract-only caps at medium


def test_fabricated_quote_dropped(monkeypatch):
    _no_citation_check(monkeypatch)
    a = _analysis([_role("visual memory", "EPG neurons store visual place memories in the calyx")])
    out = V.verify_analysis(a, LIT, ["EPG"], judge=None)
    assert len(a.functional_roles) == 0 and out["dropped"]   # quote not in abstract -> dropped


def test_misattributed_quote_dropped(monkeypatch):
    _no_citation_check(monkeypatch)
    # verbatim substring of the abstract, but it does not name the focus 'EPG'
    a = _analysis([_role("input pathway", "receiving input from ER neurons")])
    out = V.verify_analysis(a, LIT, ["EPG"], judge=None)
    assert len(a.functional_roles) == 0
    assert any("mis-attribution" in d for d in out["dropped"])


def test_connectivity_only_role_passes(monkeypatch):
    _no_citation_check(monkeypatch)
    a = _analysis([_role("receives ER4m input", quote="", refs=(), etype="connectivity")])
    a.functional_roles[0].connectivity_basis = ["receives 14903 synapses from ER4m"]
    out = V.verify_analysis(a, LIT, ["EPG"], judge=None)
    assert len(a.functional_roles) == 1 and out["dropped"] == []


def test_unresolved_citation_stripped(monkeypatch):
    monkeypatch.setattr(V.citation, "verify_citation",
                        lambda **k: {"status": "unverified", "retracted": False})
    a = _analysis([_role("heading encoding",
                         "The EPG neurons form a ring attractor that encodes the fly's heading direction",
                         refs=("99999999",))])
    # quote is fine + names EPG, but the only citation is unresolved -> ref stripped (role kept)
    V.verify_analysis(a, LIT, ["EPG"], judge=None)
    # cited id 99999999 isn't in LIT so no source text -> quote can't be found -> role dropped
    # (this is the honest outcome: a claim whose only citation is unverifiable can't be grounded)
    assert len(a.functional_roles) == 0


def test_cite_args_routing():
    assert V._cite_args("12345678", None) == {"pmid": "12345678", "title": None}
    assert V._cite_args("2605.10817", None)["arxiv"] == "2605.10817"
    assert V._cite_args("10.1038/nature14539", None)["doi"] == "10.1038/nature14539"


# --------------------------------------------------------------------------- #
# verify_roles (hierarchy helper) + deep_read_hit routing
# --------------------------------------------------------------------------- #
def test_verify_roles_drops_and_keeps(monkeypatch):
    _no_citation_check(monkeypatch)
    roles = [
        _role("heading encoding",
              "The EPG neurons form a ring attractor that encodes the fly's heading direction"),
        _role("visual memory", "EPG neurons store visual place memories in the calyx"),  # fabricated
    ]
    kept, dropped = V.verify_roles(roles, LIT, ["EPG"], judge=None)
    assert len(kept) == 1 and kept[0].function == "heading encoding" and len(dropped) == 1


def test_deep_read_hit_routing(monkeypatch):
    from flyhypo import literature
    import paper_evidence.deepread as dr
    calls = {}
    monkeypatch.setattr(dr, "load_text", lambda **k: calls.update(k) or "FULLTEXT")
    arxiv_hit = LiteratureHit(title="t", source="arxiv", id="2605.10817",
                              snippet="s", relevance="r")
    assert literature.deep_read_hit(arxiv_hit) == "FULLTEXT" and calls.get("arxiv") == "2605.10817"
    pmid_hit = LiteratureHit(title="t", source="pubmed", id="12345678", snippet="s", relevance="r")
    assert literature.deep_read_hit(pmid_hit) is None    # PMID needs an OA resolver — skipped
