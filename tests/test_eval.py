"""Evaluation scorer (coarse keyword matching)."""

from flyhypo.eval import score_type

GOLD = {
    "type": "X",
    "known_functions": [
        {"name": "heading", "keywords": ["heading", "compass"]},
        {"name": "sleep", "keywords": ["sleep", "arousal"]},
    ],
    "key_partners": ["ER", "PEN"],
}


def test_recall_precision_partner_coverage():
    report = {
        "functional_roles": [
            {"function": "encodes heading via a compass", "references": ["10.1/x"]},
            {"function": "unrelated widget role", "references": []},
        ],
        "hypotheses": [],
        "fingerprint": {"upstream": [{"type": "ER4m"}], "downstream": [{"type": "PEG"}]},
    }
    s = score_type(GOLD, report)
    assert s["recall"] == 0.5            # heading matched, sleep missing
    assert s["missing"] == ["sleep"]
    assert s["precision"] == 0.5         # 1 of 2 roles maps to a gold function
    assert s["partner_cov"] == 0.5       # ER present (ER4m), PEN absent
    assert s["cites_ok"] is True


def test_bad_citation_flagged():
    report = {"functional_roles": [{"function": "heading compass", "references": ["n/a"]}],
              "hypotheses": [], "fingerprint": {}}
    assert score_type(GOLD, report)["cites_ok"] is False
