"""Evaluation scorer (coarse keyword matching)."""

from flyhypo.eval import score_negative, score_type

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


NEG = {"type": "SA1", "negative": True,
       "expect": {"found": False, "suggestions": True,
                  "max_confidence": "speculative", "no_fabricated_citations": True}}


def test_negative_graceful_pass():
    report = {"functional_roles": [], "hypotheses": [], "literature": [],
              "fingerprint": {"resolved": [], "suggestions": ["SA1_a", "SA1_b"]}}
    s = score_negative(NEG, report)
    assert s["ok"] and all(s["checks"].values())


def test_negative_fabricated_and_overconfident_fail():
    report = {
        "functional_roles": [{"function": "x", "confidence": "high", "references": ["10.9/fake"]}],
        "hypotheses": [], "literature": [],  # cited id not in literature → fabricated
        "fingerprint": {"resolved": [{"bodyId": 1}], "suggestions": []},  # found when expected not
    }
    s = score_negative(NEG, report)
    assert not s["ok"]
    assert s["checks"]["no_fab"] is False and s["checks"]["max_conf"] is False
    assert s["checks"]["found"] is False  # found True but expected False
