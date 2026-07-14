"""Deterministic connectivity-number guard (numverify)."""

from flyhypo.numverify import check_analysis, fingerprint_numbers
from flyhypo.schema import (
    FunctionalRole,
    HypothesisAnalysis,
    Partner,
    ResolvedInstance,
    RoiWeight,
    StructuralFingerprint,
)


def _fp() -> StructuralFingerprint:
    return StructuralFingerprint(
        cell_type_query="EPG",
        dataset="hemibrain:v1.2.1",
        resolved=[ResolvedInstance(bodyId=1, type="EPG")],
        input_rois=[RoiWeight(roi="EB", weight=157749)],
        upstream=[Partner(type="ER4m", n_cells=10, total_weight=14903, **{"class": None})],
        downstream=[Partner(type="Delta7", n_cells=42, total_weight=17286, **{"class": None})],
    )


def _role(conf, basis):
    return FunctionalRole(function="f", evidence_type="connectivity",
                          connectivity_basis=basis, confidence=conf)


def test_fingerprint_numbers_collects_weights():
    nums = fingerprint_numbers(_fp())
    assert {157749, 14903, 17286}.issubset(nums)


def test_valid_number_not_flagged():
    a = HypothesisAnalysis(functional_roles=[
        _role("high", ["receives 14903 synapses from ER4m"])])
    note = check_analysis(a, _fp())
    assert note == ""
    assert a.functional_roles[0].confidence == "high"  # untouched


def test_fabricated_number_downgraded_and_flagged():
    a = HypothesisAnalysis(functional_roles=[
        _role("high", ["receives 99999 synapses from ER4m"])])
    note = check_analysis(a, _fp())
    assert "99999" in note
    assert a.functional_roles[0].confidence == "medium"  # high -> medium


def test_years_and_small_counts_ignored():
    a = HypothesisAnalysis(functional_roles=[
        _role("high", ["a 2024 study; n=8 cells; 46 of them"])])
    assert check_analysis(a, _fp()) == ""
    assert a.functional_roles[0].confidence == "high"


def test_replication_weights_accepted():
    a = HypothesisAnalysis(functional_roles=[
        _role("high", ["21202 synapses (replicates across male-cns)"])])
    repl = {"replicated_partner_types": [{"weights": {"male-cns:v1.0": 21202}}]}
    assert check_analysis(a, _fp(), repl) == ""
    assert a.functional_roles[0].confidence == "high"
