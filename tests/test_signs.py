"""NT → predicted synapse sign mapping."""

from flyhypo.connectome import nt_to_sign


def test_sign_mapping():
    assert nt_to_sign("acetylcholine") == "excitatory"
    assert nt_to_sign("GABA") == "inhibitory"          # case-insensitive
    assert nt_to_sign("glutamate") == "inhibitory"     # fly CNS heuristic
    assert nt_to_sign("dopamine") == "modulatory"
    assert nt_to_sign("serotonin") == "modulatory"


def test_unknown_and_empty():
    assert nt_to_sign(None) is None
    assert nt_to_sign("") is None
    assert nt_to_sign("mystery-transmitter") is None
