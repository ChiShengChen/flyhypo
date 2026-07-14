"""FlyWire adapter — parser proven against a synthetic Codex-style export."""

import flyhypo.flywire as fw


def _write_export(tmp_path):
    (tmp_path / "classification.csv").write_text(
        "root_id,hemibrain_type\n"
        "100,EPG\n101,EPG\n200,ER4m\n201,ER4m\n300,PEN_a(PEN1)\n400,Delta7\n")
    (tmp_path / "connections.csv").write_text(
        "pre_root_id,post_root_id,syn_count\n"
        "200,100,50\n201,101,40\n300,100,30\n100,400,60\n101,400,20\n100,101,10\n")
    fw._load.cache_clear()
    return str(tmp_path)


def test_available_and_partners(tmp_path):
    d = _write_export(tmp_path)
    assert fw.flywire_available(d)

    fp = fw.flywire_fingerprint("EPG", top_k=15, ddir=d)
    assert fp.dataset == "flywire"
    assert len(fp.resolved) == 2  # root_ids 100, 101

    up = {p.type: (p.total_weight, p.n_cells) for p in fp.upstream}
    assert up["ER4m"] == (90, 2)          # 50 + 40, two presynaptic cells
    assert up["PEN_a(PEN1)"] == (30, 1)

    down = {p.type: p.total_weight for p in fp.downstream}
    assert down["Delta7"] == 80           # 60 + 20


def test_missing_data_degrades(tmp_path):
    fw._load.cache_clear()
    fp = fw.flywire_fingerprint("EPG", ddir=str(tmp_path))  # empty dir
    assert fp.dataset == "flywire" and not fp.found
    assert "not configured" in (fp.notes or "")


def test_type_absent(tmp_path):
    d = _write_export(tmp_path)
    fp = fw.flywire_fingerprint("MBON01", ddir=d)
    assert not fp.found and "not found" in (fp.notes or "")
