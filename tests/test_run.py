import json
import sys
from pathlib import Path

from kqt import grids
import run


def test_e1_grid_size_and_order():
    cfgs = grids.e1("m")
    assert len(cfgs) == 19  # fp16 + rtn/hqq x3 bits + gptq/awq x3 bits x2 calib langs
    assert cfgs[0]["method"] == "fp16"
    assert {c["calib_lang"] for c in cfgs if c["method"] in ("gptq", "awq")} == {"en", "ko"}
    assert {c["calib_lang"] for c in cfgs if c["method"] in ("rtn", "hqq")} == {None}


def test_cfg_id_stable_and_order_free():
    a = {"model": "m", "method": "rtn", "bits": 4}
    b = {"bits": 4, "method": "rtn", "model": "m"}
    assert run.cfg_id(a) == run.cfg_id(b) and len(run.cfg_id(a)) == 12
    assert run.cfg_id(a) != run.cfg_id({**a, "bits": 3})


def test_pending_skips_finished(tmp_path):
    cfgs = grids.e1("m")[:3]
    (tmp_path / f"{run.cfg_id(cfgs[1])}.json").write_text(json.dumps({"done": True}))
    assert run.pending(cfgs, tmp_path) == [cfgs[0], cfgs[2]]


def test_missing_baseline_puts_fp16_first(tmp_path):
    cfgs = grids.e1("m")
    (tmp_path / f"{run.cfg_id(cfgs[0])}.json").write_text(json.dumps({"done": True}))
    todo = run.pending(cfgs, tmp_path)
    assert cfgs[0] not in todo
    assert run.with_baseline(todo, cfgs, tmp_path, "m")[0] == cfgs[0]


def test_present_baseline_leaves_todo_unchanged(tmp_path):
    cfgs = grids.e1("m")
    for lang in ("ko", "en"):
        (tmp_path / f"base-m-{lang}.pt").write_bytes(b"")
    todo = run.pending(cfgs, tmp_path)
    assert run.with_baseline(todo, cfgs, tmp_path, "m") == todo


def test_smoke_cfgs_marked_and_get_distinct_ids():
    full = grids.e1("m")
    smoke = run.select(full, smoke=True)
    assert len(smoke) == 3 and all(c["smoke"] is True for c in smoke)
    assert {run.cfg_id(c) for c in smoke}.isdisjoint({run.cfg_id(c) for c in full})
    assert run.select(full, smoke=False) == full


def test_tax_table():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.e1_summary import tax_table

    def rec(method, bits, ko_flip, en_flip, ko_ppl, en_ppl):
        r = lambda f, p: {"acc": 0.5, "flip": f, "kl": f / 10, "ppl": p, "n_mcq": 100}
        return {"cfg": {"model": "m", "method": method, "bits": bits, "calib_lang": None},
                "results": {"ko": r(ko_flip, ko_ppl), "en": r(en_flip, en_ppl)}}

    df = tax_table([rec("fp16", 16, 0, 0, 10, 5), rec("rtn", 4, 0.2, 0.1, 12, 5.5)])
    row = df[df.method == "rtn"].iloc[0]
    assert abs(row.flip_tax - 2.0) < 1e-9
    assert abs(row.ko_ppl_ratio - 1.2) < 1e-9 and abs(row.en_ppl_ratio - 1.1) < 1e-9
