import json

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
