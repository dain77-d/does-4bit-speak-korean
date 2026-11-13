import copy

import pytest
import torch

from kqt.model import get_layers, linears, quantize_model


def levels(W, group):
    Wg = W.reshape(W.shape[0], -1, group)
    return max(len(torch.unique(Wg[r, c])) for r in range(Wg.shape[0]) for c in range(Wg.shape[1]))


def test_get_layers_and_linears(tiny_model):
    layers = get_layers(tiny_model)
    assert len(layers) == 2
    assert len(linears(layers[0])) == 7  # q k v o gate up down


def test_fp16_is_noop(tiny_model):
    before = copy.deepcopy(tiny_model.state_dict())
    quantize_model(tiny_model, "fp16", 16)
    for k, v in tiny_model.state_dict().items():
        assert torch.equal(v, before[k])


@pytest.mark.parametrize("method", ["rtn", "hqq", "gptq", "awq"])
def test_methods_quantize_all_linears_and_keep_head(tiny_model, method):
    head = tiny_model.lm_head.weight.clone()
    orig = [{n: m.weight.data.clone() for n, m in linears(l).items()} for l in get_layers(tiny_model)]
    calib = torch.randint(2, 100, (8, 32))
    quantize_model(tiny_model, method, bits=3, group=32, calib_ids=calib)
    for layer, o in zip(get_layers(tiny_model), orig):
        for n, m in linears(layer).items():
            if method == "awq":
                # AWQ folds 1/s into the weight, so values are off the grid; just check it changed
                assert not torch.equal(m.weight.data, o[n])
            else:
                assert levels(m.weight.data, 32) <= 8
    assert torch.equal(tiny_model.lm_head.weight, head)
    out = tiny_model(torch.randint(2, 100, (2, 16))).logits
    assert torch.isfinite(out).all()


def test_gptq_closer_to_fp_than_rtn(tiny_model):
    calib = torch.randint(2, 100, (16, 32))
    x = torch.randint(2, 100, (4, 32))
    ref = tiny_model(x).logits
    errs = {}
    for method in ("rtn", "gptq"):
        m = copy.deepcopy(tiny_model)
        quantize_model(m, method, bits=3, group=32, calib_ids=calib)
        errs[method] = (m(x).logits - ref).pow(2).mean().item()
    assert errs["gptq"] < errs["rtn"]
