import pytest
import torch

from kqt import quant


def heavy_tailed(out=32, inp=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out, inp, generator=g)
    W[:, :4] *= 8  # a few outlier input channels, like real LLM weights
    return W


def levels_per_group(W, group):
    Wg = W.reshape(W.shape[0], -1, group)
    return max(len(torch.unique(Wg[r, c])) for r in range(Wg.shape[0]) for c in range(Wg.shape[1]))


def test_rtn_shape_dtype_and_levels():
    W = heavy_tailed().half()
    Q = quant.rtn(W, bits=4, group=32)
    assert Q.shape == W.shape and Q.dtype == W.dtype
    assert levels_per_group(Q.float(), 32) <= 16


def test_rtn_error_decreases_with_bits():
    W = heavy_tailed()
    errs = [(W - quant.rtn(W, b, 32)).pow(2).mean().item() for b in (3, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


def test_rtn_rejects_bad_group():
    with pytest.raises(AssertionError):
        quant.rtn(torch.randn(4, 100), 4, 32)


def test_hqq_levels_and_not_worse_than_rtn():
    W = heavy_tailed(seed=1)
    Q = quant.hqq(W, bits=4, group=32)
    assert Q.shape == W.shape
    assert levels_per_group(Q, 32) <= 16
    e_hqq = (W - Q).abs().mean().item()
    e_rtn = (W - quant.rtn(W, 4, 32)).abs().mean().item()
    assert e_hqq <= e_rtn * 1.05


def correlated_inputs(n=512, inp=128, seed=2):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, 8, generator=g)
    mix = torch.randn(8, inp, generator=g)
    return base @ mix + 0.1 * torch.randn(n, inp, generator=g)


def test_gptq_beats_rtn_on_output_error():
    W = heavy_tailed(seed=3)
    X = correlated_inputs()
    H = 2 * X.T @ X / X.shape[0]
    Q = quant.gptq(W, H, bits=3, group=32)
    assert levels_per_group(Q, 32) <= 8
    out_err = lambda Wq: (X @ (W - Wq).T).pow(2).mean().item()
    assert out_err(Q) < out_err(quant.rtn(W, 3, 32))


def test_gptq_handles_dead_inputs():
    W = heavy_tailed(seed=4)
    X = correlated_inputs()
    X[:, 5] = 0
    Q = quant.gptq(W, 2 * X.T @ X / X.shape[0], bits=4, group=32)
    assert torch.isfinite(Q).all()


def test_awq_never_worse_than_rtn_on_its_sample():
    W = heavy_tailed(seed=5)
    X = correlated_inputs(seed=6)
    X[:, :4] *= 10  # salient input channels
    Q = quant.awq(W, X.abs().mean(0), X, bits=3, group=32)
    out_err = lambda Wq: (X @ (W - Wq).T).pow(2).mean().item()
    assert out_err(Q) <= out_err(quant.rtn(W, 3, 32)) + 1e-6
    assert Q.shape == W.shape
