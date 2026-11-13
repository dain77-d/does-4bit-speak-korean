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
