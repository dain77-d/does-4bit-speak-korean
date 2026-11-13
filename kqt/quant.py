"""Group-wise fake quantization. Every public function returns a dequantized weight with W's shape and dtype."""
import torch


def _minmax_params(w: torch.Tensor, bits: int):
    """Asymmetric min-max parameters per group along the last dim."""
    qmax = 2 ** bits - 1
    wmin = w.amin(dim=-1, keepdim=True)
    wmax = w.amax(dim=-1, keepdim=True)
    scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
    zero = torch.round(-wmin / scale)
    return scale, zero, qmax


def _fake(w, scale, zero, qmax):
    q = torch.clamp(torch.round(w / scale) + zero, 0, qmax)
    return (q - zero) * scale


def _groups(W: torch.Tensor, group: int) -> torch.Tensor:
    out, inp = W.shape
    assert inp % group == 0, f"in_features {inp} not divisible by group {group}"
    return W.float().reshape(out, inp // group, group)


def rtn(W: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    Wg = _groups(W, group)
    s, z, qmax = _minmax_params(Wg, bits)
    return _fake(Wg, s, z, qmax).reshape(W.shape).to(W.dtype)
