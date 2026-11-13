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


def _shrink_lp(x: torch.Tensor, beta: float, p: float = 0.7) -> torch.Tensor:
    """Generalized soft-thresholding for the l_p (p<1) proximal step. |x|^(p-1) is inf at 0; relu maps that to 0."""
    return torch.sign(x) * torch.relu(x.abs() - (1.0 / beta) * x.abs().pow(p - 1))


def hqq(W: torch.Tensor, bits: int, group: int = 128, iters: int = 20, beta: float = 10.0, kappa: float = 1.01):
    """Half-Quadratic Quantization (Badri & Shaji 2023): calibration-free zero-point optimisation under l_p error."""
    Wg = _groups(W, group)
    qmax = 2 ** bits - 1
    wmin, wmax = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    inv_s = qmax / (wmax - wmin).clamp(min=1e-8)
    zero = -wmin * inv_s
    best_err, best_zero = float("inf"), zero
    for _ in range(iters):
        Wq = torch.round(Wg * inv_s + zero).clamp(0, qmax)
        Wr = (Wq - zero) / inv_s
        err = (Wg - Wr).abs().mean().item()
        if err >= best_err:
            break
        best_err, best_zero = err, zero
        We = _shrink_lp(Wg - Wr, beta)
        zero = (Wq - (Wg - We) * inv_s).mean(-1, keepdim=True)
        beta *= kappa
    Wq = torch.round(Wg * inv_s + best_zero).clamp(0, qmax)
    return ((Wq - best_zero) / inv_s).reshape(W.shape).to(W.dtype)
