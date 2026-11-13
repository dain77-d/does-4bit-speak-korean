"""Apply kqt.quant to every Linear inside the decoder layers (lm_head and embeddings stay FP)."""
import torch
import torch.nn as nn

from kqt import quant

LAYER_PATHS = ("model.layers", "transformer.h")  # Qwen2 / EXAONE


def get_layers(model) -> nn.ModuleList:
    for path in LAYER_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"decoder layers not found under {LAYER_PATHS}")


def linears(layer) -> dict:
    return {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}


class _Stop(Exception):
    pass


class _Catcher(nn.Module):
    """Stands in for layer 0 to record its inputs; forwards attribute reads (e.g. attention_type) to the real layer."""

    def __init__(self, layer, inps, kw):
        super().__init__()
        self.layer, self.inps, self.kw = layer, inps, kw

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, x, *args, **kwargs):
        self.inps.append(x)
        self.kw.update(kwargs)
        raise _Stop


@torch.no_grad()
def capture_layer0_inputs(model, calib_ids, batch_size=8):
    layers = get_layers(model)
    first, inps, kw = layers[0], [], {}
    layers[0] = _Catcher(first, inps, kw)
    try:
        for i in range(0, len(calib_ids), batch_size):
            try:
                model(calib_ids[i:i + batch_size].to(model.device), use_cache=False)
            except _Stop:
                pass
    finally:
        layers[0] = first
    return inps, kw


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def quantize_model(model, method, bits, group=128, calib_ids=None, x_sample=512):
    if method == "fp16":
        return model
    layers = get_layers(model)
    if method in ("rtn", "hqq"):
        fn = quant.rtn if method == "rtn" else quant.hqq
        for layer in layers:
            for m in linears(layer).values():
                m.weight.data = fn(m.weight.data, bits, group)
        return model
    if method not in ("gptq", "awq"):
        raise ValueError(f"unknown method {method}")
    if calib_ids is None:
        raise ValueError(f"{method} needs calib_ids")
    inps, kw = capture_layer0_inputs(model, calib_ids)
    for layer in layers:
        lins = linears(layer)
        stats = {n: {"H": None, "amean": None, "n": 0, "X": []} for n in lins}

        def hook(name):
            def f(mod, args, out):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                st = stats[name]
                if st["H"] is None:
                    st["H"] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                    st["amean"] = torch.zeros(x.shape[1], device=x.device)
                st["H"] += x.T @ x
                st["amean"] += x.abs().sum(0)
                st["n"] += x.shape[0]
                if sum(t.shape[0] for t in st["X"]) < x_sample:
                    st["X"].append(x[torch.randperm(x.shape[0], device=x.device)[:64]])
            return f

        handles = [m.register_forward_hook(hook(n)) for n, m in lins.items()]
        for x in inps:
            layer(x, **kw)
        for h in handles:
            h.remove()
        # ponytail: all Linears in a block see FP inputs (not true-sequential GPTQ); upgrade if the gap matters.
        for n, m in lins.items():
            st = stats[n]
            if method == "gptq":
                m.weight.data = quant.gptq(m.weight.data, 2 * st["H"] / st["n"], bits, group)
            else:
                m.weight.data = quant.awq(m.weight.data, st["amean"] / st["n"], torch.cat(st["X"]), bits, group)
        inps = [_out(layer(x, **kw)) for x in inps]
    return model
