def e1(model):
    cfgs = [dict(model=model, method="fp16", bits=16, group=128, calib_lang=None, seed=0)]
    for method in ("rtn", "hqq", "gptq", "awq"):
        for bits in (8, 4, 3):
            for cl in (("en", "ko") if method in ("gptq", "awq") else (None,)):
                cfgs.append(dict(model=model, method=method, bits=bits, group=128, calib_lang=cl, seed=0))
    return cfgs
