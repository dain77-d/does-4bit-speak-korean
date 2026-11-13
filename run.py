"""Run a grid of quantization configs for one model; one JSON per config, resumable."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from kqt import data, evaluate, grids
from kqt.model import quantize_model


def cfg_id(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def pending(cfgs, out_dir):
    return [c for c in cfgs if not (Path(out_dir) / f"{cfg_id(c)}.json").exists()]


def with_baseline(todo, cfgs, out_dir, slug):
    # the fp16 run writes the reference probabilities; any later config needs them
    if all((Path(out_dir) / f"base-{slug}-{lang}.pt").exists() for lang in ("ko", "en")):
        return todo
    fp16 = next((c for c in cfgs if c["method"] == "fp16"), None)
    if fp16 is None:
        return todo
    return [fp16] + [c for c in todo if c != fp16]


def select(cfgs, smoke):
    if not smoke:
        return cfgs
    keep = {("fp16", 16, None), ("rtn", 4, None), ("gptq", 4, "ko")}
    return [{**c, "smoke": True} for c in cfgs if (c["method"], c["bits"], c["calib_lang"]) in keep]


def load_baseline(path, n_items):
    base = torch.load(path)
    if base.shape[0] != n_items:
        raise ValueError(f"{path} has {base.shape[0]} rows, expected {n_items}; stale baseline, delete it and rerun")
    return base


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda" if torch.cuda.is_available() else None).eval()
    return model, tok


def prepare_data(tok, seed, smoke):
    n_calib, n_ppl, n_mcq = (8, 2, 5) if smoke else (128, 32, 100)
    out = {}
    for lang in ("ko", "en"):
        _, calib, ev = data.split_docs(data.load_wiki(lang, 300 if smoke else 3000), seed)
        out[lang] = {"calib": data.chunk_tokens(calib, tok, n_calib, 512),
                     "ppl": data.chunk_tokens(ev, tok, n_ppl, 512),
                     "mcq": data.load_mcq(lang, n_mcq, seed)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--grid", default="e1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny data + 3 configs, to check the pipeline")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfgs = select(getattr(grids, args.grid)(args.model), args.smoke)
    slug = args.model.split("/")[-1]
    todo = with_baseline(pending(cfgs, out), cfgs, out, slug)
    if not todo:
        return
    import transformers

    model, tok = load_model(args.model)
    orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    d = prepare_data(tok, seed=0, smoke=args.smoke)
    for cfg in todo:
        t0 = time.time()
        torch.manual_seed(cfg["seed"])
        model.load_state_dict(orig)
        calib = d[cfg["calib_lang"]]["calib"] if cfg["calib_lang"] else None
        quantize_model(model, cfg["method"], cfg["bits"], cfg["group"], calib)
        res = {}
        for lang in ("ko", "en"):
            lp = evaluate.mcq_logprobs(model, tok, d[lang]["mcq"], lang)
            base_path = out / f"base-{slug}-{lang}.pt"
            if cfg["method"] == "fp16":
                torch.save(lp, base_path)
            answers = torch.tensor([it["answer"] for it in d[lang]["mcq"]])
            res[lang] = evaluate.mcq_metrics(lp, answers, load_baseline(base_path, len(answers)))
            res[lang]["ppl"] = evaluate.perplexity(model, d[lang]["ppl"])
            res[lang]["n_mcq"] = len(answers)
        record = {"cfg": cfg, "id": cfg_id(cfg), "results": res, "seconds": round(time.time() - t0, 1),
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        (out / f"{cfg_id(cfg)}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
        print(json.dumps({"id": record["id"], **{k: cfg[k] for k in ("method", "bits", "calib_lang")}, **res}))


if __name__ == "__main__":
    main()
