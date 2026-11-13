"""E1: Korean vs English quantization loss. Writes results/e1_summary.md and results/e1_flip.png."""
import glob
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    if not n:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def tax_table(records):
    rows = []
    for r in records:
        c, res = r["cfg"], r["results"]
        row = {k: c[k] for k in ("model", "method", "bits", "calib_lang")}
        for lang in ("ko", "en"):
            for m in ("acc", "flip", "kl", "ppl"):
                row[f"{lang}_{m}"] = res[lang].get(m, float("nan"))
            n = row[f"{lang}_n"] = res[lang]["n_mcq"]
            row[f"{lang}_flip_lo"], row[f"{lang}_flip_hi"] = wilson(row[f"{lang}_flip"] * n, n)
        rows.append(row)
    df = pd.DataFrame(rows)
    base = df[df.method == "fp16"].set_index("model")
    for lang in ("ko", "en"):
        df[f"{lang}_ppl_ratio"] = df[f"{lang}_ppl"] / df.model.map(base[f"{lang}_ppl"])
        df[f"{lang}_fp16_acc"] = df.model.map(base[f"{lang}_acc"])
        df[f"{lang}_acc_drop"] = df[f"{lang}_fp16_acc"] - df[f"{lang}_acc"]
    # too few English flips -> unstable ratio
    df["flip_tax"] = [k / e if e * n >= 10 else math.nan for k, e, n in zip(df.ko_flip, df.en_flip, df.en_n)]
    return df.drop(columns=["ko_ppl", "en_ppl"])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = []
    for f in glob.glob(str(ROOT / "results/kqt-e1-*/out/*.json")):
        with open(f, encoding="utf-8") as fh:
            recs.append(json.load(fh))
    recs = [r for r in recs if not r["cfg"].get("smoke")]
    assert recs, "no non-smoke result records found under results/kqt-e1-*/out/"
    df = tax_table(recs).sort_values(["model", "method", "bits", "calib_lang"], na_position="first")
    (ROOT / "results/e1_summary.md").write_text(df.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    models = sorted(df.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), dpi=140, squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = df[(df.model == model) & (df.method != "fp16")]
        for (method, cl), g in sub.groupby(["method", sub.calib_lang.fillna("-")]):
            g = g.sort_values("bits")
            ax.plot(g.bits, g.ko_flip, "o-", label=f"{method}/{cl} ko")
            ax.plot(g.bits, g.en_flip, "x--", label=f"{method}/{cl} en")
        ax.set_title(model.split("/")[-1])
        ax.set_xlabel("weight bits")
        ax.set_ylabel("flip rate vs FP16")
        ax.set_xticks([3, 4, 8])
        ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "results/e1_flip.png")
    print(df.to_string())


if __name__ == "__main__":
    main()
