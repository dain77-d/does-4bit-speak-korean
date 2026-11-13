"""E1: Korean vs English quantization loss. Writes results/e1_summary.md and results/e1_flip.png."""
import glob
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def tax_table(records):
    rows = []
    for r in records:
        c, res = r["cfg"], r["results"]
        row = {k: c[k] for k in ("model", "method", "bits", "calib_lang")}
        for lang in ("ko", "en"):
            for m in ("acc", "flip", "kl", "ppl"):
                row[f"{lang}_{m}"] = res[lang].get(m, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)
    base = df[df.method == "fp16"].set_index("model")
    for lang in ("ko", "en"):
        df[f"{lang}_ppl_ratio"] = df[f"{lang}_ppl"] / df.model.map(base[f"{lang}_ppl"])
    df["flip_tax"] = [k / e if e else math.nan for k, e in zip(df.ko_flip, df.en_flip)]
    return df.drop(columns=["ko_ppl", "en_ppl"])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.load(open(f, encoding="utf-8")) for f in glob.glob(str(ROOT / "results/kqt-e1-*/out/*.json"))]
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
