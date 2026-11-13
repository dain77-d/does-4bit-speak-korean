"""Zero-shot 4-choice scoring by next-letter log-prob, flip rate / KL vs FP16 (Dutta et al. 2024), perplexity."""
import math

import torch
import torch.nn.functional as F

LETTERS = ["A", "B", "C", "D"]
PROMPT = {
    "ko": "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n정답:",
    "en": "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:",
}


def letter_ids(tok):
    return [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]


@torch.no_grad()
def mcq_logprobs(model, tok, items, lang, batch_size=8):
    ids = letter_ids(tok)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(items), batch_size):
        texts = [PROMPT[lang].format(q=it["question"], a=it["choices"][0], b=it["choices"][1],
                                     c=it["choices"][2], d=it["choices"][3]) for it in items[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos).logits
        out.append(F.log_softmax(logits[:, -1, ids].float(), dim=-1).cpu())
    return torch.cat(out)


def mcq_metrics(lp, answers, base_lp=None):
    pred = lp.argmax(-1)
    res = {"acc": (pred == answers).float().mean().item()}
    if base_lp is not None:
        base_ok = base_lp.argmax(-1) == answers
        res["flip"] = ((pred == answers) != base_ok).float().mean().item()
        res["kl"] = (base_lp.exp() * (base_lp - lp)).sum(-1).mean().item()
    return res


@torch.no_grad()
def perplexity(model, chunks, batch_size=4):
    nll, n = 0.0, 0
    for i in range(0, len(chunks), batch_size):
        x = chunks[i:i + batch_size].to(model.device)
        logits = model(x).logits[:, :-1].float()
        nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum").item()
        n += x[:, 1:].numel()
    return math.exp(nll / n)
