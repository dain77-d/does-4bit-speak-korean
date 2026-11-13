"""Data loading and deterministic splits. load_* need network + `datasets` (run on Kaggle)."""
import random

import torch

KMMLU_TO_MMLU = {
    "Computer-Science": "college_computer_science",
    "Economics": "high_school_macroeconomics",
    "Biology": "college_biology",
    "Chemistry": "college_chemistry",
    "Law": "professional_law",
    "Math": "high_school_mathematics",
    "Korean-History": "high_school_world_history",
    "Psychology": "professional_psychology",
    "Health": "clinical_knowledge",
    "Accounting": "professional_accounting",
}


def split_docs(docs, seed=0, fracs=(0.4, 0.3, 0.3)):
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    a = int(len(docs) * fracs[0])
    b = a + int(len(docs) * fracs[1])
    pick = lambda ids: [docs[i] for i in ids]
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])


def chunk_tokens(texts, tok, n, seqlen):
    need, ids = n * seqlen, []
    for t in texts:
        ids.extend(tok.encode(t, add_special_tokens=False) + [tok.eos_token_id])
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise ValueError(f"need {need} tokens, got {len(ids)}")
    return torch.tensor(ids[:need]).reshape(n, seqlen)


def sample_items(items, n, seed):
    return list(items) if len(items) <= n else random.Random(seed).sample(list(items), n)


def kmmlu_item(row):
    return {"question": row["question"], "choices": [row[k] for k in "ABCD"], "answer": int(row["answer"]) - 1}


def mmlu_item(row):
    return {"question": row["question"], "choices": list(row["choices"]), "answer": int(row["answer"])}


def load_mcq(lang, n_per_subject=100, seed=0):
    from datasets import load_dataset

    items = []
    for ko, en in KMMLU_TO_MMLU.items():
        if lang == "ko":
            rows, conv = load_dataset("HAERAE-HUB/KMMLU", ko, split="test"), kmmlu_item
        else:
            rows, conv = load_dataset("cais/mmlu", en, split="test"), mmlu_item
        items += sample_items([conv(r) for r in rows], n_per_subject, seed)
    return items


def load_wiki(lang, n_docs=3000):
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    return [r["text"] for _, r in zip(range(n_docs), ds) if len(r["text"]) > 500]
