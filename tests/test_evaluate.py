import math

import torch

from conftest import FakeTok
from kqt import evaluate


ITEMS = [{"question": f"질문 {i}", "choices": ["가", "나", "다", "라"], "answer": i % 4} for i in range(6)]


def test_mcq_logprobs_shape_and_normalised(tiny_model):
    lp = evaluate.mcq_logprobs(tiny_model, FakeTok(), ITEMS, "ko", batch_size=4)
    assert tuple(lp.shape) == (6, 4)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(6), atol=1e-5)


def test_mcq_logprobs_independent_of_batching(tiny_model):
    questions = ["짧다", "조금 더 긴 질문", "질문" * 5, "질문" * 15, "질문" * 40]
    items = [{"question": q, "choices": ["가", "나", "다", "라"], "answer": i % 4} for i, q in enumerate(questions)]
    single = evaluate.mcq_logprobs(tiny_model, FakeTok(), items, "ko", batch_size=1)
    batched = evaluate.mcq_logprobs(tiny_model, FakeTok(), items, "ko", batch_size=5)
    assert torch.allclose(single, batched, atol=1e-4)


def test_mcq_metrics_identity_and_flip():
    lp = torch.log_softmax(torch.tensor([[3.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 3.0, 0]]), -1)
    ans = torch.tensor([0, 1, 3])
    same = evaluate.mcq_metrics(lp, ans, base_lp=lp)
    assert abs(same["acc"] - 2 / 3) < 1e-6 and same["flip"] == 0 and abs(same["kl"]) < 1e-6
    other = torch.log_softmax(torch.tensor([[0, 3.0, 0, 0], [0, 3.0, 0, 0], [0, 0, 0, 3.0]]), -1)
    m = evaluate.mcq_metrics(other, ans, base_lp=lp)
    assert abs(m["flip"] - 2 / 3) < 1e-6  # item0 right->wrong, item2 wrong->right
    assert m["kl"] > 0


def test_perplexity_finite_and_matches_uniform_bound(tiny_model):
    x = torch.randint(2, 100, (3, 20))
    ppl = evaluate.perplexity(tiny_model, x, batch_size=2)
    assert math.isfinite(ppl) and 1 < ppl < 1000
