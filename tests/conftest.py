import pytest
import torch
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


class FakeTok:
    """Character-level tokenizer over a 100-id vocab: enough for shape/plumbing tests."""
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token = None
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % 98) for c in text]

    def __call__(self, texts, return_tensors="pt", padding=True):
        seqs = [self.encode(t) for t in texts]
        n = max(map(len, seqs))
        ids, mask = [], []
        for s in seqs:
            pad = [self.eos_token_id] * (n - len(s))
            m = [0] * len(pad) + [1] * len(s) if self.padding_side == "left" else [1] * len(s) + [0] * len(pad)
            ids.append(pad + s if self.padding_side == "left" else s + pad)
            mask.append(m)
        return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)})
