import pytest

from kqt import data
from conftest import FakeTok


def test_split_docs_disjoint_and_deterministic():
    docs = [f"doc{i}" for i in range(100)]
    a, b, c = data.split_docs(docs, seed=0)
    assert (len(a), len(b), len(c)) == (40, 30, 30)
    assert not (set(a) & set(b)) and not (set(a) & set(c)) and not (set(b) & set(c))
    assert data.split_docs(docs, seed=0) == (a, b, c)
    assert data.split_docs(docs, seed=1)[0] != a


def test_chunk_tokens_shape_and_error():
    tok = FakeTok()
    x = data.chunk_tokens(["hello world " * 20] * 5, tok, n=4, seqlen=16)
    assert tuple(x.shape) == (4, 16)
    with pytest.raises(ValueError):
        data.chunk_tokens(["hi"], tok, n=4, seqlen=16)


def test_item_converters():
    k = data.kmmlu_item({"question": "q", "answer": "2", "A": "a", "B": "b", "C": "c", "D": "d", "Category": "x"})
    assert k == {"question": "q", "choices": ["a", "b", "c", "d"], "answer": 1}
    m = data.mmlu_item({"question": "q", "subject": "s", "choices": ["a", "b", "c", "d"], "answer": 3})
    assert m == {"question": "q", "choices": ["a", "b", "c", "d"], "answer": 3}


def test_sample_items():
    items = list(range(50))
    assert data.sample_items(items, 10, seed=0) == data.sample_items(items, 10, seed=0)
    assert len(data.sample_items(items, 10, seed=0)) == 10
    assert data.sample_items(items, 100, seed=0) == items


def test_subject_map_has_ten_pairs():
    assert len(data.KMMLU_TO_MMLU) == 10
