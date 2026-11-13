# Plan 1: 기반 코드 + E1(한국어 양자화 세금 측정) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 시뮬레이션 양자화(RTN·HQQ·GPTQ·AWQ), 평가(MCQ 정확도·flip·KL·PPL), Kaggle 실행기를 만들고 E1(모델 2 × 방법 4 × 비트 3 × 보정 언어)을 실행해 한·영 손실 비교표를 만든다.

**Architecture:** `kqt/quant.py`는 가중치 행렬 하나를 받아 역양자화된 같은 모양의 행렬을 돌려주는 순수 함수들이다. `kqt/model.py`가 이를 디코더 층 단위로 모델에 적용한다(GPTQ·AWQ는 층별 보정 통계 수집). `run.py`가 실험 설정 목록을 실행해 설정 해시별 JSON을 남기고, `launch/kaggle_job.py`가 이를 Kaggle T4 작업으로 올리고 결과를 회수한다.

**Tech Stack:** Python 3.10+, PyTorch, transformers, datasets(Kaggle에서만), pytest, Kaggle CLI 2.x, pandas·matplotlib(분석)

**Spec:** `docs/superpowers/specs/2026-11-14-korean-quant-tax-design.md`

**이 계획의 범위:** 스펙 10절 일정의 1주차(기반 코드 + E1)와 E1 분석. E2~E5(neurons·protect·kv·speed)는 E1 결과를 본 뒤 Plan 2·3으로 작성한다. HAE-RAE는 데이터 형식 확인이 필요해 Plan 2로 미룬다(E1은 KMMLU·MMLU·PPL로 판정).

## Global Constraints

- 커밋 작성자: `dain77-d <149990219+dain77-d@users.noreply.github.com>`. 커밋 메시지에 Co-Authored-By·Claude 관련 트레일러 금지.
- 양자화: 가중치 전용, 비대칭 min-max, 그룹 크기 128, 비트 {8, 4, 3}. lm_head와 임베딩은 양자화하지 않는다.
- 보정: 128개 × 512토큰, 위키 보정 분할에서만. 뉴런 탐지·보정·PPL 평가는 서로 다른 문서 분할.
- 모델: `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` (`trust_remote_code=True`, FP16).
- 데이터: `HAERAE-HUB/KMMLU`(answer는 1~4 문자열), `cais/mmlu`(answer는 0~3 정수), `wikimedia/wikipedia` `20231101.ko`/`20231101.en`. 모든 표본 추출은 고정 시드.
- KMMLU는 CC-BY-ND: 문항 원문을 저장소에 커밋하지 않는다(점수만).
- 단위 테스트는 GPU·네트워크 없이 로컬 CPU에서 통과해야 한다.
- 모든 결과 JSON에 cfg, 모델 이름, torch/transformers 버전, 시드, 실행 시간을 기록한다.
- Kaggle 사용자명 `dainjuser`, GPU 작업은 private.

## File Structure

```
does-4bit-speak-korean/
  pyproject.toml            패키지·pytest 설정
  requirements.txt          로컬 개발용
  requirements-kaggle.txt   Kaggle 작업에서 추가 설치할 것
  .gitignore
  kqt/__init__.py
  kqt/quant.py              rtn, hqq, gptq, awq (행렬 단위 순수 함수)
  kqt/model.py              get_layers, linears, capture_layer0_inputs, quantize_model
  kqt/data.py               split_docs, chunk_tokens, sample_items, kmmlu_item, mmlu_item, load_mcq, load_wiki
  kqt/evaluate.py           letter_ids, mcq_logprobs, mcq_metrics, perplexity
  kqt/grids.py              e1(model) 실험 설정 목록
  run.py                    설정 목록 실행 → results/<id>.json
  launch/kaggle_job.py      Kaggle 작업 생성·제출·대기·회수
  analysis/e1_summary.py    E1 표·그래프
  tests/conftest.py         tiny_model 픽스처, FakeTok
  tests/test_quant.py
  tests/test_model.py
  tests/test_data.py
  tests/test_evaluate.py
  tests/test_run.py
  tests/test_launch.py
```

---

### Task 0: 저장소 연결과 패키지 뼈대

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `requirements-kaggle.txt`, `.gitignore`, `kqt/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: 픽스처 `tiny_model`(transformers `Qwen2ForCausalLM`, hidden 64, intermediate 128, 2층, vocab 100, FP32, eval 모드), `FakeTok` 클래스(`encode`, `__call__`, `pad_token`, `eos_token`, `eos_token_id`, `padding_side`).

- [ ] **Step 1: GitHub 저장소 연결** (사용자가 `dain77-d/does-4bit-speak-korean`을 만든 뒤)

```bash
git remote add origin https://github.com/dain77-d/does-4bit-speak-korean.git
git push -u origin main
```

- [ ] **Step 2: 설정 파일 작성**

`pyproject.toml`:
```toml
[project]
name = "kqt"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools]
packages = ["kqt"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`requirements.txt`:
```
torch>=2.1
transformers>=4.45
pytest>=8
pandas>=2
matplotlib>=3.8
```

`requirements-kaggle.txt`:
```
datasets>=2.19
```

`.gitignore`:
```
__pycache__/
*.egg-info/
.pytest_cache/
results/**/*.pt
jobs/
```

`kqt/__init__.py`: 빈 파일.

`tests/conftest.py`:
```python
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
```

- [ ] **Step 3: 로컬 개발 도구 설치** (로컬에는 torch 2.14 CPU, transformers 5.19가 이미 있고 pytest가 없음)

Run: `python -m pip install --user -q pytest pandas tabulate matplotlib`

- [ ] **Step 4: 픽스처가 동작하는지 확인**

Run: `python -c "import sys; sys.path.insert(0,'tests'); from conftest import FakeTok; print(FakeTok()(['ab','c']).input_ids.shape)"`
Expected: `torch.Size([2, 2])`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt requirements-kaggle.txt .gitignore kqt/__init__.py tests/conftest.py
git commit -m "패키지 뼈대와 테스트 픽스처"
```

---

### Task 1: RTN 그룹 양자화

**Files:**
- Create: `kqt/quant.py`
- Test: `tests/test_quant.py`

**Interfaces:**
- Produces: `rtn(W: Tensor[out,in], bits: int, group: int = 128) -> Tensor[out,in]` (W와 같은 dtype). 내부 헬퍼 `_minmax_params(w, bits) -> (scale, zero, qmax)`, `_fake(w, scale, zero, qmax) -> Tensor`, `_groups(W, group) -> Tensor[out, in//group, group]`(float32).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_quant.py`:
```python
import pytest
import torch

from kqt import quant


def heavy_tailed(out=32, inp=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out, inp, generator=g)
    W[:, :4] *= 8  # a few outlier input channels, like real LLM weights
    return W


def levels_per_group(W, group):
    Wg = W.reshape(W.shape[0], -1, group)
    return max(len(torch.unique(Wg[r, c])) for r in range(Wg.shape[0]) for c in range(Wg.shape[1]))


def test_rtn_shape_dtype_and_levels():
    W = heavy_tailed().half()
    Q = quant.rtn(W, bits=4, group=32)
    assert Q.shape == W.shape and Q.dtype == W.dtype
    assert levels_per_group(Q.float(), 32) <= 16


def test_rtn_error_decreases_with_bits():
    W = heavy_tailed()
    errs = [(W - quant.rtn(W, b, 32)).pow(2).mean().item() for b in (3, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


def test_rtn_rejects_bad_group():
    with pytest.raises(AssertionError):
        quant.rtn(torch.randn(4, 100), 4, 32)
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_quant.py -v`
Expected: FAIL (`ModuleNotFoundError` 또는 `AttributeError: module 'kqt.quant' has no attribute 'rtn'`)

- [ ] **Step 3: 구현**

`kqt/quant.py`:
```python
"""Group-wise fake quantization. Every public function returns a dequantized weight with W's shape and dtype."""
import torch


def _minmax_params(w: torch.Tensor, bits: int):
    """Asymmetric min-max parameters per group along the last dim."""
    qmax = 2 ** bits - 1
    wmin = w.amin(dim=-1, keepdim=True)
    wmax = w.amax(dim=-1, keepdim=True)
    scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
    zero = torch.round(-wmin / scale)
    return scale, zero, qmax


def _fake(w, scale, zero, qmax):
    q = torch.clamp(torch.round(w / scale) + zero, 0, qmax)
    return (q - zero) * scale


def _groups(W: torch.Tensor, group: int) -> torch.Tensor:
    out, inp = W.shape
    assert inp % group == 0, f"in_features {inp} not divisible by group {group}"
    return W.float().reshape(out, inp // group, group)


def rtn(W: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    Wg = _groups(W, group)
    s, z, qmax = _minmax_params(Wg, bits)
    return _fake(Wg, s, z, qmax).reshape(W.shape).to(W.dtype)
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_quant.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add kqt/quant.py tests/test_quant.py
git commit -m "RTN 그룹 양자화"
```

---

### Task 2: HQQ

**Files:**
- Modify: `kqt/quant.py` (끝에 추가)
- Test: `tests/test_quant.py` (끝에 추가)

**Interfaces:**
- Consumes: `_groups` (Task 1)
- Produces: `hqq(W, bits, group=128, iters=20, beta=10.0, kappa=1.01) -> Tensor` (W와 같은 shape·dtype)

- [ ] **Step 1: 실패하는 테스트 추가**

```python
def test_hqq_levels_and_not_worse_than_rtn():
    W = heavy_tailed(seed=1)
    Q = quant.hqq(W, bits=4, group=32)
    assert Q.shape == W.shape
    assert levels_per_group(Q, 32) <= 16
    e_hqq = (W - Q).abs().mean().item()
    e_rtn = (W - quant.rtn(W, 4, 32)).abs().mean().item()
    assert e_hqq <= e_rtn * 1.05
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_quant.py::test_hqq_levels_and_not_worse_than_rtn -v`
Expected: FAIL (`AttributeError: ... 'hqq'`)

- [ ] **Step 3: 구현** (`kqt/quant.py` 끝에 추가)

```python
def _shrink_lp(x: torch.Tensor, beta: float, p: float = 0.7) -> torch.Tensor:
    """Generalized soft-thresholding for the l_p (p<1) proximal step. |x|^(p-1) is inf at 0; relu maps that to 0."""
    return torch.sign(x) * torch.relu(x.abs() - (1.0 / beta) * x.abs().pow(p - 1))


def hqq(W: torch.Tensor, bits: int, group: int = 128, iters: int = 20, beta: float = 10.0, kappa: float = 1.01):
    """Half-Quadratic Quantization (Badri & Shaji 2023): calibration-free zero-point optimisation under l_p error."""
    Wg = _groups(W, group)
    qmax = 2 ** bits - 1
    wmin, wmax = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    inv_s = qmax / (wmax - wmin).clamp(min=1e-8)
    zero = -wmin * inv_s
    best_err, best_zero = float("inf"), zero
    for _ in range(iters):
        Wq = torch.round(Wg * inv_s + zero).clamp(0, qmax)
        Wr = (Wq - zero) / inv_s
        err = (Wg - Wr).abs().mean().item()
        if err >= best_err:
            break
        best_err, best_zero = err, zero
        We = _shrink_lp(Wg - Wr, beta)
        zero = (Wq - (Wg - We) * inv_s).mean(-1, keepdim=True)
        beta *= kappa
    Wq = torch.round(Wg * inv_s + best_zero).clamp(0, qmax)
    return ((Wq - best_zero) / inv_s).reshape(W.shape).to(W.dtype)
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_quant.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add kqt/quant.py tests/test_quant.py
git commit -m "HQQ 양자화"
```

---

### Task 3: GPTQ

**Files:**
- Modify: `kqt/quant.py` (끝에 추가)
- Test: `tests/test_quant.py` (끝에 추가)

**Interfaces:**
- Consumes: `_minmax_params`, `_fake` (Task 1)
- Produces: `gptq(W, H: Tensor[in,in], bits, group=128, percdamp=0.01) -> Tensor`. H는 입력 X에 대해 `2 XᵀX / n`. 블록 크기 = 그룹 크기.

- [ ] **Step 1: 실패하는 테스트 추가**

```python
def correlated_inputs(n=512, inp=128, seed=2):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, 8, generator=g)
    mix = torch.randn(8, inp, generator=g)
    return base @ mix + 0.1 * torch.randn(n, inp, generator=g)


def test_gptq_beats_rtn_on_output_error():
    W = heavy_tailed(seed=3)
    X = correlated_inputs()
    H = 2 * X.T @ X / X.shape[0]
    Q = quant.gptq(W, H, bits=3, group=32)
    assert levels_per_group(Q, 32) <= 8
    out_err = lambda Wq: (X @ (W - Wq).T).pow(2).mean().item()
    assert out_err(Q) < out_err(quant.rtn(W, 3, 32))


def test_gptq_handles_dead_inputs():
    W = heavy_tailed(seed=4)
    X = correlated_inputs()
    X[:, 5] = 0
    Q = quant.gptq(W, 2 * X.T @ X / X.shape[0], bits=4, group=32)
    assert torch.isfinite(Q).all()
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_quant.py -k gptq -v`
Expected: FAIL (`AttributeError: ... 'gptq'`)

- [ ] **Step 3: 구현** (`kqt/quant.py` 끝에 추가)

```python
def gptq(W: torch.Tensor, H: torch.Tensor, bits: int, group: int = 128, percdamp: float = 0.01) -> torch.Tensor:
    """GPTQ (Frantar et al. 2023). Static min-max params per group, computed when the group's block starts;
    block size equals group size so every group is quantized from already error-corrected weights."""
    dtype = W.dtype
    W = W.float().clone()
    H = H.float().clone()
    cols = W.shape[1]
    assert cols % group == 0, f"in_features {cols} not divisible by group {group}"
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(cols, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, group):
        i2 = i1 + group
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        scale, zero, qmax = _minmax_params(W1, bits)
        for i in range(group):
            w = W1[:, i:i + 1]
            q = _fake(w, scale, zero, qmax)
            Q[:, i1 + i] = q[:, 0]
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err @ Hinv1[i:i + 1, i:]
            Err1[:, i] = err[:, 0]
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q.to(dtype)
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_quant.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add kqt/quant.py tests/test_quant.py
git commit -m "GPTQ 양자화"
```

---

### Task 4: AWQ식 스케일링

**Files:**
- Modify: `kqt/quant.py` (끝에 추가)
- Test: `tests/test_quant.py` (끝에 추가)

**Interfaces:**
- Consumes: `rtn` (Task 1)
- Produces: `awq(W, act_mean: Tensor[in], X: Tensor[n,in], bits, group=128, grid=20) -> Tensor`. `act_mean`은 입력 채널별 평균 |x|, X는 출력 오차 계산용 표본.

- [ ] **Step 1: 실패하는 테스트 추가**

```python
def test_awq_never_worse_than_rtn_on_its_sample():
    W = heavy_tailed(seed=5)
    X = correlated_inputs(seed=6)
    X[:, :4] *= 10  # salient input channels
    Q = quant.awq(W, X.abs().mean(0), X, bits=3, group=32)
    out_err = lambda Wq: (X @ (W - Wq).T).pow(2).mean().item()
    assert out_err(Q) <= out_err(quant.rtn(W, 3, 32)) + 1e-6
    assert Q.shape == W.shape
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_quant.py -k awq -v`
Expected: FAIL (`AttributeError: ... 'awq'`)

- [ ] **Step 3: 구현** (`kqt/quant.py` 끝에 추가)

```python
def awq(W: torch.Tensor, act_mean: torch.Tensor, X: torch.Tensor, bits: int, group: int = 128, grid: int = 20):
    """AWQ-style search (Lin et al. 2024): s = mean|x|^a per input channel, quantize W*s, fold 1/s back.
    a=0 is plain RTN, so the result is never worse than RTN on X."""
    Wf, Xf = W.float(), X.float()
    ref = Xf @ Wf.T
    best_err, best = float("inf"), None
    for k in range(grid):
        s = act_mean.float().clamp(min=1e-5).pow(k / grid)
        s = s / (s.max() * s.min()).sqrt()
        Wq = rtn(Wf * s, bits, group) / s
        err = ((Xf @ Wq.T - ref) ** 2).mean().item()
        if err < best_err:
            best_err, best = err, Wq
    return best.to(W.dtype)
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_quant.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add kqt/quant.py tests/test_quant.py
git commit -m "AWQ식 스케일 탐색"
```

---

### Task 5: 모델 전체에 양자화 적용

**Files:**
- Create: `kqt/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `quant.rtn`, `quant.hqq`, `quant.gptq`, `quant.awq`
- Produces:
  - `get_layers(model) -> nn.ModuleList` (`model.layers` 또는 `transformer.h`)
  - `linears(layer) -> dict[str, nn.Linear]`
  - `capture_layer0_inputs(model, calib_ids: LongTensor[n,seq], batch_size=8) -> (list[Tensor], dict)`
  - `quantize_model(model, method: str, bits: int, group=128, calib_ids=None, x_sample=512) -> model` (제자리 수정). method ∈ {"fp16","rtn","hqq","gptq","awq"}.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_model.py`:
```python
import copy

import pytest
import torch

from kqt.model import get_layers, linears, quantize_model


def levels(W, group):
    Wg = W.reshape(W.shape[0], -1, group)
    return max(len(torch.unique(Wg[r, c])) for r in range(Wg.shape[0]) for c in range(Wg.shape[1]))


def test_get_layers_and_linears(tiny_model):
    layers = get_layers(tiny_model)
    assert len(layers) == 2
    assert len(linears(layers[0])) == 7  # q k v o gate up down


def test_fp16_is_noop(tiny_model):
    before = copy.deepcopy(tiny_model.state_dict())
    quantize_model(tiny_model, "fp16", 16)
    for k, v in tiny_model.state_dict().items():
        assert torch.equal(v, before[k])


@pytest.mark.parametrize("method", ["rtn", "hqq", "gptq", "awq"])
def test_methods_quantize_all_linears_and_keep_head(tiny_model, method):
    head = tiny_model.lm_head.weight.clone()
    calib = torch.randint(2, 100, (8, 32))
    quantize_model(tiny_model, method, bits=3, group=32, calib_ids=calib)
    for layer in get_layers(tiny_model):
        for m in linears(layer).values():
            assert levels(m.weight.data, 32) <= 8
    assert torch.equal(tiny_model.lm_head.weight, head)
    out = tiny_model(torch.randint(2, 100, (2, 16))).logits
    assert torch.isfinite(out).all()


def test_gptq_closer_to_fp_than_rtn(tiny_model):
    calib = torch.randint(2, 100, (16, 32))
    x = torch.randint(2, 100, (4, 32))
    ref = tiny_model(x).logits
    errs = {}
    for method in ("rtn", "gptq"):
        m = copy.deepcopy(tiny_model)
        quantize_model(m, method, bits=3, group=32, calib_ids=calib)
        errs[method] = (m(x).logits - ref).pow(2).mean().item()
    assert errs["gptq"] < errs["rtn"]
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_model.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'kqt.model'`)

- [ ] **Step 3: 구현**

`kqt/model.py`:
```python
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
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_model.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add kqt/model.py tests/test_model.py
git commit -m "모델 단위 양자화 적용"
```

---

### Task 6: 데이터 준비

**Files:**
- Create: `kqt/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `KMMLU_TO_MMLU: dict[str, str]` (10쌍)
  - `split_docs(docs: list[str], seed=0, fracs=(0.4, 0.3, 0.3)) -> (detect, calib, eval)` (서로소)
  - `chunk_tokens(texts, tok, n, seqlen) -> LongTensor[n, seqlen]`
  - `sample_items(items, n, seed) -> list`
  - `kmmlu_item(row) -> {"question","choices","answer"}` (answer 0~3), `mmlu_item(row)` 같은 형식
  - `load_mcq(lang, n_per_subject=100, seed=0) -> list[item]`, `load_wiki(lang, n_docs=3000) -> list[str]` (네트워크, Kaggle 전용)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_data.py`:
```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_data.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: 구현**

`kqt/data.py`:
```python
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
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add kqt/data.py tests/test_data.py
git commit -m "데이터 분할·토큰 청크·벤치마크 로더"
```

---

### Task 7: 평가 지표

**Files:**
- Create: `kqt/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `letter_ids(tok) -> list[int]` (" A"~" D"의 마지막 토큰 id)
  - `mcq_logprobs(model, tok, items, lang, batch_size=8) -> Tensor[n,4]` (4지선다 log-softmax, CPU)
  - `mcq_metrics(lp, answers: LongTensor[n], base_lp=None) -> {"acc"} | {"acc","flip","kl"}`
  - `perplexity(model, chunks: LongTensor[n,seq], batch_size=4) -> float`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_evaluate.py`:
```python
import math

import torch

from conftest import FakeTok
from kqt import evaluate


ITEMS = [{"question": f"질문 {i}", "choices": ["가", "나", "다", "라"], "answer": i % 4} for i in range(6)]


def test_mcq_logprobs_shape_and_normalised(tiny_model):
    lp = evaluate.mcq_logprobs(tiny_model, FakeTok(), ITEMS, "ko", batch_size=4)
    assert tuple(lp.shape) == (6, 4)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(6), atol=1e-5)


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
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: 구현**

`kqt/evaluate.py`:
```python
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
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_evaluate.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add kqt/evaluate.py tests/test_evaluate.py
git commit -m "MCQ 채점, flip·KL, perplexity"
```

---

### Task 8: 실험 설정과 실행기

**Files:**
- Create: `kqt/grids.py`, `run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `data.*`, `evaluate.*`, `model.quantize_model`
- Produces:
  - `grids.e1(model: str) -> list[dict]` (cfg 키: model, method, bits, group, calib_lang, seed), FP16이 첫 번째
  - `run.cfg_id(cfg) -> str` (12자리 sha1, 키 순서 무관)
  - `run.pending(cfgs, out_dir) -> list[dict]` (결과 JSON이 없는 것만)
  - CLI: `python run.py --model <hf id> --grid e1 --out <dir> [--smoke]`. 결과 `<out>/<cfg_id>.json`, 기준 확률 `<out>/base-<slug>-<lang>.pt`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_run.py`:
```python
import json

from kqt import grids
import run


def test_e1_grid_size_and_order():
    cfgs = grids.e1("m")
    assert len(cfgs) == 19  # fp16 + rtn/hqq x3 bits + gptq/awq x3 bits x2 calib langs
    assert cfgs[0]["method"] == "fp16"
    assert {c["calib_lang"] for c in cfgs if c["method"] in ("gptq", "awq")} == {"en", "ko"}
    assert {c["calib_lang"] for c in cfgs if c["method"] in ("rtn", "hqq")} == {None}


def test_cfg_id_stable_and_order_free():
    a = {"model": "m", "method": "rtn", "bits": 4}
    b = {"bits": 4, "method": "rtn", "model": "m"}
    assert run.cfg_id(a) == run.cfg_id(b) and len(run.cfg_id(a)) == 12
    assert run.cfg_id(a) != run.cfg_id({**a, "bits": 3})


def test_pending_skips_finished(tmp_path):
    cfgs = grids.e1("m")[:3]
    (tmp_path / f"{run.cfg_id(cfgs[1])}.json").write_text(json.dumps({"done": True}))
    assert run.pending(cfgs, tmp_path) == [cfgs[0], cfgs[2]]
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_run.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: 구현**

`kqt/grids.py`:
```python
def e1(model):
    cfgs = [dict(model=model, method="fp16", bits=16, group=128, calib_lang=None, seed=0)]
    for method in ("rtn", "hqq", "gptq", "awq"):
        for bits in (8, 4, 3):
            for cl in (("en", "ko") if method in ("gptq", "awq") else (None,)):
                cfgs.append(dict(model=model, method=method, bits=bits, group=128, calib_lang=cl, seed=0))
    return cfgs
```

`run.py`:
```python
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
    cfgs = getattr(grids, args.grid)(args.model)
    if args.smoke:
        cfgs = [c for c in cfgs if (c["method"], c["bits"], c["calib_lang"]) in
                {("fp16", 16, None), ("rtn", 4, None), ("gptq", 4, "ko")}]
    todo = pending(cfgs, out)
    if not todo:
        return
    import transformers

    model, tok = load_model(args.model)
    orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    d = prepare_data(tok, seed=0, smoke=args.smoke)
    slug = args.model.split("/")[-1]
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
            res[lang] = evaluate.mcq_metrics(lp, answers, torch.load(base_path))
            res[lang]["ppl"] = evaluate.perplexity(model, d[lang]["ppl"])
            res[lang]["n_mcq"] = len(answers)
        record = {"cfg": cfg, "id": cfg_id(cfg), "results": res, "seconds": round(time.time() - t0, 1),
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        (out / f"{cfg_id(cfg)}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
        print(json.dumps({"id": record["id"], **{k: cfg[k] for k in ("method", "bits", "calib_lang")}, **res}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_run.py -v`
Expected: 3 passed

- [ ] **Step 5: 전체 테스트**

Run: `pytest -q`
Expected: 모두 통과 (25 passed)

- [ ] **Step 6: Commit**

```bash
git add kqt/grids.py run.py tests/test_run.py
git commit -m "E1 설정 목록과 재개 가능한 실행기"
```

---

### Task 9: Kaggle 실행기

**Files:**
- Create: `launch/kaggle_job.py`
- Test: `tests/test_launch.py`

**Interfaces:**
- Consumes: `run.py` CLI
- Produces:
  - `build(job_dir, slug, commit, run_args: list[str]) -> Path` (job.py + kernel-metadata.json 생성)
  - CLI: `python launch/kaggle_job.py --slug <slug> [--commit <sha>] -- <run.py 인자...>` → 제출, 완료 대기, `results/<slug>/`로 회수

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_launch.py`:
```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launch"))
import kaggle_job


def test_build_writes_valid_job(tmp_path):
    d = kaggle_job.build(tmp_path / "j", "kqt-e1-qwen", "abc123", ["--model", "Qwen/Qwen2.5-3B-Instruct", "--grid", "e1"])
    meta = json.loads((d / "kernel-metadata.json").read_text())
    assert meta["id"] == "dainjuser/kqt-e1-qwen"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    src = (d / "job.py").read_text()
    compile(src, "job.py", "exec")
    assert "abc123" in src and "Qwen/Qwen2.5-3B-Instruct" in src and "/kaggle/working/out" in src
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_launch.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'kaggle_job'`)

- [ ] **Step 3: 구현**

`launch/kaggle_job.py`:
```python
"""Push one private Kaggle GPU job that runs run.py at a pinned commit, wait for it, and download its results."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

USER = "dainjuser"
REPO = "https://github.com/dain77-d/does-4bit-speak-korean"
ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """import subprocess, sys
subprocess.run(["git", "clone", "-q", "{repo}", "/tmp/src"], check=True)
subprocess.run(["git", "-C", "/tmp/src", "checkout", "-q", "{commit}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/src/requirements-kaggle.txt"], check=True)
subprocess.run([sys.executable, "run.py", {args}, "--out", "/kaggle/working/out"], cwd="/tmp/src", check=True)
"""


def build(job_dir, slug, commit, run_args):
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.py").write_text(SCRIPT.format(repo=REPO, commit=commit, args=", ".join(repr(a) for a in run_args)))
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "job.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (job_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return job_dir


def kaggle(*args):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                       env={**os.environ, "PYTHONUTF8": "1"})
    if r.returncode:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def wait(slug, poll=60, timeout=6 * 3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        status = kaggle("kernels", "status", f"{USER}/{slug}").strip().splitlines()[-1]
        print(time.strftime("%H:%M:%S"), status, flush=True)
        if "COMPLETE" in status.upper():
            return True
        if "ERROR" in status.upper() or "CANCEL" in status.upper():
            return False
        time.sleep(poll)
    raise TimeoutError(slug)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("run_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    commit = a.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_args = [x for x in a.run_args if x != "--"]
    job = build(ROOT / "jobs" / a.slug, a.slug, commit, run_args)
    print(kaggle("kernels", "push", "-p", str(job)))
    ok = wait(a.slug)
    dest = ROOT / "results" / a.slug
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{USER}/{a.slug}", "-p", str(dest))
    print("ok" if ok else "FAILED", "->", dest)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_launch.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add launch/kaggle_job.py tests/test_launch.py
git commit -m "Kaggle GPU 작업 실행기"
```

---

### Task 10: Kaggle 스모크 실행 (두 모델 로딩·파이프라인 확인)

**Files:**
- 코드 변경 없음(문제 발견 시 `requirements-kaggle.txt`만 수정)

- [ ] **Step 1: 원격에 push** (작업이 커밋을 clone하므로)

```bash
git push
```

- [ ] **Step 2: Qwen 스모크**

Run: `python launch/kaggle_job.py --slug kqt-smoke-qwen -- --model Qwen/Qwen2.5-3B-Instruct --grid e1 --smoke`
Expected: `ok -> results/kqt-smoke-qwen`, 그 아래 `out/`에 JSON 3개(fp16, rtn4, gptq4-ko). fp16의 `flip`·`kl`은 0.

- [ ] **Step 3: EXAONE 스모크**

Run: `python launch/kaggle_job.py --slug kqt-smoke-exaone -- --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --grid e1 --smoke`
Expected: Qwen과 같은 형태의 JSON 3개.
EXAONE 로딩 오류(원격 코드와 transformers 버전 충돌)가 나면: `requirements-kaggle.txt`에 `transformers==4.46.3` 줄을 추가하고 커밋·push 후 Step 2와 3을 다시 실행한다. 그래도 실패하면 스펙 11절대로 원인을 기록하고 사용자에게 대체 모델을 묻는다.

- [ ] **Step 4: 스모크 결과 점검**

Run: `python -c "import json,glob; [print(json.load(open(f))['cfg']['method'], json.load(open(f))['results']) for f in glob.glob('results/kqt-smoke-*/out/*.json')]"`
Expected: 모든 값이 유한하고, gptq4의 ko·en PPL이 fp16의 3배 이내.

- [ ] **Step 5: Commit** (요구사항을 바꾼 경우에만)

```bash
git add requirements-kaggle.txt
git commit -m "Kaggle 환경 transformers 버전 고정"
```

---

### Task 11: E1 본 실행

**Files:**
- 결과: `results/kqt-e1-qwen/out/*.json`, `results/kqt-e1-exaone/out/*.json` (`.pt`는 .gitignore로 제외)

- [ ] **Step 1: Qwen E1 (19개 설정)**

Run: `python launch/kaggle_job.py --slug kqt-e1-qwen -- --model Qwen/Qwen2.5-3B-Instruct --grid e1`
Expected: `ok`, JSON 19개(예상 2~3시간). 실패하거나 JSON이 19개보다 적으면 Kaggle 작업 로그(`results/kqt-e1-qwen/*.log`)에서 오류를 확인해 기록하고, 원인을 고친 뒤 같은 명령을 다시 실행한다(Kaggle 작업은 매번 빈 디렉터리에서 시작하므로 19개 전부 다시 돈다).

- [ ] **Step 2: EXAONE E1 (19개 설정)**

Run: `python launch/kaggle_job.py --slug kqt-e1-exaone -- --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --grid e1`
Expected: `ok`, JSON 19개.

- [ ] **Step 3: Commit**

```bash
git add results/kqt-e1-qwen/out/*.json results/kqt-e1-exaone/out/*.json
git commit -m "E1 결과: 2개 모델 x 19개 설정"
```

---

### Task 12: E1 분석 (한국어 양자화 세금 표·그래프)

**Files:**
- Create: `analysis/e1_summary.py`
- Test: `tests/test_run.py`에 `tax_table` 테스트 추가
- 결과: `results/e1_summary.md`, `results/e1_flip.png`

**Interfaces:**
- Produces: `analysis.e1_summary.tax_table(records: list[dict]) -> pandas.DataFrame` (열: model, method, bits, calib_lang, ko_acc, en_acc, ko_flip, en_flip, ko_kl, en_kl, ko_ppl_ratio, en_ppl_ratio, flip_tax). `flip_tax = ko_flip / en_flip` (en_flip이 0이면 NaN), `*_ppl_ratio = ppl / 같은 모델 fp16 ppl`.

- [ ] **Step 1: 실패하는 테스트 추가** (`tests/test_run.py` 끝)

```python
def test_tax_table():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.e1_summary import tax_table

    def rec(method, bits, ko_flip, en_flip, ko_ppl, en_ppl):
        r = lambda f, p: {"acc": 0.5, "flip": f, "kl": f / 10, "ppl": p, "n_mcq": 100}
        return {"cfg": {"model": "m", "method": method, "bits": bits, "calib_lang": None},
                "results": {"ko": r(ko_flip, ko_ppl), "en": r(en_flip, en_ppl)}}

    df = tax_table([rec("fp16", 16, 0, 0, 10, 5), rec("rtn", 4, 0.2, 0.1, 12, 5.5)])
    row = df[df.method == "rtn"].iloc[0]
    assert abs(row.flip_tax - 2.0) < 1e-9
    assert abs(row.ko_ppl_ratio - 1.2) < 1e-9 and abs(row.en_ppl_ratio - 1.1) < 1e-9
```
(파일 맨 위 import에 `import sys` 와 `from pathlib import Path` 추가)

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_run.py::test_tax_table -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'analysis'`)

- [ ] **Step 3: 구현**

`analysis/__init__.py`: 빈 파일.

`analysis/e1_summary.py`:
```python
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
```
(`to_markdown`은 `tabulate`가 필요하다: `requirements.txt`에 `tabulate>=0.9` 추가)

- [ ] **Step 4: 통과 확인**

Run: `pytest -q`
Expected: 모두 통과 (27 passed)

- [ ] **Step 5: 실제 E1 결과로 실행하고 확인**

Run: `python analysis/e1_summary.py`
Expected: 38행 표 출력, `results/e1_summary.md`와 `results/e1_flip.png` 생성. 그래프를 열어 축·범례가 읽히는지 확인한다.

- [ ] **Step 6: Commit**

```bash
git add analysis/ tests/test_run.py requirements.txt results/e1_summary.md results/e1_flip.png
git commit -m "E1 분석: 한국어 양자화 세금 표와 flip 그래프"
```

---

## 완료 후

E1 표에서 다음을 확인하고 Plan 2(E2 원인 분석 + E3 뉴런 보호 + HAE-RAE 추가)를 작성한다:
- `flip_tax > 1`이 방법·비트 전반에서 나타나는지, 보정 언어(ko vs en)가 GPTQ와 AWQ에서 각각 차이를 만드는지.
- 3비트에서 한국어 PPL 비율이 영어보다 큰지.
