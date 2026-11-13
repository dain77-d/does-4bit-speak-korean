# Does 4-bit Speak Korean? — 설계 문서

- 작성일: 2026-11-14
- 저장소(가칭): `does-4bit-speak-korean`
- 상태: 설계 승인됨 → 사용자 검토 대기

## 1. 동기와 기여

LLM을 4비트 이하로 양자화하면 비라틴 문자 언어가 더 크게 손상된다는 보고(Marchisio et al., EMNLP Findings 2024)와, 영어 importance matrix로 양자화해도 다국어 성능이 불균형하게 떨어지지 않는다는 보고(arXiv 2503.03592)가 충돌한다. 보정 데이터 언어(Chimoto et al., EACL 2026)와 언어 맞춤 보정 + 혼합 정밀도(Waskito et al., AACL Findings 2026), 언어 인지 압축(LangCompress, 2025)은 이미 연구되었다. 그러나 (1) 한국어에 대한 통제된 측정, (2) 언어 특화 뉴런(LAPE, Tang et al. 2024)을 양자화에서 보호하는 방법, (3) 한국어의 높은 토큰 비용과 KV 캐시 양자화의 상호작용은 다뤄지지 않았다.

이 프로젝트의 기여:
1. **측정 (RQ1)**: 같은 모델·같은 하네스로 한국어 vs 영어 양자화 손실을 비교해 "한국어 양자화 세금"을 정량화하고, 충돌하는 선행 결론을 방법·지표별로 판정한다.
2. **원인 (RQ2)**: 한국어 특화 뉴런과 한국어 토큰 활성값이 양자화 오차에 특히 취약한지 분석한다.
3. **해결 (RQ3, 핵심)**: 한국어 특화 뉴런 보호 양자화 — 같은 실제 평균 비트에서 기존 보호 전략보다 한국어 손실을 줄이는지 검증한다.
4. **응용 (RQ4)**: 같은 내용당 토큰·KV 캐시 비용과 KV 캐시 양자화 손실을 한·영으로 비교하고, 4GB/8GB 기기용 한국어 배포 추천표를 만든다.

## 2. 연구 질문과 성공 기준

| RQ | 질문 | 판정 기준 |
|---|---|---|
| RQ1 | 같은 비트에서 한국어 손실(정확도 하락, flip rate, KL)이 영어보다 큰가? 방법·보정 언어에 따라 달라지는가? | 모델 2개 × 방법 4개 × 비트 3개에서 한·영 상대 손실 비율과 부호를 표로 보고 |
| RQ2 | 한국어 뉴런의 양자화 오차가 무작위·영어 뉴런보다 큰가? 오차가 어느 층에 몰리는가? | 뉴런 집합별 평균 출력 오차의 차이와 층별 히트맵 |
| RQ3 | 한국어 뉴런 보호가 같은 평균 비트에서 한국어 손실을 더 줄이는가? | (a) KMMLU flip rate가 활성값 기준 보호보다 낮고, 시드 3개 평균 차이 > 표준편차 (b) 영어 MMLU 정확도 하락 ≤ 0.5%p (FP16 대비 추가 하락 기준) |
| RQ4 | 한국어는 같은 내용에 더 많은 KV 캐시를 쓰고, KV 양자화에 더 취약한가? | 병렬 문장 기준 토큰 수·KV 메모리 비율, KV 4/2비트에서 한·영 PPL 증가율, 메모리 예산별 추천 설정 |

RQ3가 실패하면 음성 결과로 그대로 보고한다. RQ1·2·4는 RQ3 결과와 독립적으로 성립한다.

## 3. 범위

포함: 가중치 전용 양자화(W8/W4/W3, 그룹 128), 보정 언어 비교(영/한), 뉴런 보호, KV 캐시 4/2비트, W8A8 정수 추론 1줄 비교, T4 실측 속도·메모리.

제외: 2비트 가중치, 양자화 인지 학습(QAT), 실제 NPU 하드웨어 측정, 사람 평가·LLM-judge 생성 평가(대신 flip rate와 KL로 미세한 손상을 잡는다).

## 4. 모델과 데이터

**모델**
- `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` — 한국어 중심 모델 (연구용 라이선스)
- `Qwen/Qwen2.5-3B-Instruct` — 다국어 범용 모델
- `Qwen/Qwen2.5-7B-Instruct` — 규모 확인용 (핵심 설정만)

**데이터** (모두 인증 없이 접근 가능함을 2026-11-14 확인)
| 용도 | 데이터 | 사용 방식 |
|---|---|---|
| 한국어 지식 | `HAERAE-HUB/KMMLU` | 10개 과목 test에서 약 1,000문항, 고정 시드로 표본 |
| 한국어 문화·언어 | `HAERAE-HUB/HAE_RAE_BENCH_1.0` | 전체 또는 과제별 표본 |
| 영어 대조 | `cais/mmlu` | KMMLU와 대응되는 분야 약 1,000문항 |
| 뉴런 탐지·보정·PPL | `wikimedia/wikipedia` (`20231101.ko`, `20231101.en`) | 문서 단위로 겹치지 않게 3분할: 탐지 / 보정(128×512 토큰) / PPL 평가 |
| 병렬 비교 (RQ4) | `Helsinki-NLP/opus-100` (`en-ko` test) | 같은 내용의 한·영 문장 쌍 |

데이터 누수 방지: 뉴런 탐지, 보정, 평가는 서로 다른 문서 집합을 쓴다. 표본 추출은 모두 고정 시드.

## 5. 시스템 구조

정확도 실험은 **시뮬레이션 양자화**(가중치를 양자화 후 역양자화해 FP16으로 계산)로 하고, 속도·메모리는 **실제 커널**로 따로 측정한다. 새 기법(뉴런 보호)을 자유롭게 넣기 위함이다.

```
kqt/
  data.py      데이터 로딩·표본 추출·분할 (고정 시드)
  quant.py     group-wise 양자화: rtn, gptq, awq_scale, hqq → 역양자화 가중치 반환
  neurons.py   LAPE: 언어별 FFN 뉴런 활성 확률 → 한국어/영어 특화 뉴런 인덱스
  protect.py   보호 전략별 뉴런 선택 + 해당 가중치 줄 FP16 유지 + 실제 평균 비트 계산
  evaluate.py  MCQ 로그우도 채점, PPL, flip rate, FP16 대비 KL
  kv.py        HF 양자화 KV 캐시(4/2비트), 병렬 문장 토큰·메모리·지연 측정
  speed.py     bitsandbytes NF4/INT8, llama.cpp GGUF 실측 (tokens/s, peak memory)
run.py         실험 설정(YAML 한 줄) → 결과 JSON
launch/        Kaggle 작업 생성·제출·결과 회수 (Modal은 메모리 부족 시 대체)
analysis/      결과 JSON → 표·그래프·배포 추천표
tests/         단위 테스트
```

각 단위의 인터페이스:
- `quant.quantize_linear(W, bits, group=128, method, calib_X=None) -> W_hat`
- `neurons.lape(model, texts_by_lang) -> {lang: {layer: idx}}`
- `protect.apply(model, method, bits, strategy, budget, neurons) -> (model, avg_bits)`
- `evaluate.run(model, suite) -> {acc, flip_rate, kl, ppl}` (FP16 기준 출력은 캐시해 재사용)

**뉴런 보호의 정의**: 층 l의 FFN 뉴런 i를 보호하면 `gate_proj`·`up_proj`의 i번째 행과 `down_proj`의 i번째 열을 FP16으로 둔다. 실제 평균 비트 = (양자화 가중치 비트 + 그룹 스케일/영점 + 보호 가중치 16비트 + 인덱스 오버헤드) / 전체 가중치 수. 보호 전략 간 비교는 같은 실제 평균 비트에서 한다.

**보호 전략 (7종)**: 없음, 무작위, 가중치 크기, 활성값 크기(AWQ식 salient), 영어 뉴런, 한국어 뉴런, 한국어+영어 뉴런.

## 6. 실험

| ID | 내용 | 설정 수(대략) |
|---|---|---|
| E1 | 모델 2 × {RTN, GPTQ, AWQ, HQQ} × {8, 4, 3비트} + FP16; GPTQ·AWQ는 보정 언어 {en, ko} | 38 |
| E2 | E1의 GPTQ 4·3비트 모델에서 뉴런 집합별·층별 출력 오차, 한국어/영어 토큰 활성값 이상치 통계 | 분석 |
| E3 | 모델 2 × GPTQ {4, 3비트} × 예산 {0.25, 0.5, 1%} × 전략 7 (전략 간 비교는 시드 3개로 반복할 핵심 셀만) | 약 84 |
| E4 | 병렬 문장 토큰 수·KV 메모리; KV {16, 4, 2비트} × 가중치 {FP16, 4비트} × 언어 2; 실측 속도·메모리; W8A8(SmoothQuant식) 1줄 | 약 20 |
| E5 | Qwen2.5-7B에서 E1·E3 핵심 설정 재현 | 약 6 |

연산: 1회 5~10분(T4), 총 15~20 GPU시간. Kaggle 무료 한도(주 30시간) 안에서 2주에 나눠 실행. 7B처럼 메모리가 빠듯한 경우만 Modal.

## 7. 오류 처리와 재현성

- 각 실행은 설정 해시로 결과 JSON을 저장하고, 이미 있으면 건너뛴다 (중단 후 재개 가능).
- Kaggle 작업 실패 시 로그를 회수해 원인을 기록하고, 같은 설정을 최대 1회 재시도한다.
- 모든 결과 JSON에 모델 리비전, 라이브러리 버전, 시드, 실제 평균 비트를 기록한다.
- FP16 기준 로짓은 한 번만 계산해 저장하고 flip rate·KL 계산에 재사용한다.

## 8. 테스트

- `quant`: 임의 행렬에서 비트가 높을수록 재구성 오차 단조 감소; GPTQ 오차 ≤ RTN 오차 (보정 입력이 있을 때).
- `protect`: 보호한 행·열이 원본 FP16과 비트 단위로 동일; 실제 평균 비트 계산이 손으로 계산한 작은 예와 일치.
- `neurons`: 특정 언어에서만 켜지도록 만든 합성 뉴런을 LAPE가 찾아냄.
- `evaluate`: 정답을 아는 작은 MCQ에서 채점 정확; 같은 모델끼리 flip rate 0, KL 0.
- 모든 테스트는 GPU 없이 작은 텐서로 로컬에서 실행.

## 9. 산출물

- GitHub 저장소 (작성자 dain77-d noreply, Claude 공동 작성자 표기 없음)
- 결과 그래프: 한·영 손실 비교, 층별 오차 히트맵, 보호 전략별 Pareto 곡선(평균 비트 vs flip rate), KV 비트별 손실, 속도·메모리
- 4GB/8GB 기기용 한국어 배포 추천표
- 논문 형식 README (동기, 관련 연구, 방법, 결과, 한계)

## 10. 일정

| 주차 | 작업 |
|---|---|
| 1 | data·quant·evaluate + 테스트, Kaggle 실행기 → E1 |
| 2 | neurons·protect → E2, E3 |
| 3 | kv·speed → E4, E5 |
| 4 | 분석·그래프·추천표·README |

## 11. 위험과 대응

| 위험 | 대응 |
|---|---|
| EXAONE 커스텀 코드가 최신 transformers와 충돌 | 동작하는 transformers 버전을 고정; 안 되면 Korean-capable 대체 모델(Qwen2.5-1.5B 등)로 교체하고 기록 |
| 3B 모델 GPTQ 보정이 T4에서 느림 | 층 단위 순차 처리, 보정 토큰 수 128×512로 제한 |
| 한국어 뉴런 보호 효과 없음 | 음성 결과로 보고 (RQ3 기준 미달을 명시) |
| KMMLU 표본이 작아 차이가 노이즈에 묻힘 | flip rate·KL을 함께 보고, 핵심 셀은 시드 3개 반복 |

## 참고 문헌

- Marchisio et al. 2024, How Does Quantization Affect Multilingual LLMs? (arXiv 2407.03211)
- English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance (arXiv 2503.03592)
- Chimoto et al. 2026, Calibrating Beyond English (arXiv 2601.18306, EACL 2026)
- Waskito et al. 2026, Multilinguality after Quantization (AACL Findings 2026)
- LangCompress: Language-Aware Compression of LLMs (IJCNLP-AACL 2025)
- Tang et al. 2024, Language-Specific Neurons (arXiv 2402.16438)
- Dutta et al. 2024, Accuracy is Not All You Need (arXiv 2407.09141)
- Frantar et al. 2023, GPTQ; Lin et al. 2024, AWQ; Badri & Shaji 2023, HQQ; Xiao et al. 2023, SmoothQuant; Liu et al. 2024, KIVI
- Thunder-Tok (arXiv 2506.15138)
