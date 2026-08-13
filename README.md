# VRINGON Vision QA — 신발 완제품 외관 검사 데모

**라이브 데모: https://jhkim1543.github.io/vringon-vision-qa/**

신발 이미지를 업로드하면 **부위 분할 → 컬러웨이 레퍼런스 검색 → PatchCore 이상탐지 → 부위 조건 결함 분류 → QA 판정(PASS/REVIEW/FAIL)** 을 수행하는 데모입니다.

## 구조

```
vringon-qa/
├─ pipeline/
│  ├─ download_data.py    # 공개 데이터 다운로드 (VisA cashew, ipogorelov/sneakers)
│  ├─ filter_lateral.py   # 흰 배경 측면 뷰 자동 필터 → data/sku/<SKU>/
│  ├─ engine.py           # 실루엣·부위분할·PatchCore(WideResNet50, LOO 캘리브레이션)
│  ├─ make_defects.py     # 부위 인식 합성 결함 5종 (GT 마스크 생성)
│  ├─ run_samples.py      # 전체 샘플 추론 → docs/assets/samples/ (이미지+JSON)
│  ├─ bench_visa.py       # VisA 실측 불량 벤치마크 (image/pixel AUROC)
│  └─ server.py           # FastAPI 실시간 업로드 추론 (포트 5210)
├─ data/                  # 다운로드 데이터 (재생성 가능, 배포 제외)
└─ docs/                  # 정적 웹 데모 (GitHub Pages 배포 가능)
```

## 실행

```bash
# 정적 데모(사전 계산 샘플만): docs/를 아무 정적 서버로 열기
# 실시간 업로드 추론 포함:
.venv\Scripts\python.exe pipeline\server.py   # → http://localhost:5210
```

재현: `download_data.py` → `filter_lateral.py` → `run_samples.py` → `bench_visa.py`

## 데이터 출처 (실제 공개 데이터)

| 데이터 | 용도 | 라이선스 |
|---|---|---|
| [ipogorelov/sneakers](https://huggingface.co/datasets/ipogorelov/sneakers) | 실측 스니커 사진 → SKU 정상 레퍼런스·테스트 샘플 | MIT |
| [VisA](https://github.com/amazon-science/spot-diff) (HF 미러 imaadd05/visa-anomaly-detection) | 실제 공장 불량 벤치마크 (cashew, 정상 500·불량 100·픽셀 GT) | CC BY 4.0 |

신발 전용 공개 결함 데이터가 없어, 결함 샘플은 **부위 인식 기반 합성**(오염·접착제 과다·실밥·스커프 등을 해당 부위에만 주입, GT 마스크 보유)으로 만들고, 모델 자체 품질은 **VisA 실측 불량**으로 검증합니다. 이는 2026 footwear 연구의 part-aware synthetic defect 접근을 따른 것입니다.

## 판정 파이프라인

1. 실루엣 추출 (흰 배경 가정) → 토 방향 정규화
2. 규칙 기반 부위 분할 7영역 (어퍼·토·칼라·힐·미드솔·아웃솔·접착 경계)
3. 동일 SKU·컬러웨이 레퍼런스 Top-8 검색 (HSV 히스토그램)
4. PatchCore: WideResNet50 layer2+3 패치 특징, SAHI식 5뷰(전체+2×2 타일), 코어셋 kNN
5. **융합 히트맵 공간 캘리브레이션**: 레퍼런스별 leave-one-out 히트맵을 테스트와 동일 경로
   (z정규화→멀티뷰 max 융합→블러)로 만들고, 가장 in-distribution인 레퍼런스의 피크 × 1.06을 임계값으로 사용
6. **2단계 검출**: z > thr → 정식 검출(minor/major), 0.8×thr < z ≤ thr → REVIEW 후보(저신뢰 플래그)
7. 부위·형상·대비(부위 중앙값 대비) 규칙으로 결함 유형 분류
8. Severity → PASS / REVIEW / FAIL — major는 z > 2×thr 또는 면적 > 2.5%일 때만 (정상 FAIL 오판 0%)

`reclassify.py`는 저장된 z 히트맵으로 검출·분류·지표만 재계산(모델 재실행 없음),
`tune_thr.py`는 REVIEW 후보 임계 계수 그리드 서치 도구입니다.

## 지표 — 같은 엔진, 두 가지 촬영 조건

| 지표 | 고정 리그 (통제 촬영) | 자유 촬영 (공개 사진) |
|---|---:|---:|
| 이미지 판별 AUROC (정상/불량) | **1.00** | 0.65 |
| 결함 우세율 (결함 > 오검출) | **100%** | 60% |
| 1순위 적중 (최상위 표시가 진짜 결함) | **100%** | 56% |
| 정상 PASS (정상을 정상이라 판정) | **100%** | 30% |
| Pixel AUROC (히트맵 위치) | **0.997** | 0.946 |
| 샘플 수 | 12 (정상 6 · 결함 6) | 20 (정상 10 · 결함 10) |

실측 산업 데이터 검증 — **VisA cashew** (정상 40 · 불량 60, 사람 주석 마스크):
**Image AUROC 0.958 · Pixel AUROC 0.992**

**두 트랙의 차이는 알고리즘이 아니라 레퍼런스입니다.** 공개 데이터에는 같은 제품의 사진이 여러 장
있는 그룹이 없어(brand·model만 제공, 제품 ID 없음) 자유 촬영 트랙의 레퍼런스는 전부 **서로 다른
실물 개체**입니다. 진단 결과 개체·포즈 차이로 생기는 이상 신호가 진짜 결함과 같은 크기였고
(결함 peak 2.12 vs 노이즈 1.54처럼 간발), 점수 규칙 8종·부위 자기참조·AND 융합을 모두 시도해도
0.72가 한계였습니다. 반면 같은 개체를 같은 지그에서 재촬영한 조건(고정 리그)에서는 동일 코드가
판별 AUROC 1.00을 냅니다. 즉 **촬영 통제가 이 시스템의 전제**입니다.

고정 리그 트랙은 `make_rig_samples.py`가 한 장의 원본 사진에서 미세 회전(±1.2°)·배율(±1.2%)·
평행이동·노출 드리프트·화이트밸런스 드리프트·센서 노이즈·JPEG 재압축을 적용해 "같은 신발을 지그에서
다시 찍은 샷"들을 만들고, 그중 6장을 레퍼런스 뱅크로, 홀드아웃 1장을 테스트로 씁니다.

**대응 — 레퍼런스 적합도 기반 기권**: 업로드 시 최상위 레퍼런스 유사도로 golden(≥0.985) /
near(≥0.90) / mismatch를 판정하고, mismatch면 verdict를 `unknown`(판정 불가)으로 돌립니다.
검증: 라이브러리에 없는 정상 신발 3종(유사도 0.35~0.78)은 기권 전 FAIL·검출 11건까지 냈으나
기권 후 오판이 사라졌고, 골든 레퍼런스가 있을 때는 정상 PASS(z 0.55) / 어퍼 오염 REVIEW(z 2.62,
부위·유형 정확) / 접착제 과다 REVIEW(z 2.17, 접착 경계)로 정확히 동작합니다.

## 평가·튜닝 도구

`eval_pairs.py` 정상/결함 쌍 판별력 · `diag_signal.py` 결함이 노이즈를 이기는지 진단 ·
`eval_score.py` 이미지 점수 규칙 8종 비교 · `eval_selfref.py`·`eval_fusion.py` 부위 자기참조 및
AND 융합 실험 · `honest_metrics.py` 과대평가 불가능한 지표로 metrics.json 재작성
