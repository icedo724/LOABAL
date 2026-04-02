# 모델 기본 정보

## 학습 환경

| 항목 | 사양 |
|------|------|
| OS | Windows 11 Pro (Build 26200) |
| Python | 3.12.3 |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA GeForce RTX 4060 |
| VRAM | 8.0 GB |
| CUDA Compute Capability | 8.9 |
| Transformers | 5.1.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.0 |
| numpy | 2.4.2 |

---

## 사전학습 모델

**`monologg/koelectra-base-v3-discriminator`**

ELECTRA 아키텍처는 MLM 방식의 BERT와 달리 Generator-Discriminator 구조로 모든 토큰에 학습 신호를 생성한다. 동일 파라미터 수 대비 데이터 효율이 높고, 한국어 NLP 벤치마크(NSMC, KorSTS)에서 KoBERT 동등 이상의 성능을 보인다. 본 프로젝트의 입력이 expression 단위(30~80자) 단문임을 고려해 선택하였다.

---

## 하이퍼파라미터

| 파라미터 | 값 | 선정 근거 |
|----------|----|-----------|
| MAX_LENGTH | 128 | expression 단위 입력은 대부분 30~60자. 128 토큰으로 충분히 커버 |
| TRAIN_BATCH | 16 | RTX 4060 8GB 기준. 32는 OOM 위험, 8은 학습 불안정 |
| EPOCHS | 5 | ~4,000건 규모에서 3~5 에포크가 일반적. Early Stopping으로 자동 탐색 |
| LR | 3e-5 | BERT 계열 권장 범위(1e-5~5e-5). 수렴 안정성과 사전학습 지식 보존의 균형 |
| WARMUP_RATIO | 0.1 | 초기 스텝 LR 선형 증가로 파국적 망각 방지. BERT 원 논문 권장값 |
| WEIGHT_DECAY | 0.01 | AdamW L2 정규화. BERT 계열 표준값 |
| PATIENCE | 2 | Val Macro F1 기준 연속 2 에포크 미개선 시 Early Stop |
| TEST_SIZE | 0.1 | 전체의 10% 고정. Stratified split으로 클래스 비율 유지 |
| VAL_SIZE | 0.1 | Train의 10% (전체 약 9%) |

---

## 클래스 정의

| 레이블 | 의미 |
|--------|------|
| negative | 너프/하향 불만, 버프/상향 요청, 딜 약함 호소 등 밸런스 부정 표현 |
| positive | 강하다, 충분하다, 너프 불필요 등 밸런스 긍정 표현 |
| neutral | 밸런스 관련이나 긍정/부정 방향 불분명 (패치 언제, 체감 애매 등) |
| unrelated | 장비/팔찌/각인 질문, 빌드 공략, 강화, 잡담 등 밸런스 무관 |

---

## 평가 지표

| 지표 | 설명 | 용도 |
|------|------|------|
| Macro F1 | 클래스별 F1 단순 평균. 불균형에 무관 | **주 평가 지표** |
| Accuracy | 전체 정확도 | 참고 |
| Weighted F1 | 샘플 수 비례 가중 F1 | 참고 |
| Confusion Matrix | 클래스 간 혼동 패턴 | 오류 분석 |

---

## 모델 구성 계획

| 모델 | 학습 데이터 | 목적 |
|------|------------|------|
| 통합 (`unified`) | 전체 직업 | 베이스라인 |
| 딜러 (`dealer`) | 딜러 직업 전체 | 딜러 특화 |
| 서폿 (`support`) | 바드 / 홀리나이트 / 도화가 / 발키리 | 서포터 특화 |
| 직업 개별 | 데이터 상위 직업 3~4개 | 단일 직업 특화, 통합 모델과 비교 |

직업 개별 모델 대상은 라벨링 완료 후 직업별 데이터 분포 확인 후 선정한다.
