# LOABAL

로스트아크 직업 밸런스 감성 분석 한국어 NLP 프로젝트

---

## 진행 현황

| 항목 | 상태 |
|------|------|
| 크롤링 | 완료 (2026.03.20, 12,661건) |
| 라벨링 | **완료** — 2,929 / 2,929건 (100%, expression 10,258개) |
| 모델 학습 | v5 완료 (통합 / 딜러 / 서포터) |
| 실사용 추론 | 완료 (2026.04.04~08, 전 직업 29개) |

---

## 파이프라인

```
크롤링 (Selenium + BeautifulSoup)
    ↓
자동 라벨링 (Groq API / llama-3.3-70b-versatile)
    ↓
규칙 기반 후처리 (auto_review.py)
    ↓
모델 학습 (KoELECTRA-base-v3 파인튜닝)
    ↓
직업별 밸런스 만족도 추론 (inference.py)
```

---

## 데이터

- **출처:** 인벤 로스트아크 직업별 게시판 29개
- **수집 기간:** 직업당 최신 게시글 기준 (2026.03.20 수집)
- **원본 게시글:** 12,661건 (unique post_id: 2,929건)
- **분석 단위:** expression — 게시글당 밸런스 관련 표현 N개 추출
- **레이블:** `positive / negative / neutral / unrelated`

---

## 모델

- 사전학습 모델: `monologg/koelectra-base-v3-discriminator`
- 입력 형식: `직업: {job_class} 표현: {expression_text}`

| 모델 | 학습 데이터 | Test Macro F1 | Test Accuracy | HuggingFace |
|------|------------|--------------|--------------|-------------|
| 통합 (unified) | 10,258개 | 0.6281 | 69.4% | [loabal-koelectra-unified](https://huggingface.co/mininiming/loabal-koelectra-unified) |
| 딜러 (dealer) | 8,972개 | 0.6320 | 69.6% | [loabal-koelectra-dealer](https://huggingface.co/mininiming/loabal-koelectra-dealer) |
| 서포터 (support) | 1,286개 | 0.5520 | 59.7% | [loabal-koelectra-support](https://huggingface.co/mininiming/loabal-koelectra-support) |

---

## 실사용 추론 결과 (2026.04.04~08)

전 직업 29개 게시판 1,192건 수집 후 v5 통합 모델 기준 만족도 산출

**만족도 상위 5개**

| 직업 | 만족도 |
|------|--------|
| 데모닉 | 33.3% |
| 블래스터 | 32.3% |
| 서머너 | 31.1% |
| 홀리나이트 | 29.7% |
| 워로드 | 29.3% |

**만족도 하위 5개**

| 직업 | 만족도 |
|------|--------|
| 호크아이 | 9.8% |
| 데빌헌터 | 12.0% |
| 건슬링어 | 13.4% |
| 환수사 | 14.5% |
| 스트라이커 | 15.6% |

> 만족도 = positive / (positive + negative + neutral), unrelated 제외

---

## 실행

```bash
# 크롤링
python -X utf8 crawling/crawler.py --since 2026-04-04

# 라벨링
python -X utf8 labeling/auto_labeler.py

# 후처리
python labeling/auto_review.py

# 학습
python model/train.py --mode unified   # 통합
python model/train.py --mode dealer    # 딜러
python model/train.py --mode support   # 서포터

# 추론
python model/inference.py --mode unified
python model/inference.py --mode dealer
python model/inference.py --mode support

# 데모
python demo.py
```

---

## 참여자

조민서 · icedo724@gmail.com
