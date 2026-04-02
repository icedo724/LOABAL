# LOABAL

로스트아크 직업 밸런스 감성 분석 프로젝트

---

## 진행 현황

| 항목 | 상태 |
|------|------|
| 크롤링 | 완료 (2026.03.20) |
| 라벨링 | **진행 중** — 1,648 / 2,929건 (56.3%) |
| 모델 학습 | v2 완료 (통합 / 딜러 / 서포터) |
| 만족도 분석 | 예정 |

---

## 파이프라인

```
크롤링 (Selenium)
    ↓
자동 라벨링 (Groq API / llama-3.3-70b-versatile)
    ↓
규칙 기반 후처리 (auto_review.py)
    ↓
모델 학습 (KoELECTRA-base-v3 파인튜닝)
    ↓
직업별 밸런스 만족도 분석
```

## 데이터

- **출처:** 인벤 로스트아크 직업별 게시판 29개
- **수집 기간:** 직업당 최신 게시글 기준 (2026.03.20 수집)
- **원본 게시글:** 12,661건 (unique post_id: 2,929건)
- **분석 단위:** expression — 게시글당 밸런스 관련 표현 N개 추출

## 모델

| 모델 | 설명 | 현재 성능 (Test Macro F1) |
|------|------|--------------------------|
| 통합 | 전체 직업 | 0.611 (v2) |
| 딜러 | 딜러 직업 전체 | 0.614 (v2) |
| 서포터 | 바드 / 홀리나이트 / 도화가 / 발키리 | 0.559 (v2) |

- 사전학습 모델: `monologg/koelectra-base-v3-discriminator`
- 레이블: `positive / negative / neutral / unrelated`

## 실행

```bash
# 라벨링
python -X utf8 -u labeling/auto_labeler.py

# 후처리
python labeling/auto_review.py

# 학습
python model/train.py --mode unified   # 통합
python model/train.py --mode dealer    # 딜러
python model/train.py --mode support   # 서포터

# 데모
python demo.py
```
