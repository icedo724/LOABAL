"""
labeling/auto_labeler.py

Groq API (llama-3.3-70b-versatile) 로 크롤링 CSV 전체를 expression-level 자동 라벨링.
게시글 1건 → 밸런스 관련 표현 N개 추출 → CSV N행으로 저장.

사용법:
  python labeling/auto_labeler.py
  python labeling/auto_labeler.py --input data/lostark_crawled_20260322_1200.csv
  python labeling/auto_labeler.py --no-resume

[Groq 무료 한도] llama-3.3-70b-versatile : RPM 30, RPD 1,000, TPD 100,000

[.env 설정]
  GROQ_API_KEY_1=gsk_...
  GROQ_API_KEY_2=gsk_...  (선택 — 키 1개당 1,000 RPD 추가)
"""

import os
import re
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_schema import (
    BalanceLabel, LABEL_GUIDE, INPUT_COLUMNS, EXPRESSION_COLUMNS,
    get_project_paths, find_latest_crawled
)

RPD_PER_KEY       = 1_000
RPM_PER_KEY       = 30
CONFIDENCE_THRESH = 0.75
BATCH_SAVE_EVERY  = 100
MODEL_ID          = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = f"""Lost Ark 한국 커뮤니티 게시글에서 직업 밸런스 관련 표현을 추출하라.

감성 분류({LABEL_GUIDE}):
- negative: 너프/하향 불만, 버프/상향 요청, 딜 약함, 살려줘, 극저점, ▇━━●▅▇█▇▆▅▄▇(드러눕기시위=강한불만)
- positive: 강하다, 충분하다, 너프불필요, 밸런스좋다
- neutral: 패치언제, 체감애매, 모르겠다
- unrelated: 장비/팔찌/각인 질문, 빌드공략, 강화, 잡담

규칙:
- 밸런스 표현 전부 추출 + unrelated 최대 2개
- 드러눕기 이모지 → negative 1개
- 밸런스 내용 없으면 → unrelated 1개

예시1: "디트 버프해줘" → {{"expression_text":"디트 버프해줘","sentiment":"negative","confidence":0.97,"reason":"버프 요청"}}
예시2: "붕쯔 팔찌 어떤거요?" → {{"expression_text":"붕쯔 팔찌 어떤거요","sentiment":"unrelated","confidence":0.97,"reason":"장비 질문"}}
예시3: "붕쯔는 너프 필요없다" → {{"expression_text":"붕쯔는 너프 필요없다","sentiment":"positive","confidence":0.90,"reason":"현상태 만족"}}

JSON 배열만 출력. 마크다운 금지.
[{{"expression_text":"...","sentiment":"...","confidence":0.0,"reason":"..."}}]"""


class RateTracker:
    """단일 Groq 키의 RPM 슬라이딩 윈도우 + RPD 카운터."""

    def __init__(self):
        self.daily_count = 0
        self._rpm_window = []

    def wait_rpm(self):
        now = time.time()
        self._rpm_window = [t for t in self._rpm_window if now - t < 60]
        if len(self._rpm_window) >= RPM_PER_KEY:
            wait = 60 - (now - self._rpm_window[0]) + 0.5
            if wait > 0:
                print(f"  [RPM 대기] {wait:.1f}초 대기...")
                time.sleep(wait)

    def record(self):
        self._rpm_window.append(time.time())
        self.daily_count += 1

    def exhausted(self) -> bool:
        return self.daily_count >= RPD_PER_KEY

    def status_line(self) -> str:
        return f"{self.daily_count}/{RPD_PER_KEY}"


def build_prompt(job_class: str, title: str, content: str, comments: str) -> str:
    content_preview  = (content[:300]  + "...") if len(content)  > 300  else content  or "(없음)"
    comments_preview = (comments[:400] + "...") if len(comments) > 400  else comments or "(없음)"
    return (
        f"직업(Job Class): {job_class}\n"
        f"제목(Title): {title}\n"
        f"본문(Content): {content_preview}\n"
        f"댓글(Comments): {comments_preview}\n\n"
        "Extract balance-related expressions from this post."
    )


def parse_response(raw: str) -> list[dict]:
    """응답 문자열에서 JSON 배열을 추출하고 유효성을 검증."""
    cleaned      = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    valid_labels = {lb.value for lb in BalanceLabel}

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        items = json.loads(match.group())
    else:
        obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if obj_match:
            items = [json.loads(obj_match.group())]
        else:
            raise ValueError(f"JSON 없음. 원본: {raw[:200]}")

    validated = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text      = str(item.get("expression_text", "")).strip()
        sentiment = item.get("sentiment", "")
        if not text or sentiment not in valid_labels:
            continue
        validated.append({
            "expression_text": text,
            "sentiment"      : sentiment,
            "confidence"     : float(item.get("confidence", 0.8)),
            "reason"         : str(item.get("reason", "")),
        })

    if not validated:
        raise ValueError(f"유효한 expression 없음. 원본: {raw[:200]}")
    return validated


def call_groq(client: Groq, tracker: RateTracker,
              job_class: str, title: str, content: str, comments: str) -> list[dict]:
    """Groq API 호출. RPM 초과 시 대기 후 재시도, TPD 소진 시 RuntimeError('TPD_EXHAUSTED')."""
    prompt = build_prompt(job_class, title, content, comments)

    while True:
        if tracker.exhausted():
            raise RuntimeError("일일 한도가 소진되었습니다.")
        tracker.wait_rpm()

        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            tracker.record()
            return parse_response(response.choices[0].message.content.strip())

        except Exception as e:
            err   = str(e)
            is_429 = "429" in err or "rate_limit" in err.lower()
            if not is_429:
                raise

            if "tokens per day" in err.lower() or "tpd" in err.lower():
                raise RuntimeError("TPD_EXHAUSTED")

            # retry-after 파싱: "try again in 1m3.936s" 또는 "try again in 31.2s"
            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", err)
            if m:
                wait = int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
            else:
                wait = 62
            print(f"  [TPM/RPM] 한도 → {wait:.0f}초 대기 후 재시도...")
            time.sleep(wait)


def label_csv(input_path: str, output_path: str, resume: bool = True):
    paths = get_project_paths()
    load_dotenv(dotenv_path=paths["env"])

    keys = [v for i in range(1, 20)
            if (v := os.getenv(f"GROQ_API_KEY_{i}"))]
    if not keys:
        print(
            "\n[오류] Groq API 키가 없습니다.\n"
            "  1. https://console.groq.com → API Keys → Create API Key\n"
            f"  2. {paths['env']} 에 GROQ_API_KEY_1=gsk_... 형식으로 추가\n"
        )
        sys.exit(1)

    clients  = [Groq(api_key=k) for k in keys]
    trackers = [RateTracker() for _ in keys]
    key_idx  = 0

    df_input = pd.read_csv(input_path, dtype={"post_id": str})
    for col in INPUT_COLUMNS:
        if col not in df_input.columns:
            raise ValueError(f"필수 컬럼 '{col}' 없음.")

    processed_ids = set()
    if resume and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
        df_existing   = pd.read_csv(output_path, dtype={"post_id": str})
        processed_ids = set(df_existing["post_id"])
        results       = df_existing.to_dict("records")
        print(f"[재개] 처리 완료 포스트 {len(processed_ids)}건 건너뜀")
    else:
        results = []

    df_todo   = df_input[~df_input["post_id"].isin(processed_ids)].drop_duplicates(subset="post_id")
    total     = len(df_todo)
    if total == 0:
        print("모든 항목 처리 완료.")
        return

    total_rpd = RPD_PER_KEY * len(keys)
    est_min   = total / (RPM_PER_KEY * len(keys))
    print(f"\n{'='*56}")
    print(f" 모델     : {MODEL_ID}")
    print(f" API 키   : {len(keys)}개  (일일 한도 합산 {total_rpd:,}건)")
    print(f" 처리 예정: {total:,}건  |  예상 시간: 약 {est_min:.0f}분")
    print(f" 출력 경로: {output_path}")
    if total > total_rpd:
        days = (total + total_rpd - 1) // total_rpd
        print(f" [!] 일일 한도 초과 → {days}일 분할 처리 (자동 이어하기)")
    print(f"{'='*56}\n")

    error_count = 0

    for i, (_, row) in enumerate(df_todo.iterrows(), 1):
        while key_idx < len(keys) and trackers[key_idx].exhausted():
            print(f"  [키{key_idx+1} 소진] 다음 키로 전환...")
            key_idx += 1
        if key_idx >= len(keys):
            print(f"\n[!] 모든 키 일일 한도 소진. 저장 후 종료.")
            break

        post_id   = str(row["post_id"])
        job_class = str(row.get("job_class", ""))
        title     = str(row.get("title",     ""))
        content   = str(row.get("content",   ""))
        comments  = str(row.get("comments",  ""))

        def _append(expressions: list[dict]):
            for expr in expressions:
                results.append({
                    "post_id"        : post_id,
                    "job_class"      : job_class,
                    "expression_text": expr["expression_text"],
                    "sentiment"      : expr["sentiment"],
                    "confidence"     : round(expr["confidence"], 3),
                    "reason"         : expr["reason"],
                    "is_reviewed"    : "NEEDS_REVIEW" if expr["confidence"] < CONFIDENCE_THRESH else False,
                })

        try:
            expressions  = call_groq(clients[key_idx], trackers[key_idx], job_class, title, content, comments)
            needs_review = sum(1 for e in expressions if e["confidence"] < CONFIDENCE_THRESH)
            flag         = "[검수필요]" if needs_review else "[OK]     "
            _append(expressions)
        except RuntimeError as e:
            if "TPD_EXHAUSTED" in str(e):
                flag          = "[ERR]    "
                expressions   = []
                all_exhausted = False
                err           = str(e)
                while "TPD_EXHAUSTED" in err:
                    print(f"  [키{key_idx+1} TPD 소진] 다음 키로 전환...")
                    trackers[key_idx].daily_count = RPD_PER_KEY
                    key_idx += 1
                    if key_idx >= len(keys):
                        print(f"\n[!] 모든 키 일일 토큰 소진. 저장 후 종료.")
                        all_exhausted = True
                        break
                    try:
                        expressions = call_groq(clients[key_idx], trackers[key_idx], job_class, title, content, comments)
                        flag = "[OK]     "
                        _append(expressions)
                        err  = ""
                    except RuntimeError as re2:
                        err = str(re2)
                        if "TPD_EXHAUSTED" not in err:
                            error_count += 1
                            results.append({"post_id": post_id, "job_class": job_class,
                                            "expression_text": "", "sentiment": "ERROR",
                                            "confidence": 0.0, "reason": err[:200], "is_reviewed": "NEEDS_REVIEW"})
                            err = ""
                    except Exception as ge:
                        err = ""
                        error_count += 1
                        results.append({"post_id": post_id, "job_class": job_class,
                                        "expression_text": "", "sentiment": "ERROR",
                                        "confidence": 0.0, "reason": str(ge)[:200], "is_reviewed": "NEEDS_REVIEW"})
                if all_exhausted:
                    break
            else:
                print(f"\n[중단] {e}")
                break
            used    = sum(t.daily_count for t in trackers)
            cur_key = min(key_idx, len(keys) - 1)
            print(
                f"[{i:>5}/{total}] {flag} "
                f"{job_class:<12} | {title[:22]:<22} → "
                f"{len(expressions)}개 표현  (키{cur_key+1}: {trackers[cur_key].status_line()} | 합계 {used}/{total_rpd})"
            )

        except Exception as e:
            error_count += 1
            err_msg = str(e)
            print(f"[{i:>5}/{total}] [ERR] post_id={post_id}: {err_msg[:80]}")
            results.append({
                "post_id"        : post_id,
                "job_class"      : job_class,
                "expression_text": "",
                "sentiment"      : "ERROR",
                "confidence"     : 0.0,
                "reason"         : err_msg[:200],
                "is_reviewed"    : "NEEDS_REVIEW",
            })

        if i % BATCH_SAVE_EVERY == 0:
            used = sum(t.daily_count for t in trackers)
            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"  [중간 저장] {i}건 처리 완료 | 합계 {used}/{total_rpd}")

    df_out       = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    needs_review = (df_out["is_reviewed"] == "NEEDS_REVIEW").sum()
    total_expr   = len(df_out)
    total_used   = sum(t.daily_count for t in trackers)
    remaining    = total - total_used

    print(f"\n{'='*56}")
    print(f" 완료! 포스트 {total_used}건 처리 → expression {total_expr}건")
    for j, t in enumerate(trackers):
        print(f"  키{j+1}: {t.daily_count}건 사용")
    print(f"  오류         : {error_count}건")
    print(f"  수동검수 필요: {needs_review}건 (신뢰도 < {CONFIDENCE_THRESH})")
    if remaining > 0:
        print(f"  미처리 잔량  : {remaining}건 → 내일 재실행 시 자동 이어서 처리")
    print(f"\n  다음 단계: python labeling/label_reviewer.py")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    paths = get_project_paths()

    parser = argparse.ArgumentParser(
        description="로스트아크 밸런스 표현 추출 자동 라벨러 (Groq llama-3.3-70b)",
    )
    parser.add_argument("--input",     default=None,        help="입력 CSV (미지정 시 최신 크롤링 파일 자동 선택)")
    parser.add_argument("--output",    default=None,        help="출력 CSV (미지정 시 data/labeled/expressions_auto.csv)")
    parser.add_argument("--no-resume", action="store_true", help="이어하기 비활성화 (처음부터 재시작)")
    args = parser.parse_args()

    input_path = args.input or find_latest_crawled(paths["data"])
    if not input_path:
        print("[오류] data/ 에 크롤링 파일이 없습니다. --input 으로 직접 지정하세요.")
        sys.exit(1)
    print(f"[입력] {input_path}")

    output_path = args.output or os.path.join(paths["labeled"], "expressions_auto.csv")
    label_csv(input_path, output_path, resume=not args.no_resume)
