"""
model/inference.py

학습된 KoELECTRA 로 신규 크롤링 데이터 추론 (expression-level).

처리 흐름:
  1. 게시글 → split_into_units() 로 표현 단위 분리
  2. 각 표현에 대해 모델 분류 (positive/negative/neutral/unrelated)
  3. 게시글별 집계 → 긍정/부정 비율, 만족도 점수 산출

사용법:
  python model/inference.py                              # 통합 모델
  python model/inference.py --mode dealer                # 딜러 모델
  python model/inference.py --mode support               # 서폿 모델
  python model/inference.py --mode job --job 디스트로이어  # 단일 직업 모델
  python model/inference.py --input data/lostark_crawled_20260322_1200.csv
  python model/inference.py --input data/new.csv --output data/result.csv
"""

import os
import sys
import argparse
import pandas as pd
import torch
from collections import defaultdict
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling"))
from label_schema import get_project_paths, ID2LABEL, NUM_LABELS, build_text, find_latest_crawled, split_into_units, SUPPORT_JOBS

MAX_LENGTH = 128
BATCH_SIZE = 64


def get_model_dir(mode: str = "unified", job: str = "") -> str:
    paths     = get_project_paths()
    ckpt_name = f"job_{job}" if mode == "job" and job else mode
    return os.path.join(paths["root"], "model", "checkpoints", ckpt_name, "best_model")


def run_inference(input_path: str, output_path: str,
                  mode: str = "unified", job: str = ""):
    paths     = get_project_paths()
    model_dir = get_model_dir(mode, job)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_dir):
        print(f"[오류] 학습된 모델 없음: {model_dir}")
        print(f"       python model/train.py --mode {mode}"
              + (f" --job {job}" if mode == "job" else "") + " 를 먼저 실행하세요.")
        sys.exit(1)

    all_jobs = set(pd.read_csv(input_path, dtype={"post_id": str})["job_class"].dropna().unique())
    if mode == "unified":
        job_filter, mode_label = None, "통합"
    elif mode == "dealer":
        job_filter, mode_label = all_jobs - SUPPORT_JOBS, "딜러"
    elif mode == "support":
        job_filter, mode_label = SUPPORT_JOBS, "서폿"
    elif mode == "job":
        job_filter, mode_label = {job}, f"job_{job}"
    else:
        print(f"[오류] 알 수 없는 모드: {mode}")
        sys.exit(1)

    print(f"[모드] {mode_label}  |  [모델 로드] {model_dir}")
    tokenizer = ElectraTokenizerFast.from_pretrained(model_dir)
    model     = ElectraForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    df = pd.read_csv(input_path, dtype={"post_id": str})
    if job_filter is not None:
        df = df[df["job_class"].isin(job_filter)].reset_index(drop=True)
    total = len(df)
    print(f"[추론] {total}건 포스트 처리 중...\n")

    unit_records = []
    for _, row in df.iterrows():
        for unit_text in split_into_units(row):
            unit_records.append({
                "post_id"  : str(row["post_id"]),
                "job_class": str(row.get("job_class", "")),
                "unit_text": unit_text,
            })

    unit_texts = [r["unit_text"] for r in unit_records]
    n_units    = len(unit_texts)
    print(f"  표현 단위 총 {n_units}개 생성 (포스트당 평균 {n_units/total:.1f}개)")

    all_preds = []
    for start in range(0, n_units, BATCH_SIZE):
        end      = min(start + BATCH_SIZE, n_units)
        encoding = tokenizer(
            unit_texts[start:end],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding.get(
            "token_type_ids", torch.zeros_like(encoding["input_ids"])
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)

        if (start // BATCH_SIZE + 1) % 10 == 0:
            print(f"  {end}/{n_units} 표현 처리 완료")

    post_agg = defaultdict(lambda: {"pos": 0, "neg": 0, "neu": 0, "unrel": 0})
    for rec, pred in zip(unit_records, all_preds):
        lbl = ID2LABEL[pred]
        pid = rec["post_id"]
        if   lbl == "positive" : post_agg[pid]["pos"]   += 1
        elif lbl == "negative" : post_agg[pid]["neg"]   += 1
        elif lbl == "neutral"  : post_agg[pid]["neu"]   += 1
        else                   : post_agg[pid]["unrel"] += 1

    out_rows = []
    for _, row in df.iterrows():
        pid  = str(row["post_id"])
        agg  = post_agg[pid]
        pos, neg, neu, unrel = agg["pos"], agg["neg"], agg["neu"], agg["unrel"]
        bal  = pos + neg + neu

        r = row.to_dict()
        r.update({
            "pos_count"         : pos,
            "neg_count"         : neg,
            "neu_count"         : neu,
            "unrel_count"       : unrel,
            "balance_total"     : bal,
            "satisfaction_score": round(pos / bal, 4) if bal > 0 else None,
            "negativity_score"  : round(neg / bal, 4) if bal > 0 else None,
        })
        out_rows.append(r)

    result_df = pd.DataFrame(out_rows)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  추론 완료: {total}건 포스트 → {output_path}")
    print(f"{'='*60}")
    print(f"  {'직업':<14} | {'긍정':>5} {'부정':>5} {'중립':>5} {'무관':>5} | 만족도")
    print(f"  {'-'*58}")

    for job_name in sorted(result_df["job_class"].unique()):
        sub   = result_df[result_df["job_class"] == job_name]
        pos   = sub["pos_count"].sum()
        neg   = sub["neg_count"].sum()
        neu   = sub["neu_count"].sum()
        unrel = sub["unrel_count"].sum()
        bal   = sub["balance_total"].sum()
        score = round(pos / bal * 100, 1) if bal > 0 else 0.0
        print(f"  {job_name:<14} | {pos:>5} {neg:>5} {neu:>5} {unrel:>5} | {score:.1f}%")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    paths = get_project_paths()

    parser = argparse.ArgumentParser(description="KoELECTRA expression-level 밸런스 감성 추론")
    parser.add_argument("--input",  default=None, help="입력 CSV (미지정 시 최신 크롤링 파일 자동 선택)")
    parser.add_argument("--output", default=None, help="출력 CSV (미지정 시 data/inference_result_{mode}.csv)")
    parser.add_argument(
        "--mode",
        default="unified",
        choices=["unified", "dealer", "support", "job"],
        help="추론 모드: unified(전체) | dealer(딜러) | support(서폿) | job(단일 직업)",
    )
    parser.add_argument("--job", default="", help="--mode job 사용 시 직업명 (예: 디스트로이어)")
    args = parser.parse_args()

    input_path = args.input or find_latest_crawled(paths["data"])
    if not input_path:
        print("[오류] data/ 에 크롤링 파일이 없습니다.")
        sys.exit(1)
    print(f"[자동 탐색] 입력: {input_path}")

    mode_label  = f"job_{args.job}" if args.mode == "job" and args.job else args.mode
    output_path = args.output or os.path.join(paths["data"], f"inference_result_{mode_label}.csv")
    run_inference(input_path, output_path, mode=args.mode, job=args.job)
