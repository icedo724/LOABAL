"""
model/evaluate.py

학습된 모델 상세 평가 + 직업별 밸런스 만족도 집계 (expression-level)

[평가 구조]
  Part 1 — 모델 품질 (Accuracy / F1 / Confusion Matrix)
           train.py 가 저장한 test_split.csv 만 사용 (data leakage 방지)
  Part 2 — 직업별 만족도
           전체 expression 데이터에 모델 예측 적용 → 포스트별 비율 → 직업별 집계

사용법:
  python model/evaluate.py                              # 통합 모델
  python model/evaluate.py --mode dealer                # 딜러 모델
  python model/evaluate.py --mode support               # 서폿 모델
  python model/evaluate.py --mode job --job 디스트로이어  # 단일 직업 모델
  python model/evaluate.py --input data/labeled/expressions_auto.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling"))
from label_schema import get_project_paths, LABEL2ID, ID2LABEL, NUM_LABELS, BalanceLabel, build_text, SUPPORT_JOBS

MAX_LENGTH = 128
EVAL_BATCH = 32


def get_dirs(mode: str = "unified", job: str = ""):
    paths     = get_project_paths()
    ckpt_name = f"job_{job}" if mode == "job" and job else mode
    ckpt_dir  = os.path.join(paths["root"], "model", "checkpoints", ckpt_name)
    model_dir = os.path.join(ckpt_dir, "best_model")
    return paths, ckpt_dir, model_dir


def _run_model(model, tokenizer, device, df) -> tuple[list, list]:
    """df 의 모든 expression 행에 대해 모델 추론 → (preds, probs) 반환."""
    texts = [build_text(row) for _, row in df.iterrows()]
    total = len(texts)
    all_preds, all_probs = [], []

    for start in range(0, total, EVAL_BATCH):
        end      = min(start + EVAL_BATCH, total)
        encoding = tokenizer(
            texts[start:end],
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
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(probs.argmax(axis=-1).tolist())
        all_probs.extend(probs.tolist())

    return all_preds, all_probs


def evaluate_quality(model, tokenizer, device, ckpt_dir: str):
    """test_split.csv 기준 모델 품질 평가 (Accuracy / F1 / Confusion Matrix)."""
    test_split_path = os.path.join(ckpt_dir, "test_split.csv")

    if not os.path.exists(test_split_path):
        print("\n[경고] test_split.csv 없음. python model/train.py 를 먼저 실행하세요.")
        print("       모델 품질 평가를 건너뜁니다.\n")
        return

    df_test     = pd.read_csv(test_split_path, dtype={"post_id": str})
    true_labels = [LABEL2ID[row["sentiment"]] for _, row in df_test.iterrows()]

    print(f"\n[Part 1] 모델 품질 — test split {len(df_test)}개 expression (훈련 미사용)")
    all_preds, _ = _run_model(model, tokenizer, device, df_test)

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report      = classification_report(true_labels, all_preds, target_names=label_names, digits=4)
    cm          = confusion_matrix(true_labels, all_preds)
    overall_acc = accuracy_score(true_labels, all_preds)
    macro_f1    = f1_score(true_labels, all_preds, average="macro")
    weighted_f1 = f1_score(true_labels, all_preds, average="weighted")

    print("=" * 60)
    print("  모델 성능 지표 (Expression-level Test Set)")
    print("=" * 60)
    print(f"  Accuracy   : {overall_acc:.4f}")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print()
    print(report)

    print("  혼동 행렬:")
    header = f"  {'실제↓ 예측→':14}" + "  ".join(f"{n[:9]:>9}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:<14}" + "  ".join(f"{v:>9}" for v in row))


def evaluate_satisfaction(model, tokenizer, device, labeled_path: str, data_dir: str,
                          job_filter: set | None = None, mode_label: str = "통합"):
    """전체 expression에 모델 적용 → 포스트별/직업별 만족도 집계."""
    df = pd.read_csv(labeled_path, dtype={"post_id": str})
    valid_labels = {lb.value for lb in BalanceLabel}
    df = df[df["sentiment"].isin(valid_labels)].copy()
    df = df[df["is_reviewed"] != "NEEDS_REVIEW"]
    df = df[df["sentiment"] != "ERROR"]

    if job_filter is not None:
        df = df[df["job_class"].isin(job_filter)]

    df = df.reset_index(drop=True)

    print(f"\n[Part 2] [{mode_label}] 직업별 만족도 — {len(df)}개 expression에 모델 적용")
    all_preds, all_probs = _run_model(model, tokenizer, device, df)

    df["pred_label"]      = [ID2LABEL[p] for p in all_preds]
    df["pred_confidence"] = [round(float(max(p)), 3) for p in all_probs]

    post_rows = []
    for pid, grp in df.groupby("post_id"):
        cnts  = grp["pred_label"].value_counts().to_dict()
        pos   = cnts.get("positive",  0)
        neg   = cnts.get("negative",  0)
        neu   = cnts.get("neutral",   0)
        unrel = cnts.get("unrelated", 0)
        bal   = pos + neg + neu
        score = round(pos / bal, 4) if bal > 0 else None

        post_rows.append({
            "post_id"           : pid,
            "job_class"         : grp["job_class"].iloc[0],
            "pos_count"         : pos,
            "neg_count"         : neg,
            "neu_count"         : neu,
            "unrel_count"       : unrel,
            "balance_total"     : bal,
            "satisfaction_score": score,
        })

    post_df   = pd.DataFrame(post_rows)
    post_path = os.path.join(data_dir, f"post_satisfaction_{mode_label}.csv")
    post_df.to_csv(post_path, index=False, encoding="utf-8-sig")
    print(f"  포스트별 만족도 저장 → {post_path}")

    print("\n" + "=" * 60)
    print("  직업별 밸런스 만족도 분석 (expression 비율 기반)")
    print("=" * 60)

    summary_rows = []
    for job in sorted(df["job_class"].unique()):
        sub   = df[df["job_class"] == job]
        cnts  = sub["pred_label"].value_counts().to_dict()
        pos   = cnts.get("positive",  0)
        neg   = cnts.get("negative",  0)
        neu   = cnts.get("neutral",   0)
        unrel = cnts.get("unrelated", 0)
        bal   = pos + neg + neu
        score = round(pos / bal * 100, 1) if bal > 0 else 0.0

        n_posts = sub["post_id"].nunique()
        summary_rows.append({
            "job_class"   : job,
            "post_count"  : n_posts,
            "expr_total"  : len(sub),
            "bal_related" : bal,
            "positive"    : pos,
            "negative"    : neg,
            "neutral"     : neu,
            "unrelated"   : unrel,
            "satisfaction": score,
        })

        bar_pos = "█" * int(score / 5)
        bar_neg = "░" * int((neg / bal * 100) / 5) if bal > 0 else ""
        print(
            f"  {job:<14} | 만족도 {score:5.1f}% "
            f"| 긍정 {pos:>4} 부정 {neg:>4} 중립 {neu:>4} 무관 {unrel:>4} "
            f"| {bar_pos}{bar_neg}"
        )

    summary_df   = pd.DataFrame(summary_rows).sort_values("satisfaction", ascending=False)
    summary_path = os.path.join(data_dir, f"job_satisfaction_{mode_label}.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n  직업별 만족도 저장 → {summary_path}")

    print("\n" + "=" * 60)
    print("  밸런스 만족도 TOP / BOTTOM 5")
    print("=" * 60)
    print("  [상위 5 — 만족도 높음]")
    for _, r in summary_df.head(5).iterrows():
        print(f"    {r['job_class']:<14} {r['satisfaction']:5.1f}%  (포스트 {r['post_count']}건)")

    print("  [하위 5 — 불만족 높음]")
    for _, r in summary_df.tail(5).iterrows():
        print(f"    {r['job_class']:<14} {r['satisfaction']:5.1f}%  (포스트 {r['post_count']}건)")

    print(f"\n  다음 단계: python model/inference.py  (신규 데이터 추론)")


def evaluate(labeled_path: str, mode: str = "unified", job: str = ""):
    paths, ckpt_dir, model_dir = get_dirs(mode, job)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_dir):
        print(f"[오류] 학습된 모델이 없습니다: {model_dir}")
        print(f"       먼저 python model/train.py --mode {mode}"
              + (f" --job {job}" if mode == "job" else "") + " 를 실행하세요.")
        sys.exit(1)

    all_jobs = set(pd.read_csv(labeled_path, dtype={"post_id": str})["job_class"].dropna().unique())
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

    print(f"\n[모드] {mode_label}  |  [모델 로드] {model_dir}")
    tokenizer = ElectraTokenizerFast.from_pretrained(model_dir)
    model     = ElectraForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    evaluate_quality(model, tokenizer, device, ckpt_dir)
    evaluate_satisfaction(model, tokenizer, device, labeled_path, paths["data"],
                          job_filter=job_filter, mode_label=mode_label)


if __name__ == "__main__":
    paths         = get_project_paths()
    default_input = os.path.join(paths["labeled"], "expressions_auto.csv")

    parser = argparse.ArgumentParser(description="KoELECTRA 모델 평가 + 직업별 만족도 집계")
    parser.add_argument("--input", default=default_input, help=f"expression CSV 경로 (기본: {default_input})")
    parser.add_argument(
        "--mode",
        default="unified",
        choices=["unified", "dealer", "support", "job"],
        help="평가 모드: unified(전체) | dealer(딜러) | support(서폿) | job(단일 직업)",
    )
    parser.add_argument("--job", default="", help="--mode job 사용 시 직업명 (예: 디스트로이어)")
    args = parser.parse_args()

    evaluate(args.input, mode=args.mode, job=args.job)
