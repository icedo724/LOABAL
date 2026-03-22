"""
model/train.py

KoELECTRA-base-v3 파인튜닝 — 로스트아크 직업 밸런스 감성 분류
RTX 4060 8GB 기준으로 최적화 (배치/길이 조정)

사용법:
  python model/train.py                              # 통합 모델 (전체 직업)
  python model/train.py --mode dealer                # 딜러 모델
  python model/train.py --mode support               # 서폿 모델
  python model/train.py --mode job --job 디스트로이어  # 단일 직업 모델
  python model/train.py --epochs 5 --batch 16
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling"))
from label_schema import (
    get_project_paths, LABEL2ID, ID2LABEL, NUM_LABELS,
    BalanceLabel, build_text, EXPRESSION_COLUMNS, SUPPORT_JOBS
)

PRETRAINED_MODEL = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH       = 128
TRAIN_BATCH      = 16
EVAL_BATCH       = 32
EPOCHS           = 5
LR               = 3e-5
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
SEED             = 42
TEST_SIZE        = 0.1
VAL_SIZE         = 0.1
PATIENCE         = 2


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_paths(mode: str = "unified", job: str = ""):
    paths     = get_project_paths()
    ckpt_name = f"job_{job}" if mode == "job" and job else mode
    model_dir = os.path.join(paths["root"], "model", "checkpoints", ckpt_name)
    os.makedirs(model_dir, exist_ok=True)
    return paths, model_dir


class BalanceDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"     : self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx]
                              if "token_type_ids" in self.encodings
                              else torch.zeros(self.encodings["input_ids"].shape[1], dtype=torch.long),
            "labels"        : self.labels[idx],
        }


def load_dataset(labeled_path: str, job_filter: set | None = None):
    """
    라벨링 CSV 로드 → 학습 전처리.
    ERROR / NEEDS_REVIEW 제외, unrelated 다운샘플링 (최대 negative의 1.5배).
    job_filter 지정 시 해당 직업만 사용.
    """
    df = pd.read_csv(labeled_path, dtype={"post_id": str})

    valid_labels = {lb.value for lb in BalanceLabel}
    df = df[df["sentiment"].isin(valid_labels)]
    df = df[df["is_reviewed"] != "NEEDS_REVIEW"]
    df = df[df["sentiment"] != "ERROR"]

    if job_filter is not None:
        df = df[df["job_class"].isin(job_filter)]

    print(f"\n[데이터] 총 {len(df)}건 expression 로드")
    print(df["sentiment"].value_counts().to_string())

    neg_count = (df["sentiment"] == "negative").sum()
    unrel     = df[df["sentiment"] == "unrelated"]
    if len(unrel) > neg_count * 1.5:
        unrel_sampled = unrel.sample(int(neg_count * 1.5), random_state=SEED)
        df = pd.concat([df[df["sentiment"] != "unrelated"], unrel_sampled])
        print(f"[데이터] unrelated 다운샘플링 → {len(unrel_sampled)}건")

    df     = df.reset_index(drop=True)
    texts  = [build_text(row) for _, row in df.iterrows()]
    labels = [LABEL2ID[row["sentiment"]] for _, row in df.iterrows()]

    return texts, labels, df


def train(input_path: str, epochs: int, batch_size: int,
          mode: str = "unified", job: str = ""):
    set_seed(SEED)
    paths, model_dir = get_paths(mode, job)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_jobs = set(pd.read_csv(input_path, dtype={"post_id": str})["job_class"].dropna().unique())
    if mode == "unified":
        job_filter, mode_label = None, "통합"
    elif mode == "dealer":
        job_filter, mode_label = all_jobs - SUPPORT_JOBS, "딜러"
    elif mode == "support":
        job_filter, mode_label = SUPPORT_JOBS, "서폿"
    elif mode == "job":
        if not job:
            print("[오류] --mode job 사용 시 --job 직업명 을 함께 지정하세요.")
            sys.exit(1)
        job_filter, mode_label = {job}, f"직업({job})"
    else:
        print(f"[오류] 알 수 없는 모드: {mode}")
        sys.exit(1)

    print(f"\n[모드] {mode_label}  →  체크포인트: {model_dir}")
    print(f"[환경] Device: {device}")
    if device.type == "cuda":
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    texts, labels, df = load_dataset(input_path, job_filter=job_filter)
    if len(texts) < 100:
        print(f"\n[경고] 학습 데이터가 {len(texts)}건으로 너무 적습니다.")
        print("       최소 200건 이상 권장, 500건 이상이면 포트폴리오 수치가 안정적으로 나옵니다.")
        if len(texts) < 50:
            print("[중단] 50건 미만 — 라벨링을 더 진행 후 재시도하세요.")
            sys.exit(1)

    indices = list(range(len(texts)))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        texts, labels, indices, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=SEED, stratify=y_train
    )

    print(f"\n[분할] Train {len(X_train)} / Val {len(X_val)} / Test {len(X_test)}")

    # evaluate.py 가 동일한 test set 을 사용하도록 저장
    test_split_path = os.path.join(model_dir, "test_split.csv")
    df.iloc[idx_test].to_csv(test_split_path, index=False, encoding="utf-8-sig")
    print(f"[저장] Test split → {test_split_path}")

    print(f"\n[모델] {PRETRAINED_MODEL} 로드 중...")
    tokenizer = ElectraTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    model     = ElectraForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_ds = BalanceDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_ds   = BalanceDataset(X_val,   y_val,   tokenizer, MAX_LENGTH)
    test_ds  = BalanceDataset(X_test,  y_test,  tokenizer, MAX_LENGTH)

    # Windows에서 num_workers > 0 은 멀티프로세싱 오류 유발
    num_workers  = 0 if sys.platform == "win32" else 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=EVAL_BATCH, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=EVAL_BATCH, shuffle=False, num_workers=num_workers, pin_memory=True)

    raw_weights   = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=np.array(y_train))
    weight_tensor = torch.tensor(raw_weights, dtype=torch.float).to(device)
    loss_fn       = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    print(f"\n[클래스 가중치] { {ID2LABEL[i]: round(w, 3) for i, w in enumerate(raw_weights)} }")

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    no_improve  = 0
    best_ckpt   = os.path.join(model_dir, "best_model")
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n[학습 시작] Epochs={epochs}, Batch={batch_size}, LR={LR}, MaxLen={MAX_LENGTH}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            batch_labels   = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(outputs.logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            if step % 20 == 0:
                avg = train_loss / step
                print(f"  Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | Loss {avg:.4f}")

        model.eval()
        val_preds_all = []
        val_true_all  = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                batch_labels   = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                preds   = outputs.logits.argmax(dim=-1).cpu()
                val_preds_all.extend(preds.tolist())
                val_true_all.extend(batch_labels.tolist())

        val_f1   = f1_score(val_true_all, val_preds_all, average="macro")
        val_acc  = accuracy_score(val_true_all, val_preds_all)
        avg_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch}/{epochs} 완료 | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve  = 0
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            print(f"  최고 성능 갱신 (Macro F1: {val_f1:.4f}) → {best_ckpt}")
        else:
            no_improve += 1
            print(f"  [Early Stop] {no_improve}/{PATIENCE} 연속 미개선")
            if no_improve >= PATIENCE:
                print(f"  [Early Stop] {PATIENCE} epoch 연속 개선 없음 → 학습 조기 종료")
                break

        print()

    print(f"{'='*60}")
    print("[최종 평가] 베스트 체크포인트로 테스트셋 평가")

    best_model = ElectraForSequenceClassification.from_pretrained(best_ckpt).to(device)
    best_model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            batch_labels   = batch["labels"]

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            preds   = outputs.logits.argmax(dim=-1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_labels.numpy())

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
    cm     = confusion_matrix(all_labels, all_preds)

    print(f"\n{report}")
    print("혼동 행렬 (Confusion Matrix):")
    print(f"  {'':12}" + "  ".join(f"{n[:8]:>8}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:<12}" + "  ".join(f"{v:>8}" for v in row))

    result_path = os.path.join(model_dir, f"eval_result_{timestamp}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Model   : {PRETRAINED_MODEL}\n")
        f.write(f"Mode    : {mode_label}\n")
        f.write(f"Trained : {timestamp}\n")
        f.write(f"Epochs  : {epochs}  |  Batch: {batch_size}  |  MaxLen: {MAX_LENGTH}\n")
        f.write(f"Best Val Macro F1: {best_val_f1:.4f}\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(f"  {'':12}" + "  ".join(f"{n[:8]:>8}" for n in label_names) + "\n")
        for i, row in enumerate(cm):
            f.write(f"  {label_names[i]:<12}" + "  ".join(f"{v:>8}" for v in row) + "\n")

    print(f"\n결과 저장 → {result_path}")
    print(f"모델 저장 → {best_ckpt}")
    print(f"\n다음 단계: python model/evaluate.py --mode {mode}" + (f" --job {job}" if mode == "job" else ""))


if __name__ == "__main__":
    paths, _ = get_paths()
    default_input = os.path.join(paths["labeled"], "expressions_auto.csv")

    parser = argparse.ArgumentParser(description="KoELECTRA 밸런스 감성 분류 파인튜닝")
    parser.add_argument("--input",  default=default_input, help=f"라벨링 CSV 경로 (기본: {default_input})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,      help=f"학습 에포크 (기본: {EPOCHS})")
    parser.add_argument("--batch",  type=int, default=TRAIN_BATCH, help=f"배치 크기 (기본: {TRAIN_BATCH}, OOM 시 8로)")
    parser.add_argument(
        "--mode",
        default="unified",
        choices=["unified", "dealer", "support", "job"],
        help="학습 모드: unified(전체) | dealer(딜러) | support(서폿) | job(단일 직업)",
    )
    parser.add_argument("--job", default="", help="--mode job 사용 시 직업명 (예: 디스트로이어)")
    args = parser.parse_args()

    train(args.input, args.epochs, args.batch, mode=args.mode, job=args.job)
