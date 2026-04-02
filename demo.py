"""
demo.py — 학습된 모델로 문장 감성 분류 대화형 데모
사용법: python demo.py
        python demo.py --mode dealer
        python demo.py --mode job --job 디스트로이어
"""

import os, sys, argparse
import torch
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeling"))
from label_schema import get_project_paths, ID2LABEL, NUM_LABELS, SUPPORT_JOBS

MAX_LENGTH = 128

LABEL_KO = {
    "positive" : "긍정 (밸런스 만족)",
    "negative" : "부정 (밸런스 불만)",
    "neutral"  : "중립 (방향 불분명)",
    "unrelated": "무관 (밸런스 무관)",
}
LABEL_EMOJI = {
    "positive" : "✅",
    "negative" : "❌",
    "neutral"  : "🔘",
    "unrelated": "⬜",
}


def get_model_dir(mode: str, job: str) -> str:
    paths     = get_project_paths()
    ckpt_name = f"job_{job}" if mode == "job" and job else mode
    return os.path.join(paths["root"], "model", "checkpoints", ckpt_name, "best_model")


def predict(model, tokenizer, device, job_class: str, text: str) -> dict:
    input_text = f"직업: {job_class} 표현: {text}"
    encoding   = tokenizer(
        input_text,
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    probs     = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_idx  = probs.argmax()
    label     = ID2LABEL[pred_idx]

    return {
        "label"      : label,
        "confidence" : float(probs[pred_idx]),
        "all_probs"  : {ID2LABEL[i]: float(probs[i]) for i in range(NUM_LABELS)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="unified", choices=["unified", "dealer", "support", "job"])
    parser.add_argument("--job",  default="")
    args = parser.parse_args()

    model_dir = get_model_dir(args.mode, args.job)
    if not os.path.exists(model_dir):
        print(f"[오류] 학습된 모델이 없습니다: {model_dir}")
        print(f"       python model/train.py --mode {args.mode}" + (f" --job {args.job}" if args.job else "") + " 를 먼저 실행하세요.")
        sys.exit(1)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ElectraTokenizerFast.from_pretrained(model_dir)
    model     = ElectraForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    mode_label = {"unified": "통합", "dealer": "딜러", "support": "서폿"}.get(args.mode, f"직업({args.job})")
    print(f"\n{'='*55}")
    print(f"  LOABAL 밸런스 감성 분류 데모  [{mode_label} 모델]")
    print(f"  Device: {device}  |  종료: 'q' 입력")
    print(f"{'='*55}\n")

    while True:
        job_class = input("직업명 입력 (예: 디스트로이어, 엔터 시 '전체'): ").strip() or "전체"
        if job_class.lower() == "q":
            break

        text = input("표현 입력: ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue

        result = predict(model, tokenizer, device, job_class, text)
        label  = result["label"]

        print(f"\n  {LABEL_EMOJI[label]}  {LABEL_KO[label]}  (신뢰도: {result['confidence']*100:.1f}%)")
        print(f"  {'─'*45}")
        for lbl, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"  {LABEL_KO[lbl]:<18} {prob*100:5.1f}%  {bar}")
        print()


if __name__ == "__main__":
    main()
