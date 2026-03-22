"""
labeling/label_reviewer.py

신뢰도가 낮거나 ERROR 인 항목을 터미널에서 직접 검수.

사용법:
  python labeling/label_reviewer.py
  python labeling/label_reviewer.py --input data/labeled/expressions_auto.csv
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_schema import BalanceLabel, get_project_paths

C = {
    "positive" : "\033[92m",
    "negative" : "\033[91m",
    "neutral"  : "\033[93m",
    "unrelated": "\033[90m",
    "ERROR"    : "\033[95m",
    "reset"    : "\033[0m",
    "bold"     : "\033[1m",
    "dim"      : "\033[2m",
}

KEY_MAP = {
    "1": BalanceLabel.POSITIVE.value,
    "2": BalanceLabel.NEGATIVE.value,
    "3": BalanceLabel.NEUTRAL.value,
    "4": BalanceLabel.UNRELATED.value,
    "s": "__skip__",
    "q": "__quit__",
}

LABEL_MENU = (
    f"  {C['bold']}[1]{C['reset']} {C['positive']}positive{C['reset']}   "
    f"{C['bold']}[2]{C['reset']} {C['negative']}negative{C['reset']}   "
    f"{C['bold']}[3]{C['reset']} {C['neutral']}neutral{C['reset']}   "
    f"{C['bold']}[4]{C['reset']} {C['unrelated']}unrelated{C['reset']}   "
    f"{C['dim']}[s] 건너뜀   [q] 저장 후 종료{C['reset']}"
)


def colorize(label: str) -> str:
    return f"{C.get(label, '')}{label}{C['reset']}"


def review_csv(input_path: str):
    df      = pd.read_csv(input_path, dtype={"post_id": str})
    mask    = df["is_reviewed"].isin(["NEEDS_REVIEW", "ERROR"])
    targets = df[mask]
    total   = len(targets)

    if total == 0:
        print("\n검수할 항목이 없습니다. 모두 완료되었습니다.")
        return

    print(f"\n{'='*60}")
    print(f"  {C['bold']}수동 검수 시작{C['reset']} — 총 {total}건")
    print(f"  파일: {input_path}")
    print(f"{'='*60}")
    print(LABEL_MENU)
    print(f"{'='*60}\n")

    reviewed_count = 0

    for idx, (df_idx, row) in enumerate(targets.iterrows(), 1):
        job_class  = str(row.get("job_class",       ""))
        expr_text  = str(row.get("expression_text", ""))
        cur_label  = str(row.get("sentiment",       ""))
        confidence = row.get("confidence", "?")
        reason     = str(row.get("reason",          ""))

        expr_preview = expr_text[:200] + ("..." if len(expr_text) > 200 else "")

        print(f"\n{C['bold']}[{idx}/{total}]{C['reset']}  "
              f"직업: {C['bold']}{job_class}{C['reset']}  |  "
              f"신뢰도: {confidence}  |  post_id: {row.get('post_id','')}")
        print(f"  표현   : {expr_preview}")
        print(f"  AI판정 : {colorize(cur_label)}  |  근거: {reason}")
        print()

        while True:
            try:
                key = input("  레이블 > ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                key = "q"

            if key in KEY_MAP:
                break
            print(f"  [!] 유효한 키: {list(KEY_MAP.keys())}")

        action = KEY_MAP[key]

        if action == "__quit__":
            print("\n  저장 후 종료합니다...")
            break
        elif action == "__skip__":
            print(f"  → 건너뜀 (현재 레이블 유지: {colorize(cur_label)})")
            continue
        else:
            df.at[df_idx, "sentiment"]   = action
            df.at[df_idx, "is_reviewed"] = True
            reviewed_count += 1
            print(f"  → {colorize(action)} 저장됨")

    df.to_csv(input_path, index=False, encoding="utf-8-sig")

    remaining  = (df["is_reviewed"].isin(["NEEDS_REVIEW", "ERROR"])).sum()
    total_done = (df["is_reviewed"] == True).sum()   # noqa: E712

    print(f"\n{'='*60}")
    print(f"  검수 완료: {reviewed_count}건  |  전체 완료: {total_done}건")
    print(f"  남은 검수: {remaining}건")
    print(f"  저장 경로: {input_path}")
    if remaining == 0:
        print(f"\n  모든 검수 완료! 다음 단계:")
        print(f"  → python model/train.py")
    else:
        print(f"\n  내일 이어서: python labeling/label_reviewer.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    paths         = get_project_paths()
    default_input = os.path.join(paths["labeled"], "expressions_auto.csv")

    parser = argparse.ArgumentParser(description="밸런스 감성 레이블 수동 검수")
    parser.add_argument("--input", default=default_input,
                        help=f"검수할 CSV 파일 경로 (기본: {default_input})")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[오류] 파일을 찾을 수 없습니다: {args.input}")
        print("auto_labeler.py 를 먼저 실행하세요.")
        sys.exit(1)

    review_csv(args.input)
