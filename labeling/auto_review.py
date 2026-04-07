"""
labeling/auto_review.py

NEEDS_REVIEW 항목 자동 판정.
- 명확한 케이스: 키워드 기반 자동 레이블링 → is_reviewed = "True"
- 빈 표현(ERROR/nan): unrelated 처리
- 판단 어려운 케이스: is_reviewed = "MANUAL" 로 표기 → label_reviewer.py 에서 처리

사용법:
  python labeling/auto_review.py
"""

import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_schema import get_project_paths

# ── 키워드 규칙 ──────────────────────────────────────────────
# 우선순위: negative > positive > unrelated > neutral 순으로 매칭

NEG_PATTERNS = [
    r"버프\s*(해줘|요청|좀|바람|필요|해야|받아야|달라|줘|ㅠ|ㅜ)",
    r"너프\s*(당했|됐|된|먹었|받았|맞았|이후|전|후)",
    r"(너무|너무나|매우|정말|진짜)\s*(약|구림|별로|부족|딸리|처참|비참)",
    r"약(해졌|해서|한데|하다|하네|하죠|합니다|ㅠ|ㅜ)",
    r"(딜|데미지|뎀|피해|성능|효율).{0,5}(너무\s*)?(약|구림|별로|딸리|처참|낮|부족)",
    r"(딜|데미지|뎀)\s*(못\s*)?(나온|나와|나옴|안나|안 나)",
    r"(하향|상향\s*요청|상향\s*해줘|하향\s*됐|하향\s*당했)",
    r"(구림|구리다|구려|구렸|구리네|구린데|구리죠)",
    r"(처참|비참|망했|망함|망겜|폐직|쓰레기)",
    r"(좀\s*올려줘|좀\s*강화해|좀\s*개선해|고쳐줘|고쳐야|개선해줘|개선\s*요청)",
    r"(드러눕|●▅▇|▇━━●|━━●━|●━━)",
    r"ㅠㅠ|ㅜㅜ|ㅠ\.ㅠ|흑흑|흑ㅠ|슬프|억울",
    r"(불합리|불균형|역차별|차별|편애)",
    r"(왜\s*이렇게\s*(약|딸리|구림|별로))",
    r"(밸런스\s*(붕괴|망|이상|최악|글러|문제))",
    r"개편.{0,5}(받아야|필요|해야|요청|줘|달라)",
    r"(픽스|수정|패치).{0,5}(해줘|요청|바람|필요|달라)",
]

POS_PATTERNS = [
    r"(강해졌|강해서|강하다|강하네|강하죠|강합니다)",
    r"(딜|데미지|뎀|성능|효율).{0,5}(잘|좋|충분|괜찮|나쁘지\s*않|ㄱㅊ)",
    r"(딜|데미지|뎀)\s*(잘\s*)?(나온|나와|나옴)",
    r"(너프\s*(필요\s*없|할\s*필요|하면\s*안|반대|반발))",
    r"(충분히?\s*(강|좋|됩니|쓸만|괜찮))",
    r"(쓸만하|쓸만합니|나쁘지\s*않|괜찮은\s*편|상향\s*됐|상향\s*받았|상향\s*이후)",
    r"(지금\s*상태\s*(좋|충분|괜찮|적당))",
    r"(최강|최고\s*(직업|딜러|성능)|op(?:급|임|이다))",
    r"(버프\s*(됐|받았|이후|먹었|올라|올랐))",
]

UNRELATED_PATTERNS = [
    # 장비/스펙 수치
    r"(공속|치적|치명|신속|특화|제압|인내)\s*\d+",
    r"\d+\s*(공속|치적|치명|신속|특화|제압|인내)",
    r"(팔찌|악세서리|목걸이|귀걸이|반지).{0,20}(추천|어떻게|어떤|어디|뭐|쓰세요|쓰면)",
    r"(각인|각인서).{0,20}(추천|세팅|조합|어떻게|어떤)",
    r"(빌드|세팅|장비).{0,10}(추천|질문|알려|어떻게|공유|뭐가|어떤)",
    r"(강화|초월|연마|연각|가호|초각).{0,10}(추천|어디|몇|어떻게)",
    r"(파티|공격대|레이드)\s*(구합|찾습|모집)",
    r"(보석|겁작|작열|포강|집속기).{0,15}(추천|어떻게|어디|질문|어떤|쓰면|맞춰)",
    r"(몇\s*강|아이템\s*레벨|아이레|아템레)",
    r"(스킬\s*(트리|순서|조합|세팅|추천|레벨))",
    r"(공략|가이드|입문|시작\s*하려)",
    r"(거래소|마켓|시세|가격|살까|팔까)",
    # 스킬/로테이션
    r"(쿨타임|쿨\s*돌|쿨\s*맞|쿨\s*소화|쿨\s*초기화|쿨증)",
    r"(해방|차징|코어|가동률).{0,30}(하면|되면|나와|나옴|됩니다|쓰면|씁니다|써요|됩니|올라)",
    r"(1코어|2코어|3코어|4코어|5코어)",
    r"(집속기|사이즈믹|퍼스트|어웨이크|인듀어|파숄|헤크|어스이터|풀스윙).{0,20}(순서|먼저|쓰고|사용|하면|넣어|지분|줄이|올리)",
    r"(로테이션|시너지\s*유지|가동률|중충|중력충전|아덴충전)",
    # 빌드 이름 나열/비교 (직업 슬랭 빌드명)
    r"(붕쯔|분망|차붕|특이점|즉발\s*빌드|고기\s*빌드).{0,20}(비교|vs|차이|고점|저점|어떤|선택|쓸까|나을|좋은)",
    r"(붕쯔|분망|차붕|특이점).{0,5}(고점|저점|가동|유지|비교)",
    r"고점.{0,10}저점|저점.{0,10}고점",
    # 각인 이름 나열
    r"(결대|원한|바리|저받|적주피|치피|딜증|공증).{0,5}(깎|줄|빼|넣|추가|채용|조합)",
    # DPS 수치 (감정 없는 수치 보고)
    r"\d+억.{0,10}(나온|나와|찍|됩|입니다|이에요|이야|봤는데|까진)",
    r"(?i)dps\s*\d+|\d+\s*dps",
    r"\d{1,2}\s*억\s*\d{1,4}",
    # 타수/공속 계산
    r"\d+\.?\d*\s*타",
    r"(헤드\s*어택|헤드\s*추노|포지셔닝|피면기|피격면역)",
    r"(체감\s*난이도|체감\s*차이).{0,15}(내려|올라|높|낮|크|작)",
    # 업그레이드 투자 논의
    r"(가성비|투자|골드).{0,20}(어떻게|어디|추천|이득|손해|나을까|할까|효율)",
    r"(투력|딜증|딜지분).{0,20}(차이|얼마|몇|유의미|올라|내려)",
    r"(방\s*\d+강|작\s*\d+|상상\s*악세|유각|유물\s*악세)",
    # 수치/스탯 비교 문장
    r"\d+\.?\d*\s*%",
    r"(치적|공속|신속|특화).{0,10}(높|낮|맞춰|올려|내려|먹|효율)",
    r"(오른쪽|왼쪽)\s*(팔찌|악세|세팅|코어)",
    r"(재빠공|재빠|빠공|현공).{0,10}(유물|전설|고대|이상|이하)",
    # 짧은 단순 반응 (감정 판단 불가)
    r"^.{1,10}\s*$",  # 10자 이하
    r"^(넵|넹|ㅇㅇ|ㄱㄱ|ㅋㅋ+|ㅎㅎ+|감사|고맙|알겠|알겠습|이해했|맞아요|맞습니다)\s*[.!]?\s*$",
    # 데미지 합산/수치 관련 기술적 질문
    r"(데미지\s*합산|딜\s*측정|dps\s*측정)",
    r"(합산\s*(켜고|끄고|차이))",
]

NEG_RE     = [re.compile(p) for p in NEG_PATTERNS]
POS_RE     = [re.compile(p) for p in POS_PATTERNS]
UNREL_RE   = [re.compile(p) for p in UNRELATED_PATTERNS]

# neutral 판정 유지 조건: 밸런스 관련 키워드가 존재해야 neutral로 인정
NEUTRAL_BALANCE_RE = re.compile(
    r"(패치|밸런스|버프|너프|상향|하향|개편|체급|밸패|강화\s*해|약화|조정|조율|균형"
    r"|언제\s*(해줄|올려줄|패치|버프|올까|해줘)"
    r"|기대|기다리|기대된다|기다려진다"
    r"|체감\s*(이상|아님|없|됩니|했|되|안)"  # 성능 체감 이상 언급
    r"|밸붕|밸런붕|밸런스붕괴)"
)


def judge(expr: str, cur_label: str, confidence: float) -> tuple[str, bool]:
    """
    (label, is_confident) 반환.
    is_confident=False → MANUAL 로 표기
    """
    if not isinstance(expr, str) or expr.strip() == "" or expr == "nan":
        return "unrelated", True  # 빈 표현

    text = expr.strip()

    # 명확한 negative/positive 우선 확인
    for r in NEG_RE:
        if r.search(text):
            return "negative", True

    for r in POS_RE:
        if r.search(text):
            return "positive", True

    # 명확한 unrelated 패턴
    for r in UNREL_RE:
        if r.search(text):
            return "unrelated", True

    # AI가 negative/positive/unrelated로 판정했고 신뢰도 0.6 이상 → 신뢰
    if confidence >= 0.6 and cur_label in ("negative", "positive", "unrelated"):
        return cur_label, True

    # AI가 neutral로 판정한 경우:
    # 밸런스 키워드가 있으면 neutral 유지
    if cur_label == "neutral" and NEUTRAL_BALANCE_RE.search(text):
        return "neutral", True

    # 밸런스 키워드 없이 neutral → 실제로는 unrelated (스펙/빌드/잡담)
    if cur_label == "neutral":
        return "unrelated", True

    # 그 외 (ERROR 등) → MANUAL
    return cur_label, False


def main():
    paths      = get_project_paths()
    input_path = os.path.join(paths["labeled"], "expressions_auto.csv")

    df = pd.read_csv(input_path, dtype={"post_id": str, "is_reviewed": str})

    mask    = df["is_reviewed"].isin(["NEEDS_REVIEW", "ERROR"])
    targets = df[mask].copy()
    total   = len(targets)

    if total == 0:
        print("검수할 항목이 없습니다.")
        return

    print(f"[자동 판정 시작]  대상 {total}건")

    auto_count   = 0
    manual_count = 0

    for df_idx, row in targets.iterrows():
        expr       = row.get("expression_text", "")
        cur_label  = str(row.get("sentiment", ""))
        confidence = float(row.get("confidence", 0.0))

        label, confident = judge(expr, cur_label, confidence)

        if confident:
            df.at[df_idx, "sentiment"]   = label
            df.at[df_idx, "is_reviewed"] = "True"
            auto_count += 1
        else:
            df.at[df_idx, "is_reviewed"] = "MANUAL"
            manual_count += 1

    df.to_csv(input_path, index=False, encoding="utf-8-sig")

    print(f"  자동 처리: {auto_count}건")
    print(f"  수동 필요: {manual_count}건 (is_reviewed='MANUAL')")
    print(f"  저장 완료: {input_path}")

    if manual_count > 0:
        print(f"\n  수동 검수 실행: python labeling/label_reviewer.py")


if __name__ == "__main__":
    main()
