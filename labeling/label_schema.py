"""
labeling/label_schema.py

레이블 정의 + 프로젝트 경로 유틸 — 파이프라인 전체 공유
"""

import os
import re
import glob
from enum import Enum


def get_project_paths() -> dict:
    root_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir   = os.path.join(root_dir, "data")
    backup_dir = os.path.join(data_dir, "backup")
    label_dir  = os.path.join(data_dir, "labeled")

    os.makedirs(data_dir,   exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(label_dir,  exist_ok=True)

    return {
        "root"      : root_dir,
        "data"      : data_dir,
        "backup"    : backup_dir,
        "labeled"   : label_dir,
        "env"       : os.path.join(root_dir, ".env"),
        "class_file": os.path.join(root_dir, "default", "class.txt"),
    }


# expression-level 감성 레이블
class BalanceLabel(str, Enum):
    POSITIVE  = "positive"
    NEGATIVE  = "negative"
    NEUTRAL   = "neutral"
    UNRELATED = "unrelated"


LABEL_DESCRIPTIONS = {
    BalanceLabel.POSITIVE: (
        "직업 밸런스에 긍정적인 표현. "
        "예: '요즘 딜 잘나온다', '너프 후에도 충분히 강하다', '밸런스 괜찮다'"
    ),
    BalanceLabel.NEGATIVE: (
        "직업 밸런스에 부정적인 표현. 버프/상향 요청, 너프 불만, 약함 호소 등. "
        "예: '너무 약해졌다', '상향 좀 해줘', '딜이 너무 안나온다', '사기캐다 너프해라'"
    ),
    BalanceLabel.NEUTRAL: (
        "밸런스 관련이지만 긍정/부정 방향이 불분명한 표현. "
        "예: '밸런스 패치 언제 하나', '상향됐다는데 체감이 애매하다'"
    ),
    BalanceLabel.UNRELATED: (
        "직업 밸런스와 무관한 표현. "
        "예: 스킬 공략, 팔찌/코어 질문, 파티 모집, 강화 자랑, 단순 잡담"
    ),
}

# LLM 프롬프트용 레이블 가이드
LABEL_GUIDE = "\n".join(
    f'- "{label.value}": {desc}'
    for label, desc in LABEL_DESCRIPTIONS.items()
)

# 모델 학습용 인덱스
LABEL2ID   = {lb.value: i for i, lb in enumerate(BalanceLabel)}
ID2LABEL   = {i: lb.value for i, lb in enumerate(BalanceLabel)}
NUM_LABELS = len(BalanceLabel)

# 서포터 직업 (나머지는 딜러)
SUPPORT_JOBS = {"바드", "홀리나이트", "도화가", "발키리"}


# 크롤러 출력 필수 컬럼
INPUT_COLUMNS = ["post_id", "job_class", "title", "comments"]

# 라벨링 결과 컬럼
EXPRESSION_COLUMNS = [
    "post_id", "job_class", "expression_text",
    "sentiment", "confidence", "reason", "is_reviewed",
]


def build_text(row) -> str:
    """직업 컨텍스트를 포함한 모델 입력 텍스트 구성."""
    job  = str(row.get("job_class",       ""))
    expr = str(row.get("expression_text", "")).strip()
    return f"직업: {job} 표현: {expr}"


def split_into_units(row) -> list[str]:
    """게시글(제목+본문+댓글)을 표현 단위 리스트로 분리."""
    job      = str(row.get("job_class", ""))
    title    = str(row.get("title",    "")).strip()
    content  = str(row.get("content",  "")).strip()
    comments = str(row.get("comments", "")).strip()

    units = []

    if title and title.lower() != "nan" and len(title) >= 3:
        units.append(f"직업: {job} {title}")

    if content and content.lower() != "nan":
        for para in content.split("\n"):
            para = para.strip()
            if not para:
                continue
            for seg in re.split(r"(?<=[.!?])\s+", para):
                seg = seg.strip()
                if len(seg) >= 5:
                    units.append(f"직업: {job} {seg}")

    if comments and comments.lower() != "nan":
        for cmt in comments.split("||"):
            cmt = cmt.strip()
            if len(cmt) >= 5:
                units.append(f"직업: {job} {cmt}")

    if not units:
        units.append(f"직업: {job} (내용없음)")

    return units


def find_latest_crawled(data_dir: str) -> str | None:
    """data/ 에서 가장 최근 lostark_crawled_*.csv 경로 반환."""
    pattern = os.path.join(data_dir, "lostark_crawled_*.csv")
    files   = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None
