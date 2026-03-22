"""
crawling/crawler.py

인벤 로스트아크 직업 게시판 크롤러.
직업별 게시판에서 게시글(제목/본문/댓글)을 수집해 CSV로 저장.

사용법:
  python crawling/crawler.py
  python crawling/crawler.py --since 2026-01-01
"""

import time
import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling"))
from label_schema import get_project_paths


def load_class_ids():
    paths     = get_project_paths()
    file_path = paths['class_file']

    if not os.path.exists(file_path):
        print(f"설정 파일을 찾을 수 없습니다: {file_path}")
        sys.exit(1)

    class_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                sep = ',' if ',' in line else ':'
                name, board_id = line.split(sep, 1)
                class_list.append({'name': name.strip(), 'id': board_id.strip()})
        return class_list
    except Exception as e:
        print(f"설정 파일 읽기 오류: {e}")
        sys.exit(1)


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


def get_post_details(driver, pc_url):
    """게시물 상세 페이지에서 날짜, 본문, 댓글 수집."""
    try:
        driver.get(pc_url)
        time.sleep(0.5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        date_tag       = soup.select_one('.articleDate')
        exact_date_str = date_tag.text.strip() if date_tag else "2000-01-01 00:00"
        try:
            exact_date_obj = datetime.strptime(exact_date_str, '%Y-%m-%d %H:%M')
        except ValueError:
            exact_date_obj = datetime(2000, 1, 1)

        content_div = soup.select_one('#powerbbsContent')
        content     = content_div.get_text(strip=True, separator=' ') if content_div else ""

        # 댓글은 모바일 페이지에서 수집
        mobile_url = pc_url.replace("www.inven.co.kr", "m.inven.co.kr")
        driver.get(mobile_url)
        try:
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CLASS_NAME, "cmtWrap"))
            )
        except Exception:
            pass

        time.sleep(0.5)
        mobile_soup   = BeautifulSoup(driver.page_source, 'html.parser')
        comment_divs  = mobile_soup.select('div.comment')
        comments_list = [
            div.get_text(separator=' ', strip=True)
            for div in comment_divs
            if len(div.text.strip()) > 1
        ]

        if not comments_list:
            comments_list = [
                span.get_text(separator=' ', strip=True)
                for span in mobile_soup.select('span.cmtContentOne')
            ]

        return exact_date_str, exact_date_obj, content, comments_list, len(comments_list)

    except Exception as e:
        print(f"      상세 수집 오류 ({pc_url}): {e}")
        return "날짜오류", datetime(2000, 1, 1), "", [], 0


def extract_post_id(url: str) -> str:
    """URL 마지막 숫자 세그먼트를 post_id로 추출. 실패 시 URL 해시로 대체."""
    try:
        parts = [p for p in url.rstrip('/').split('/') if p]
        for part in reversed(parts):
            if part.isdigit():
                return part
    except Exception:
        pass
    return str(abs(hash(url)))


def collect_class_data(driver, class_info: dict, target_date_str: str) -> tuple[list, int]:
    """단일 직업 게시판에서 target_date_str 날짜까지 게시물 수집."""
    paths       = get_project_paths()
    class_name  = class_info['name']
    board_id    = class_info['id']
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')

    all_posts      = []
    page           = 1
    is_running     = True
    total_comments = 0

    print(f"\n{'='*60}")
    print(f"[{class_name}] 수집 시작 (타겟: {target_date_str}까지)")
    print(f"{'='*60}")

    while is_running:
        url = f"https://www.inven.co.kr/board/lostark/{board_id}?p={page}"
        driver.get(url)
        time.sleep(1)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        rows = soup.select('.board-list tr:not(.notice)')

        if not rows or page > 500:
            break

        for row in rows:
            link_el = row.select_one('.subject-link')
            if not link_el:
                continue

            link    = link_el['href']
            post_id = extract_post_id(link)

            reco_el   = row.select_one('.reco')
            recommend = reco_el.text.strip() if reco_el else "0"

            exact_date_str, exact_date_obj, content, comments_list, c_count = \
                get_post_details(driver, link)

            if exact_date_obj.date() < target_date.date():
                print(f"\n   {exact_date_obj.date()} 발견. 목표 날짜 도달로 [{class_name}] 종료.")
                is_running = False
                break

            title = link_el.get_text(strip=True)
            print(f"   {exact_date_str[5:16]} | 댓글:{c_count} | {title[:20]}...")

            all_posts.append({
                'post_id'      : post_id,
                'job_class'    : class_name,
                'title'        : title,
                'content'      : content,
                'comments'     : " || ".join(comments_list),
                'comment_count': c_count,
                'recommend'    : recommend,
                'date_clean'   : exact_date_str,
                'url'          : link,
            })
            total_comments += c_count

        if is_running:
            print(f"   {page}페이지 완료 (누적: {len(all_posts)}건)")
            if page % 5 == 0 and all_posts:
                backup_path = os.path.join(paths['backup'], f"backup_{class_name}.csv")
                pd.DataFrame(all_posts).to_csv(backup_path, index=False, encoding='utf-8-sig')
                print(f"   [자동저장] {page}페이지 백업 완료.")
        page += 1

    return all_posts, total_comments


if __name__ == "__main__":
    paths   = get_project_paths()
    classes = load_class_ids()

    parser = argparse.ArgumentParser(description="로스트아크 인벤 직업 게시판 크롤러")
    parser.add_argument(
        "--since",
        default="2026-02-19",
        help="수집 기준 날짜 (이 날짜까지 수집, 형식: YYYY-MM-DD, 기본: 2026-02-19)"
    )
    args = parser.parse_args()
    TARGET_DATE = args.since

    print("\n" + "=" * 50)
    print("수집할 직업을 선택해주세요.")
    print("=" * 50)
    for i, c in enumerate(classes):
        print(f" {i+1}. {c['name']}")
    print(" 0. 전체 직업 수집")
    print("-" * 50)

    user_input = input("번호를 입력하세요 (여러 개: 쉼표 구분, 예: 1,3,5): ").strip()

    if user_input == '0':
        selected_classes = classes
        print("전체 직업 수집을 진행합니다.")
    else:
        indices = [
            int(x.strip()) - 1
            for x in user_input.split(',')
            if x.strip().isdigit()
        ]
        selected_classes = [classes[i] for i in indices if 0 <= i < len(classes)]

        if not selected_classes:
            print("올바른 번호가 입력되지 않았습니다. 프로그램을 종료합니다.")
            sys.exit(0)

        print(f"선택된 직업: {', '.join(c['name'] for c in selected_classes)}")

    final_data           = []
    grand_total_comments = 0
    driver               = setup_driver()

    try:
        for c in selected_classes:
            class_data, c_count = collect_class_data(driver, c, TARGET_DATE)
            final_data.extend(class_data)
            grand_total_comments += c_count

            if final_data:
                temp_path = os.path.join(paths['backup'], "backup_selected_classes_temp.csv")
                pd.DataFrame(final_data).to_csv(temp_path, index=False, encoding='utf-8-sig')

            time.sleep(2)
    finally:
        driver.quit()

    if final_data:
        filename   = f"lostark_crawled_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        final_path = os.path.join(paths['data'], filename)
        pd.DataFrame(final_data).to_csv(final_path, index=False, encoding='utf-8-sig')
        print(f"\n수집 완료! ({len(final_data)}건, 총 댓글 {grand_total_comments}건)")
        print(f"저장 경로: {final_path}")
        print(f"\n다음 단계: python -X utf8 labeling/auto_labeler.py --input {final_path}")
