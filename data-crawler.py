# 파일 이름: data_crawler.py (개선된 버전)

import requests
from bs4 import BeautifulSoup
import time
import os # 폴더 관리를 위해 추가

# ==========================================================
# 🚨🚨 여기를 네 정보로 다시 정확히 수정해야 합니다! 🚨🚨
# ==========================================================
BLOG_ID = "kobau68"
START_POST_NUM = 224087453042 # 6개월 전 포스트 번호
END_POST_NUM = 2224095240255   # 가장 최근 글 번호
# ==========================================================


def extract_post_data(post_num):
    """단일 포스트에서 날짜, 제목, 본문을 추출합니다."""
    url = f"https://blog.naver.com/PostView.naver?blogId={BLOG_ID}&logNo={post_num}"
    
    # User-Agent 추가: '나는 웹 브라우저다'라고 네이버에 알려줍니다.
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # 헤더를 포함하여 요청합니다.
        response = requests.get(url, headers=headers) 
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.text, 'html.parser')

        # [필수 검토] 만약 여기서 오류가 나면 네이버 블로그 디자인이 바뀌었을 가능성이 높습니다.
        title_element = soup.select_one('.se-viewer .se-title-text')
        date_element = soup.select_one('.se-viewer .date-info')
        
        # 요소가 없으면 건너뜁니다.
        if not title_element or not date_element:
            print(f"❌ 실패: {post_num}번 글은 형식이 맞지 않거나 비공개입니다.")
            return None

        title = title_element.text.strip()
        date = date_element.text.strip()
        
        content_paragraphs = [p.text for p in soup.select('.se-main-container p')]
        content = "\n".join(content_paragraphs)
        
        print(f"✅ 성공: {post_num}번 글 ({title})")
        return f"날짜: {date}\n제목: {title}\n본문:\n{content}\n\n---\n\n"
        
    except Exception as e:
        print(f"❌ 오류 발생: {post_num}번 글 - {e}")
        return None

# --- 메인 실행 부분 ---

# 폴더가 없으면 만듭니다. (원인 1 해결)
output_dir = "./data_raw"
os.makedirs(output_dir, exist_ok=True) 

all_diaries = ""
# 시작 번호부터 끝 번호까지 반복합니다. (실행 시간 단축을 위해 2초씩 쉬도록 수정합니다.)
for num in range(START_POST_NUM, END_POST_NUM + 1):
    diary_text = extract_post_data(num)
    if diary_text:
        all_diaries += diary_text
    
    time.sleep(2) # 차단을 피하기 위해 2초씩 쉽니다.

# 최종적으로 추출된 데이터를 파일로 저장합니다.
output_file = os.path.join(output_dir, "my_diaries_6months.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_diaries)

print("\n==============================================")
print(f"🎉 추출 완료! {output_file} 파일 확인.")
print("==============================================")