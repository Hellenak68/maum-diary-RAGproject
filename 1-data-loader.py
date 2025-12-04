# 파일 이름: 1_data_loader.py (최종 버전)

import os # <-- os 모듈을 명확히 불러옵니다 (이전 NameError 해결).
from langchain_community.document_loaders import TextLoader # <-- TextLoader를 community에서 불러옵니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter # <-- 최신 text_splitters에서 불러옵니다.

# --- 1. 데이터 로딩 ---
# NOTE: 파일 경로는 항상 './data_raw/my-diaries-7days.txt'여야 합니다.
file_path = "./data_raw/my-diaries-7days.txt" 

# --- 2. 오류 방지: 파일 및 폴더 존재 확인 ---
##if not os.path.exists("./data_raw"):
##    print("❌ 오류: 'data_raw' 폴더를 먼저 만드세요.")
##    exit()

##if not os.path.exists(file_path):
##    print("❌ 오류: 'my-diaries-7days.txt' 파일이 data_raw 폴더에 없습니다. 먼저 7개 일기를 채워주세요.")
##    exit()

# TextLoader를 사용해 파일을 불러옵니다.
# TextLoader는 상단에 import 되어 있어야 합니다.
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# --- 3. 텍스트 분할 (Splitter) ---
# 일기 텍스트를 '---' 구분자를 기준으로 나누도록 설정합니다.
text_splitter = RecursiveCharacterTextSplitter(
    separators=["---", "\n\n", "\n", " "],
    chunk_size=1000,                       
    chunk_overlap=0,                      
    length_function=len
)

# documents 리스트에 있는 텍스트를 분할합니다.
chunks = text_splitter.split_documents(documents)

# --- 4. 메타데이터 추가 및 결과 확인 ---
processed_documents = []

for i, chunk in enumerate(chunks):
    chunk.metadata['doc_type'] = 'diary_entry'
    chunk.metadata['entry_id'] = i + 1
    processed_documents.append(chunk)

# 결과 보고
print("\n==============================================")
print(f"🎉 데이터 로딩 및 분할 완료!")
print(f"총 {len(processed_documents)}개의 Document 객체 생성 (일기 개수 확인): {len(processed_documents)}개")
print("==============================================")
print("✅ 첫 번째 일기 미리보기 (메타데이터 포함):")
print(processed_documents[0])