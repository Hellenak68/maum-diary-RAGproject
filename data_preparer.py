# 파일 이름: data_preparer.py (언더바 사용 필수)

import os
from typing import List
from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Pydantic 스키마 정의 ---
class EmotionTag(BaseModel):
    emotion: str = Field(description="구체적인 감정 이름 (긍정, 불안, 분노, 슬픔 등).")
    intensity: float = Field(description="감정의 강도 (0.0에서 1.0 사이).")
    reason: str = Field(description="이 감정을 느끼게 된 구체적인 사건이나 문구.")

class EmotionAnalysisReport(BaseModel):
    summary: str = Field(description="해당 일기 청크의 핵심 내용 요약 (30자 이내).")
    emotion_tags: List[EmotionTag] = Field(description="일기 청크에서 발견된 모든 감정 태그 목록.")
# -------------------------------------------------------------


def prepare_data(file_path: str = "./data_raw/my-diaries-7days.txt"):
    """
    일기 데이터를 로드하고 LangChain Document 객체 목록으로 분할합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 오류: 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {file_path}")

    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["---", "\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_documents(documents)

    processed_documents = []
    for i, chunk in enumerate(chunks):
        chunk.metadata['doc_type'] = 'diary_entry'
        chunk.metadata['entry_id'] = i + 1
        processed_documents.append(chunk)

    return processed_documents