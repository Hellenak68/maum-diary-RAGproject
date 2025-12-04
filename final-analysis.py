# 파일 이름: final-analysis.py (모든 단계 통합 - 최종 버전)

# 1. 환경 및 라이브러리 로드
from dotenv import load_dotenv
import os
import json
import time
from typing import List
from pydantic import BaseModel, Field

# LangChain 및 Google GenAI 라이브러리
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser # 수정: 최신 모듈 경로 사용
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# --- Pydantic 스키마 정의 ---
class EmotionTag(BaseModel):
    """일기 한 문단 또는 청크에서 추출된 하나의 감정 태그."""
    emotion: str = Field(description="구체적인 감정 이름 (긍정, 불안, 분노, 슬픔 등).")
    intensity: float = Field(description="감정의 강도 (0.0에서 1.0 사이).")
    reason: str = Field(description="이 감정을 느끼게 된 구체적인 사건이나 문구.")

class EmotionAnalysisReport(BaseModel):
    """하나의 일기 청크에 대한 감정 분석 심화 보고서."""
    summary: str = Field(description="해당 일기 청크의 핵심 내용 요약 (30자 이내).")
    emotion_tags: List[EmotionTag] = Field(description="일기 청크에서 발견된 모든 감정 태그 목록.")
# -----------------------------

# 2. 데이터 로딩 및 분할
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY를 .env 파일에 정확히 입력했는지 확인하세요.")

file_path = "./data_raw/my-diaries-7days.txt"

# 파일 존재 확인 (이 코드는 주석 처리하지 않고 그대로 둡니다.)
if not os.path.exists(file_path):
    print(f"❌ 오류: 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {file_path}")
    exit()

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

print(f"🎉 데이터 로딩 및 분할 완료! 총 {len(processed_documents)}개의 Document 객체 생성.")


# 3. LLM 분석 체인 설정
parser = PydanticOutputParser(pydantic_object=EmotionAnalysisReport)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.1,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "당신은 심리 분석 전문가입니다. 일기를 분석하여 감정 유형, 강도(0.0~1.0), 그리고 원인 사건을 추출하세요. "
                "결과는 반드시 다음 형식에 맞춰서 JSON으로 출력해야 합니다.\n"
                "{format_instructions}"
            ),
        ),
        ("human", "다음 일기를 분석하여 상세 보고서를 작성해 주세요:\n\n{diary_chunk}"),
    ]
)
emotion_chain = prompt | llm | parser


# 4. 일괄 분석 및 저장
print("\n==============================================")
print("🧠 일괄 감정 분석 시작: 모든 일기 청크 분석 중...")
print("==============================================")

all_analysis_reports = []

for i, chunk in enumerate(processed_documents):
    entry_id = chunk.metadata.get('entry_id', 'Unknown')
    print(f"--- [분석 중] 청크 번호: {i+1}/{len(processed_documents)} ---")
    
    try:
        analysis_result = emotion_chain.invoke(
            {
                "diary_chunk": chunk.page_content,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        report_data = analysis_result.model_dump()
        report_data['metadata'] = chunk.metadata
        all_analysis_reports.append(report_data)
        print(f"✅ 분석 완료! (ID: {entry_id})")
        
    except Exception as e:
        print(f"❌ 분석 오류 발생 (ID: {entry_id}): {e}")
    
    time.sleep(1) 

print("\n🎉 모든 일기 분석 완료!")
output_file_path = "./emotion-reports.json" # 파일명 하이픈 적용
try:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_analysis_reports, f, ensure_ascii=False, indent=4)
    print(f"✅ 최종 보고서 저장 성공: {output_file_path}")
except Exception as e:
    print(f"❌ 파일 저장 오류 발생: {e}")


# 5. 종합 심리 보고서 생성 및 저장
reports_string = json.dumps(all_analysis_reports, ensure_ascii=False, indent=2)

report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "당신은 심리 분석가이며, 제공된 일기 분석 데이터를 기반으로 종합적인 심리 보고서를 작성합니다. "
                "보고서는 한국어로 작성하며, 주요 감정 패턴과 그 요인에 대해 전문적이고 공감적인 어조로 서술하세요. "
                "목차와 소제목(##)을 사용하여 구조를 명확하게 만드세요."
            ),
        ),
        ("human", "다음은 제 일기 분석 결과(JSON)입니다. 이를 통합하여 종합 심리 보고서를 작성해 주세요:\n\n{analysis_data}"),
    ]
)
report_chain = report_prompt | llm

print("\n==============================================")
print("📝 종합 심리 보고서 생성 시작...")
print("==============================================")

try:
    final_report = report_chain.invoke(
        {
            "analysis_data": reports_string,
        }
    )

    print("✅ 종합 보고서 생성 성공!")
    print("\n--- 최종 심리 보고서 ---")
    report_content = final_report.content
    print(report_content)
    print("-------------------------\n")
    
    report_output_file = "final-psychological-report.md" # 파일명 하이픈 적용
    with open(report_output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 보고서 파일 저장 완료: {report_output_file}")
    
except Exception as e:
    print(f"❌ 보고서 생성 중 오류 발생: {e}")


# 6. RAG 시스템 구축 및 테스트
print("\n==============================================")
print("🔍 RAG 시스템 구축 및 테스트 시작...")
print("==============================================")

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=GEMINI_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=processed_documents, 
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ("당신은 사용자의 일기 데이터베이스 기반 전문 검색 시스템입니다. "
                        "주어진 '맥락 정보'만을 사용하여 질문에 답변하세요. 답이 없다면, '정보를 찾을 수 없습니다'라고 답변해야 합니다. "
                        "\n\n--- 맥락 정보 ---\n{context}")),
            ("human", "질문: {question}"),
        ]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
    )

    test_question = "내가 일주일 동안 가장 기뻤던 사건은 무엇이며, 그 날짜는 언제야?"

    print(f"**테스트 질문:** {test_question}")

    rag_response = rag_chain.invoke({"question": test_question})
        
    print("\n--- RAG 답변 ---")
    print(rag_response.content)
    print("------------------\n")
        
    vectorstore.delete_collection()
    print("✅ RAG 시스템 테스트 완료. 메모리 정리.")

except Exception as e:
    print(f"❌ RAG 실행 중 오류 발생: {e}")
    print("Embedding 모델(embedding-001) 권한 또는 라이브러리 설치를 확인하세요.")