# 파일 이름: main.py (언더바 파일들을 임포트)

import json
import time
from data_preparer import prepare_data # 언더바 파일에서 임포트
from analysis_chains import get_emotion_analysis_chain, get_final_report_chain # 언더바 파일에서 임포트
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY를 .env 파일에 정확히 입력했는지 확인하세요.")

# 🌟🌟🌟 추가할 코드 🌟🌟🌟
# Google SDK가 기본적으로 찾는 환경 변수 이름에도 키를 설정합니다.
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY 
# 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟

def format_docs(docs):
    """RAG 검색 결과를 하나의 문자열로 합치는 헬퍼 함수"""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    """모든 단계를 실행하고 결과를 출력/저장합니다."""
    
    # 1. 데이터 준비 
    print("1. 데이터 준비 중...")
    try:
        processed_documents = prepare_data()
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        return
        
    print(f"✅ 데이터 로딩 완료. 총 {len(processed_documents)}개 청크.")
    
    # 2. 분석 체인 로드 
    emotion_chain = get_emotion_analysis_chain()
    final_report_chain = get_final_report_chain()
    
    # 3. 일괄 분석
    print("\n2. 일괄 감정 분석 시작...")
    all_analysis_reports = []
    
    for i, chunk in enumerate(processed_documents):
        try:
            # Pydantic Output Parser의 format_instructions를 가져오는 방식 변경
            analysis_result = emotion_chain.invoke(
                {
                    "diary_chunk": chunk.page_content,
                    "format_instructions": emotion_chain.steps[-1].get_format_instructions(), # 파서에서 명령어 가져오기
                }
            )
            report_data = analysis_result.model_dump()
            report_data['metadata'] = chunk.metadata
            all_analysis_reports.append(report_data)
            print(f"  [+] 청크 {i+1} 분석 완료.")
        except Exception as e:
            print(f"  [-] 청크 {i+1} 분석 오류: {e}")
        time.sleep(1) 

    # 4. JSON 저장
    output_file_path = "./emotion-reports.json" # 출력 파일은 하이픈 사용
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_analysis_reports, f, ensure_ascii=False, indent=4)
    print(f"✅ 분석 결과 JSON 저장 완료: {output_file_path}")

    
    # 5. 종합 보고서 생성 및 저장
    print("\n3. 종합 심리 보고서 생성 중...")
    reports_string = json.dumps(all_analysis_reports, ensure_ascii=False, indent=2)

    final_report = final_report_chain.invoke({"analysis_data": reports_string})
    report_content = final_report.content
    
    report_output_file = "final-psychological-report.md" # 출력 파일은 하이픈 사용
    with open(report_output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✅ 종합 보고서 저장 완료: {report_output_file}")
    
    # 6. RAG 시스템 구축 및 테스트
    print("\n4. RAG 시스템 구축 및 테스트 시작...")
    
    # 이 부분은 main.py 파일 맨 위에서 import 되었어야 합니다.
    # from dotenv import load_dotenv, os
    # load_dotenv()
    # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # 🌟 수정된 코드: embeddings 객체 생성 시 api_key 변수를 명시적으로 전달합니다.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        api_key=GEMINI_API_KEY # <-- 키를 명시적으로 전달
    )
    
    vectorstore = Chroma.from_documents(documents=processed_documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # RAG 체인 구축
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | final_report_chain 
    )
    
    test_question = "내가 일주일 동안 가장 기뻤던 사건은 무엇이며, 그 날짜는 언제야?"
    rag_response = rag_chain.invoke({"question": test_question})
    
    print(f"\n--- RAG 답변 (질문: {test_question}) ---")
    print(rag_response.content)
    print("------------------------------------------")
    
    vectorstore.delete_collection()
    print("✅ RAG 시스템 테스트 완료. 모듈화된 프로젝트 완성!")


if __name__ == "__main__":
    main()