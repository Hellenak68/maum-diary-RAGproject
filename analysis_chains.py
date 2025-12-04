# 파일 이름: analysis_chains.py (언더바 사용 필수)

from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from data_preparer import EmotionAnalysisReport # 언더바 파일명으로 임포트

# 환경 변수 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY를 .env 파일에 정확히 입력했는지 확인하세요.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.1,
)

# --- 1. 감정 분석 체인 정의 ---
def get_emotion_analysis_chain():
    """개별 일기 청크를 분석하는 LangChain 체인을 반환합니다."""
    
    parser = PydanticOutputParser(pydantic_object=EmotionAnalysisReport)

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
    return prompt | llm | parser


# --- 2. 종합 보고서 생성 체인 정의 ---
def get_final_report_chain():
    """청크 분석 결과를 통합하여 종합 보고서를 생성하는 LangChain 체인을 반환합니다."""
    
    report_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "당신은 개인 심리 분석가이며, 제공된 일기 분석 데이터를 기반으로 종합적인 심리 보고서를 작성합니다. "
                    "보고서는 한국어로 작성하며, 주요 감정 패턴과 그 요인에 대해 전문적이고 공감적인 어조로 서술하세요. "
                    "목차와 소제목(##)을 사용하여 구조를 명확하게 만드세요."
                ),
            ),
            ("human", "다음은 제 일기 분석 결과(JSON)입니다. 이를 통합하여 종합 심리 보고서를 작성해 주세요:\n\n{analysis_data}"),
        ]
    )
    return report_prompt | llm