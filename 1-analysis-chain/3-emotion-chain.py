# 파일 이름: 3_emotion_chain.py

from dotenv import load_dotenv
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser

# 1. API 키 로드
# .env 파일에서 GEMINI_API_KEY를 자동으로 불러옵니다.
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY를 .env 파일에 정확히 입력했는지 확인하세요.")

# 2. Pydantic 스키마 로드
# 2-emotion-schema.py 파일에서 정의된 설계를 불러옵니다.
# NOTE: 파일명을 '2-emotion-schema.py'로 저장했기 때문에 '2_emotion_schema'가 아닌 'emotion_schema'로 임포트해야 합니다.
# 이 파일은 같은 폴더 안에 있다고 가정합니다.
from emotion_schema import EmotionAnalysisReport

# Pydantic 파서 설정
parser = PydanticOutputParser(pydantic_object=EmotionAnalysisReport)

# 3. LLM 모델 초기화
# Gemini-2.5-flash 모델 사용 및 API 키 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.1, # 창의성(온도)은 낮춰서 일관된 분석 결과를 유도합니다.
)

# 4. 프롬프트 템플릿 정의 (분석 요청서)
# LLM에게 어떤 역할을 해야 하고, 어떤 형식으로 답변해야 하는지를 알려줍니다.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "당신은 개인 일기를 분석하여 사용자가 자신의 감정 패턴을 객관적으로 이해할 수 있도록 돕는 심리 분석 전문가입니다. "
                "사용자가 제공한 일기 내용을 분석하여 감정의 유형, 강도, 그리고 그 원인이 된 구체적인 사건을 추출해야 합니다. "
                "결과물은 반드시 다음 형식에 맞춰서 JSON으로 출력해야 합니다.\n"
                "{format_instructions}"
            ),
        ),
        ("human", "다음 일기를 분석하여 상세 보고서를 작성해 주세요:\n\n{diary_chunk}"),
    ]
)

# 5. LangChain 체인 구축 (Prompt -> LLM -> Output Parser)
# Prompt와 LLM, 파서를 연결하여 하나의 실행 체인을 만듭니다.
emotion_chain = prompt | llm | parser

# 6. 분석 실행 함수
def analyze_diary_chunk(diary_chunk):
    """주어진 일기 청크를 EmotionAnalysisReport 형식으로 분석하고 결과를 반환합니다."""
    
    # 프롬프트 변수와 파서의 출력 형식을 체인에 전달합니다.
    result = emotion_chain.invoke(
        {
            "diary_chunk": diary_chunk.page_content,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    return result


# 이 파일은 분석 체인만 정의하며, 실제 실행은 다음 파일에서 documents를 불러와 반복합니다.
print("✅ 감정 분석 체인(Emotion Chain) 정의 완료! 다음 단계로 넘어가세요.")