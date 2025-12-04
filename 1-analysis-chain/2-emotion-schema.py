# 파일 이름: 2_emotion_schema.py

from pydantic import BaseModel, Field
from typing import List

# 1. 감정 태그의 정의
# LLM이 출력해야 하는 감정 태그의 목록을 정의합니다. (주관적 만족도의 핵심)
class EmotionTag(BaseModel):
    """일기 한 문단 또는 청크에서 추출된 하나의 감정 태그."""
    
    # Field를 사용해 LLM에게 이 필드에 대한 자세한 설명을 제공합니다.
    emotion: str = Field(
        description="긍정, 부정, 불안, 기쁨, 분노, 슬픔, 평온 등 구체적인 단일 감정 이름."
    )
    intensity: float = Field(
        description="해당 감정의 강도. 0.0 (매우 약함)에서 1.0 (매우 강함) 사이의 소수점 값."
    )
    reason: str = Field(
        description="일기 내용 중 이 감정을 느끼게 된 구체적인 사건이나 문구."
    )

# 2. 분석 보고서의 최종 출력 구조 정의
# LLM이 최종적으로 반환해야 하는 JSON 객체의 전체 구조입니다.
class EmotionAnalysisReport(BaseModel):
    """하나의 일기 청크에 대한 감정 분석 심화 보고서."""
    
    summary: str = Field(
        description="해당 일기 청크의 핵심 내용과 감정적 맥락을 요약한 30자 이내의 문장."
    )
    # 감정 태그는 여러 개일 수 있으므로 List[EmotionTag]로 정의합니다.
    emotion_tags: List[EmotionTag] = Field(
        description="일기 청크에서 발견된 모든 감정 태그 목록."
    )

# 이 두 클래스가 당신의 감정 분석 결과물(JSON)의 최종 설계 도면이 됩니다.