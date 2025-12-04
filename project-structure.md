# 📂 프로젝트 모듈 구조 및 실행 흐름 (Maum Diary RAG)
## 복잡한 프로젝트를 설계할 때는 **단일 파일(final-analysis.py)**로 먼저 구현해서 기능 검증(PoC)을 한 후, 이를 **모듈(analysis_chain.py, data_preparer.py 등)**로 분리하여 코드를 깔끔하게 관리하는 것이 정석적인 개발 방식

본 프로젝트는 효율적인 코드 관리와 유지보수성 향상을 위해 **핵심 기능별로 모듈(파일)**을 분리하여 구현되었습니다.

## 1. 프로젝트 모듈 구조와 역할

| 파일명 | 역할 (담당 기능) | 코드 포함 내용 (구현 상세) |
| :--- | :--- | :--- |
| **`main.py`** | **프로젝트 실행 관리자 (Entry Point).** 전체 파이프라인의 **흐름(Flow)**을 정의하고, 각 모듈의 함수를 순서대로 호출하여 결과를 통합합니다. | 환경 변수 로드, `main()` 함수 정의, 각 모듈의 함수를 호출하여 분석, 보고서 생성, RAG를 순차적으로 실행하는 메인 로직. |
| **`data_preparer.py`** | **데이터 준비 및 전처리 전담.** 원본 일기 파일 로드 및 RAG 시스템을 위한 텍스트 분할 작업을 담당합니다. | `prepare_data()` 함수: `DirectoryLoader`로 파일 로드, `RecursiveCharacterTextSplitter`로 청크 분할 및 `Document` 객체 리스트 반환. |
| **`analysis_chains.py`** | **분석 및 보고서 생성 로직 전담.** LLM을 사용하는 모든 LangChain 체인을 정의하고 반환합니다. | `get_emotion_analysis_chain()`, `get_final_report_chain()`, `get_rag_chain()` 등 LLM 프롬프트, Pydantic 파서를 포함한 **독립적인 체인 정의**. |
| **`data_analysis.py`** | **(확장 예정)** 추가 데이터 처리 및 시각화 전담. 분석된 JSON 데이터를 기반으로 통계 또는 차트 생성을 담당합니다. | `analyze_json_for_chart()`, `calculate_emotion_frequency()` 등 분석 결과의 후처리 및 시각화 관련 함수. |

---

## 2. `main.py`에서의 모듈 사용 시점 (When & How)
###main.py는 마치 공장의 조립 라인과 같습니다. 각 모듈은 특정 단계에서 호출되어 필요한 작업을 수행하고, 그 결과물을 다음 단계로 전달합니다.

`main.py`는 각 모듈을 필요한 시점에 **임포트(Import)**하고, 해당 모듈의 **함수를 호출**하여 결과를 전달받는 방식으로 프로젝트를 진행합니다.

### A. 초기화 및 환경 변수 로드

* **시점:** 스크립트 실행 직후, 모든 다른 임포트보다 최우선.
* **사용:** `from dotenv import load_dotenv`를 사용하여 `.env` 파일을 로드하고, `os.getenv()`로 API 키를 `GEMINI_API_KEY` 변수에 저장합니다.

### B. 데이터 준비 단계

| 시점 (When) | 동작 (How) |
| :--- | :--- |
| **프로세스 시작** | **`from data_preparer import prepare_data`**를 통해 함수를 가져옵니다. |
| **데이터 처리** | `processed_documents = prepare_data(file_path)`를 호출하여 **분할된 일기 청크**를 받습니다. |

### C. LLM 분석 및 보고서 생성 단계

| 시점 (When) | 동작 (How) |
| :--- | :--- |
| **체인 구성 필요 시** | **`from analysis_chains import get_emotion_analysis_chain, get_final_report_chain`**를 통해 함수를 가져옵니다. |
| **감정 분석 실행** | `analysis_chain = get_emotion_analysis_chain(llm)`를 호출하여 **분석 체인 객체**를 받아 `invoke`로 실행합니다. |
| **보고서 생성** | `report_chain = get_final_report_chain(llm)`를 호출하여 최종 보고서를 생성할 **체인 객체**를 얻습니다. |

### D. RAG 시스템 구축 및 테스트 단계

| 시점 (When) | 동작 (How) |
| :--- | :--- |
| **벡터 데이터 저장 시** | LangChain 클래스 (`GoogleGenerativeAIEmbeddings`, `Chroma`)를 사용하여 **임베딩 및 벡터화**를 수행합니다. |
| **질의응답 시** | **`analysis_chains.py`**에 정의된 `get_rag_chain()` 함수를 호출하여 RAG 로직이 포함된 최종 질의응답 체인을 실행합니다. |

---

#요약: main.py는 마치 공장의 조립 라인과 같습니다. 각 모듈은 특정 단계에서 호출되어 필요한 작업을 수행하고, 그 결과물을 다음 단계로 전달합니다. 
**결론적으로, `main.py`는 모듈화된 각 파일이 가진 독립적인 '기능 블록'을 순서대로 조립하고 실행하는 지휘자 역할을 수행합니다. 