import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# 1. API 키 설정 (무료 계층 사용)


# 2. State 정의
class GraphState(TypedDict):
    input: str
    retrieved: Optional[str]
    response: Optional[str]

# 3. 모델 및 DB 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
db = SQLDatabase.from_uri("mysql+pymysql://root:Abcd1234@localhost:3306/digital_twin")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 4. 노드 함수들
def retrieve_from_db(state: GraphState) -> GraphState:
    print("--- [Log] DB에서 정보 찾는 중... ---")
    query = state["input"]
    prompt = f"질문: {query}\n위 질문에 대해 SQL을 생성해 조회하고 한국어로 설명해줘. 결과가 없으면 'null'을 포함해."
    try:
        retrieved = db_chain.invoke(prompt)
    except Exception as e:
        print(f"Error: {e}")
        retrieved = "null"
    return {"input": state["input"], "retrieved": retrieved}

def run_llm(state: GraphState) -> GraphState:
    print("--- [Log] AI 지식으로 답변 생성 중... ---")
    response = llm.invoke(state["input"])
    return {"response": response.content}

def return_retrieved(state: GraphState) -> GraphState:
    print("--- [Log] DB 결과 반환 중... ---")
    return {"response": state["retrieved"]}

def should_use_retrieved(state: GraphState) -> str:
    if state["retrieved"] and "null" not in state["retrieved"].lower():
        return "use_retrieved"
    return "use_llm"

# 5. 그래프 빌드
builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_from_db)
builder.add_node("use_llm", run_llm)
builder.add_node("use_retrieved", return_retrieved)

builder.set_entry_point("retrieve")
builder.add_conditional_edges("retrieve", should_use_retrieved, {
    "use_llm": "use_llm",
    "use_retrieved": "use_retrieved"
})
builder.set_finish_point("use_llm")
builder.set_finish_point("use_retrieved")

graph = builder.compile()

# 6. 실행 (Main)
if __name__ == "__main__":
    print("=== 스마트 팩토리 AI 에이전트 가동 ===")
    user_input = input("질문을 입력하세요: ")
    result = graph.invoke({"input": user_input})
    print("\n[최종 답변]:", result["response"])