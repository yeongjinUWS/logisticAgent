import re
import os
import joblib
from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

print(f"[*] Pathlib 기준 경로: {MODEL_DIR}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

class GraphState(TypedDict):
    input: str
    selected_models: Optional[list[str]] 
    
    weather: Optional[dict]
    temperature: Optional[float]
    rain_flag: Optional[int]
    wind_speed: Optional[float]
    humidity: Optional[int]
    
    requset: Optional[str]
    response: Optional[str]
    analyze_prompt: Optional[str]

# [수정] 가용 모델의 메타데이터를 스캔하는 함수 추가
def get_registered_models():
    registry = []
    if not os.path.exists(MODEL_DIR):
        return registry
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".meta.json"):
            try:
                with open(os.path.join(MODEL_DIR, file), "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    registry.append({
                        "id": meta.get("modelFile"), # 예: model_Whse_A
                        "description": meta.get("title", "정보 없음"),
                        "type": meta.get("title", "general") 
                    })
            except Exception: continue
    return registry

# LLM 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- [수정] 오케스트레이터 에이전트: 가용 모델을 보고 동적으로 선택 ---
def analyze_prompt(state: GraphState) -> GraphState:
    print("= Prompt 분석 및 모델 매칭 중 =")
    
    # 현재 폴더에 있는 모든 모델 정보를 가져와 프롬프트에 주입
    registry = get_registered_models()
    model_desc = "\n".join([f"- {m['id']}: {m['description']} (타입: {m['type']})" for m in registry])
    
    prompt = f"""
    당신은 통합 Multi AI 에이전트입니다. 질문을 분석하여 어떤 모델을 사용할지 결정하세요.
    이후 해당 모델에게 어떤 기능을 수행할지 prompt를 작성하세요.

    [사용 가능한 모델 목록]
    {model_desc}
    
    [응답 규칙]
    1. 모델 분석이 필요한 경우: [ACTION: USE_MODEL, TARGET: 모델ID1, 모델ID2, ..., PROMPT: 수요 예측, 월간 사용량]
    2. 일반 질문인 경우: [ACTION: GENERAL_CHAT]
    
    질문: {state['input']}
    """
    
    res = llm.invoke(prompt)
    content = res.content
    print(f" = 분석 결과 : {content} = ")

    if "USE_MODEL" in content:
        # 정규표현식으로 TARGET 뒤의 모델 ID들을 추출
        target_match = re.search(r'TARGET\s*:\s*([^\]\n]+)', content)
        targets = [t.strip() for t in target_match.group(1).split(',')] if target_match else []
        prompt_match = re.search(r'PROMPT\s*:\s*([^\]\n]+)', content)
        prompt = prompt_match.group(1) if prompt_match else ""

        return {
            "analyze_prompt": "모델분석",
            "selected_models": targets,
            "requset": prompt
        }
    
    return {"analyze_prompt": "일반"}

# [유지] 날씨 정보 처리
def analyze_weather(state: GraphState) -> GraphState:
    weather = state.get("weather", {})
    return {
        "temperature": float(weather.get("T1H", 0)),
        "rain_flag": 1 if (float(weather.get("RN1", 0)) > 0 or int(weather.get("PTY", 0)) > 0) else 0,
        "wind_speed": float(weather.get("WSD", 0)),
        "humidity": int(weather.get("REH", 0))
    }

# --- [수정] 범용 모델 실행 에이전트: 선택된 모델들을 동적으로 로드하여 실행 ---
def run_dynamic_models(state: GraphState) -> GraphState:
    print("=== Dynamic Model Execution Agent ===")
    selected_files = state.get("selected_models", [])
    results = []

    for model_id in selected_files:
        model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
        if not os.path.exists(model_path):
            continue

        try:
            # 실시간 로드 (메모리 관리를 위해 필요할 때 로드)
            model = joblib.load(model_path)
            
            if "Whse" in model_id:
                # 기존 물류 예측 로직 실행 (생략된 로직은 이전과 동일하게 구성 가능)
                # sample_X 생성 및 predict...
                pred = 100 # 예시 예측값
                results.append({"model": model_id, "prediction": pred, "type": "물류"})
            else:
                # 새로운 타입의 모델(예: 건설 비용)이 추가되면 여기에 로직 추가
                # results.append({"model": model_id, "prediction": "새로운 로직 결과"})
                pass
        except Exception as e:
            print(f"Error executing {model_id}: {e}")

    return {"response": results}

# [유지] 일반 LLM 대화
def run_general_llm(state: GraphState) -> GraphState:
    prompt = f"당신은 veneta AI 에이전트입니다. 질문에 답하세요: {state['input']}"
    res = llm.invoke(prompt)
    return {"response": res.content}

# [수정] 최종 요약: 모델 예측치와 날씨 정보를 결합하여 리포트
def run_finish_llm(state: GraphState) -> GraphState:
    weather_info = f"기온 {state.get('temperature')}도, 강수 {state.get('rain_flag')}"
    data_summary = state.get('response')

    prompt = f"""
    당신은 데이터 요약 전문가입니다. 아래 분석 결과를 바탕으로 날씨와 연계하여 짧게 요약하세요.
    날씨: {weather_info}
    분석결과: {data_summary}
    """
    res = llm.invoke(prompt)
    return {"response": res.content.strip()}

# --- 그래프 구성 ---
def route_input(state: GraphState) -> str:
    return "use_model" if state["analyze_prompt"] == "모델분석" else "use_llm"

builder = StateGraph(GraphState)

builder.add_node("analyze_prompt", analyze_prompt)
builder.add_node("weather_agent", analyze_weather)
builder.add_node("model_agent", run_dynamic_models) # [수정] 통합된 동적 모델 노드
builder.add_node("llm_agent", run_general_llm)
builder.add_node("run_finish_llm", run_finish_llm)

builder.add_edge("weather_agent", "model_agent")
builder.add_edge("model_agent", "run_finish_llm")
builder.set_entry_point("analyze_prompt")

builder.add_conditional_edges("analyze_prompt", route_input, {
    "use_model": "weather_agent",
    "use_llm": "llm_agent"
})

builder.set_finish_point("run_finish_llm")
builder.set_finish_point("llm_agent")

graph = builder.compile()

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Logistics AI Agent")

# === Request / Response 모델 ===
class ChatRequest(BaseModel):
    input: str
    weather: dict 

class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Spring에서 호출하는 엔드포인트
    """

    result = graph.invoke({
        "input": req.input,
        "weather": req.weather
    })
    return ChatResponse(response=result["response"])


# === 로컬 테스트용 (선택) ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
