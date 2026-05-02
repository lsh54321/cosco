import json
import os
from typing import Dict, Any, Optional

from fastapi import FastAPI, Query
from langsmith.middleware import TracingMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
# 导入之前定义的 Agent 创建函数（假设在 agents/graph.py 中）
from cosco_rag.agents.graph import create_agent

app = FastAPI(title="中远海运订舱 Agent API")
app.add_middleware(TracingMiddleware)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

from cosco_rag import config
# 引入 LangChain 和 LangSmith 相关库
from langsmith import Client

# --- 1. 配置 LangSmith 追踪 ---
# 从配置管理器中获取 LangSmith 配置
langsmith_config = {
    "LANGCHAIN_TRACING_V2": str(True).lower(),
    "LANGCHAIN_API_KEY": config.SMITH_KEY,
    "LANGCHAIN_PROJECT": config.SMITH_PROJECT,
    "LANGCHAIN_ENDPOINT": config.SMITH,
}

# 将配置写入环境变量，使 LangChain/LangSmith 能够自动拾取
for key, value in langsmith_config.items():
    if value:
        os.environ[key] = value

# 可选：初始化 LangSmith 客户端
if True:
    client = Client(tracing_sampling_rate=0.1)
    print(f"LangSmith 追踪已启用，项目名: {config.SMITH_PROJECT}")
# 存储每个会话的状态（实际生产可用 Redis）
sessions: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    """用户请求体"""
    thread_id: str          # 会话 ID，用于多轮对话和中断恢复
    message: str            # 用户消息
    resume_value: Optional[str] = None  # 人工审核恢复时使用

# ---------------------- SSE 流式处理核心 ----------------------
async def event_generator(thread_id: str, user_message: str, resume_value: str = None):
    """
    SSE 事件生成器，流式输出 Agent 的执行过程。
    如果 Agent 遇到 interrupt，则会发送一个 special 事件，等待外部恢复。
    """
    # 初始化或获取状态
    if thread_id not in sessions:
        sessions[thread_id] = {
            "agent": create_agent(),
            "config": {"configurable": {"thread_id": thread_id},
                       "tags": ["production", "user-123"],
                        "metadata": {"user_id": "user-123", "customer_tier": "vip"}},
            "current_state": {
                "messages": [],
                "booking_info": {},
                "human_approval_needed": False,
                "human_feedback": "",
                "sensitive_check": {},
                # 新增字段
                "intent": "",
                "next_agent": "",
                "active_agent": ""
            },
            "interrupted": False,
            "interrupt_data": None
        }
    session = sessions[thread_id]
    app_graph = session["agent"]
    session_config = session["config"]
    state = session["current_state"]

    # 如果是恢复中断，使用 Command(resume=...) 恢复
    if resume_value:
        from langgraph.types import Command
        events = app_graph.stream(Command(resume=resume_value), session_config)
    else:
        # 添加用户消息到状态

        state["messages"].append(HumanMessage(content=user_message))
        events = app_graph.stream(state, session_config)

    try:
        for event in events:
            # 检查是否有中断
            if "__interrupt__" in event:
                # 保存中断信息，等待外部恢复
                interrupt_data = event["__interrupt__"]
                session["interrupted"] = True
                session["interrupt_data"] = interrupt_data
                # 发送一个特殊事件，告知前端需要人工审核
                yield {
                    "event": "interrupt",
                    "data": json.dumps({
                        "message": "需要人工审核，请调用 /chat/resume 接口传入审核结果",
                        "detail": interrupt_data
                    }, ensure_ascii=False)
                }
                break  # 停止流，等待恢复

            # 正常事件流，根据节点提取消息内容并推送
            if "agent" in event and "messages" in event["agent"]:
                for msg in event["agent"]["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        yield {
                            "event": "agent",
                            "data": json.dumps({"content": msg.content}, ensure_ascii=False)
                        }
            elif "tools" in event and "messages" in event["tools"]:
                for msg in event["tools"]["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        yield {
                            "event": "tool",
                            "data": json.dumps({"result": msg.content}, ensure_ascii=False)
                        }
            elif "human_review" in event:
                # 如果人工审核节点输出了消息（理论上已经进入中断，但以防万一）
                pass
            for node_name, node_output in event.items():
                if node_name in ["query_agent", "booking_agent", "compliance_agent", "notify_agent",
                                 "document_agent"] and "messages" in node_output:
                    for msg in event[node_name]["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            print(f"🟢 Agent: {msg.content}")
                            yield {
                                "event": "agent",
                                "data": json.dumps({"content": msg.content}, ensure_ascii=False)
                            }

        # 流正常结束
        yield {"event": "end", "data": json.dumps({"message": "订舱流程结束"})}
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"error": str(e)})}

# ---------------------- API 端点 ----------------------
@app.get("/chat/stream")
async def chat_stream(
    thread_id: str = Query(...),
    message: str = Query(""),
    resume_value: str = Query("")
):
    return EventSourceResponse(
        event_generator(thread_id, message, resume_value),
        headers={"Content-Type": "text/event-stream"}
    )

@app.post("/chat/resume")
async def chat_resume(thread_id: str, feedback: str):
    """
    恢复被中断的会话，传入人工审核结果（例如 "approved" 或 "rejected 原因"）。
    """
    if thread_id not in sessions:
        return {"error": "会话不存在"}
    session = sessions[thread_id]
    if not session.get("interrupted"):
        return {"error": "当前会话未中断"}
    # 重新生成 SSE 流，使用 resume_value
    return EventSourceResponse(
        event_generator(thread_id, user_message="", resume_value=feedback),
        headers={"Content-Type": "text/event-stream"}
    )

@app.get("/session/{thread_id}")
async def get_session(thread_id: str):
    """查询会话状态（用于调试）"""
    if thread_id in sessions:
        return {"status": "active", "interrupted": sessions[thread_id].get("interrupted")}
    return {"status": "not_found"}

# 可选：后台清理过期会话（略）