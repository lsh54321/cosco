import sys
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from cosco_rag.agents.graph import create_agent

def run_multi_turn():
    app = create_agent()
    thread_id = "booking_001"
    config = {"configurable": {"thread_id": thread_id}}

    print("=== 订舱 Agent 多轮对话（Ollama 本地模型）===")
    print("输入订舱指令或问题，输入 'exit' 退出。\n")

    # 初始状态
    state = {
        "messages": [],
        "booking_info": {},
        "human_approval_needed": False,
        "human_feedback": "",
        "sensitive_check": {}
    }

    while True:
        user_input = input("\n👤 用户: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 再见！")
            break
        if not user_input:
            continue

        # 每次输入作为新消息
        state["messages"].append(HumanMessage(content=user_input))

        try:
            # 流式执行
            events = app.stream(state, config)
            interrupted = False
            for event in events:
                if "agent" in event and "messages" in event["agent"]:
                    for msg in event["agent"]["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            print(f"🟢 Agent: {msg.content}")
                if "__interrupt__" in event:
                    print("\n⏸️ 等待人工审核...")
                    fb = input("请输入审核结果 (approved/rejected 及原因): ")
                    # 恢复执行，并将 feedback 作为 resume 值传入
                    for resume_event in app.stream(Command(resume=fb), config):
                        if resume_event and resume_event.get("human_review") and resume_event["human_review"].get("messages") and resume_event["human_review"]["messages"][0]:
                            print(resume_event["human_review"]["messages"][0].content)
                    break
            if not interrupted:
                # 更新 state 中的 messages（实际已经通过 checkpoint 自动保存）
                # 但为了 local 变量同步，可以从最后状态获取
                pass
        except Exception as e:
            print(f"处理出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_multi_turn()