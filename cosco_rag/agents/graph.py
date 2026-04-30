import json
import re
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable

from cosco_rag import config
from cosco_rag.tools import tools
from cosco_rag.knowledge.milvus_client import search_sensitive_goods
from cosco_rag.utils.logger import get_logger

# 工具名称到函数的映射
tool_map = {t.name: t for t in tools}
tool_node = ToolNode(tools)   # 自动处理 tool_calls
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    booking_info: dict
    human_approval_needed: bool
    human_feedback: str
    sensitive_check: dict
tool_executor = ToolNode(tools)
tools_list = tools  # 确保 tools 是你的工具列表
llm_with_tools = config.base_llm.bind_tools(tools_list)  # 关键：创建绑定工具的新实例

import re
from typing import List, Dict, Any

logger = get_logger("agent_node")

@traceable
def parse_tool_calls_from_ai_content(content: str) -> List[Dict[str, Any]]:
    """
    从AI回复中解析出工具调用。
    支持格式：
        query_space(port_of_loading="上海", destination="洛杉矶", container_type="40HQ")
    或
        submit_booking(shipper="宁波XX公司", consignee="ABC公司", goods_name="玩具", hs_code="950300", weight_kg=500, container_type="40HQ", port_of_loading="上海", destination="洛杉矶")

    返回列表，每个元素为 {"name": 工具名, "args": {参数名: 参数值}}
    如果解析出的参数值包含占位符（如"全称"、"名称"、"编码"、"公斤"、"负责"、"待定"、"unknown"等），则视为无效调用，返回空列表。
    """
    # 匹配类似 "工具名(参数=值, 参数2=值2)" 的模式，支持跨行
    # 注意：参数值可以是双引号字符串、单引号字符串、无引号数字或布尔值
    pattern = r'(\w+)\(([^)]+)\)'
    matches = re.findall(pattern, content, re.DOTALL)

    tool_calls = []
    invalid_placeholders = ["全称", "名称", "编码", "公斤", "负责", "待定", "unknown", "请提供", "输入", "?"]

    for tool_name, args_str in matches:
        # 解析参数字符串: key=value, key2=value2
        args = {}
        # 匹配 key=value, 支持值带引号或不带引号
        arg_pattern = r'(\w+)=("([^"]*)"|\'([^\']*)\'|([^,)]+))'
        for match in re.finditer(arg_pattern, args_str):
            key = match.group(1)
            # 值可能是 group 3 (双引号), group4 (单引号), group5 (无引号)
            value = match.group(3) or match.group(4) or match.group(5)
            if value is not None:
                value = value.strip()
                # 尝试转换为数字
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                args[key] = value

        # 检查参数值是否包含占位符
        has_placeholder = False
        for val in args.values():
            if isinstance(val, str):
                for ph in invalid_placeholders:
                    if ph in val:
                        has_placeholder = True
                        break
            if has_placeholder:
                break

        if has_placeholder:
            # 无效调用，直接返回空列表（表示没有有效工具调用）
            return []

        tool_calls.append({"name": tool_name, "args": args})

    return tool_calls


@traceable
def agent_node(state: AgentState):
    tool_descriptions = """
- query_space: 舱位查询。必需参数：port_of_loading（起运港）, destination（目的港）, container_type（箱型，如20GP/40HQ）。
- submit_booking: 提交订舱。必需参数：shipper（发货人）, consignee（收货人）, goods_name（品名）, hs_code（HS编码）, weight_kg（毛重公斤）, container_type（箱型）, port_of_loading（起运港）, destination（目的港）。
- get_so_status: 查询S/O状态。必需参数：so_no（S/O号）。
- get_vgm_deadline: 查询VGM截止时间。必需参数：booking_no（订舱号）。
- submit_bl_draft: 提交提单确认件。必需参数：booking_no（订舱号）, bl_draft_json必须根据 booking_info 中的订舱信息自动生成 JSON 格式的提单草稿作为参数。
- track_container: 集装箱追踪。必需参数：container_no（箱号）。
"""
    sys_msg = f"""你是一个中远海运订舱助手。你有以下工具可用：
{tool_descriptions}

**核心指令**：
1. 当用户提供了某个工具所需的所有必需参数时，你必须**立即输出该工具调用**，格式为：工具名(参数1="值1", 参数2="值2", ...)
   - 数字参数可以不加引号，例如 weight_kg=500。
   - 不要输出任何其他解释或自然语言。
2. 如果用户提供的参数不完整，用自然语言追问缺少的参数，不要输出工具调用。
3. 一次只能调用一个工具。先判断用户意图最匹配哪个工具。

**示例**：
- 用户：查一下上海到洛杉矶的40HQ舱位
  助手：query_space(port_of_loading="上海", destination="洛杉矶", container_type="40HQ")
- 用户：帮我订一个40HQ柜子，从上海到洛杉矶，品名玩具，HS编码950300，毛重500公斤，发货人宁波XX公司，收货人ABC公司
  助手：submit_booking(shipper="宁波XX公司", consignee="ABC公司", goods_name="玩具", hs_code="950300", weight_kg=500, container_type="40HQ", port_of_loading="上海", destination="洛杉矶")
- 用户：查一下SO12345678的状态
  助手：get_so_status(so_no="SO12345678")
- 用户：订舱号COSU12345678的VGM截止时间
  助手：get_vgm_deadline(booking_no="COSU12345678")
    现在，用户请求：{state['messages'][-1].content}
    请直接输出上述格式，不要有其他内容。
"""
    # 构建消息列表
    messages = [SystemMessage(content=sys_msg)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    thread_id = state.get("config", {}).get("configurable", {}).get("thread_id", "unknown")
    logger.bind(thread_id=thread_id).info(f"Agent node input messages count: {len(state['messages'])}")
    return {"messages": [AIMessage(content=response.content,tool_calls=response.tool_calls)]}


# agents/graph.py 或 agents/tool_node.py
@traceable
def tool_node(state: AgentState):
    """执行工具并更新状态"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    new_booking_info = state.get("booking_info", {}).copy()
    results = []

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        # ========== 关键：针对 submit_bl_draft 补全参数 ==========
        if tool_name == "submit_bl_draft":
            # 检查是否缺少 bl_draft_json
            if "bl_draft_json" not in tool_args or not tool_args["bl_draft_json"]:
                # 从 state["booking_info"] 中构造提单草稿
                bl_draft_data = {
                    "shipper": new_booking_info.get("shipper", ""),
                    "consignee": new_booking_info.get("consignee", ""),
                    "goods_name": new_booking_info.get("goods_name", ""),
                    "hs_code": new_booking_info.get("hs_code", ""),
                    "weight_kg": new_booking_info.get("weight_kg", ""),
                    "container_type": new_booking_info.get("container_type", ""),
                    "port_of_loading": new_booking_info.get("port_of_loading", ""),
                    "destination": new_booking_info.get("destination", ""),
                    "booking_no": tool_args.get("booking_no", new_booking_info.get("booking_no"))
                }
                tool_args["bl_draft_json"] = json.dumps(bl_draft_data)

        # 执行工具
        tool_func = tool_map[tool_name]
        raw_result = tool_func.invoke(tool_args)
        result_data = json.loads(raw_result)
        results.append(ToolMessage(content=raw_result, tool_call_id=tc["id"]))

        # 更新 booking_info（订舱成功时）
        if tool_name == "submit_booking" and result_data.get("success"):
            new_booking_info["booking_no"] = result_data["booking_no"]
            new_booking_info["so_no"] = result_data["so_no"]
            new_booking_info.update(tool_args)

    return {"messages": results, "booking_info": new_booking_info}

@traceable
def sensitive_check_node(state: AgentState):
    """RAG 敏感品名检测（非工具调用，直接检索）"""
    # 如果从消息中提取了 goods_name，可以存入 booking_info
    goods = state.get("booking_info", {}).get("goods_name", "")
    if not goods and state["messages"] and state["messages"][1] and state["messages"][1].tool_calls and state["messages"][1].tool_calls[0]["args"].get("goods_name"):
        goods = state["messages"][1].tool_calls[0]["args"]["goods_name"]  # 示例
    if not goods:
        return {"sensitive_check": {"is_sensitive": False}}
    res = search_sensitive_goods(goods)
    if res.get("is_sensitive"):
        warn = AIMessage(content=f"⚠️ {goods} 属于【{res['risk_level']}】风险品名，需提供 {res['required_docs']}")
        return {"messages": [warn], "sensitive_check": res}
    return {"sensitive_check": res}

@traceable
def human_review_node(state: AgentState):
    """人工审核节点（interrupt）"""
    data = {
        "question": "请审核以下提单草稿，输入 'approved' 或 'rejected' 及原因。",
        "bl_draft": state["booking_info"].get("bl_draft", {}),
        "last_ai": state["messages"][-2].content if len(state["messages"]) >= 2 else ""
    }
    fb = interrupt(data)
    state["human_feedback"] = fb
    if fb.lower().startswith("approved"):
        return {"messages": [AIMessage(content="✅ 人工审核通过，订舱流程继续。")]}
    else:
        return {"messages": [AIMessage(content=f"❌ 驳回，原因：{fb}。流程终止。")]}

@traceable
def route_after_agent(state: AgentState) -> Literal["tools", "human_review", "end"]:
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_msg = messages[-1]

    # 检查是否是 AIMessage 并且有 tool_calls
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        # 如果调用了 submit_bl_draft，进入人工审核
        for tc in last_msg.tool_calls:
            if tc.get("name") == "submit_bl_draft":
                return "human_review"
        return "tools"

    return "end"

@traceable
def create_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_with_state_update)
    workflow.add_node("sensitive_check", sensitive_check_node)
    workflow.add_node("human_review", human_review_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "sensitive_check")
    workflow.add_conditional_edges("agent", route_after_agent, {
        "tools": "tools",
        "human_review": "human_review",
        "end": END
    })
    workflow.add_edge("tools", "agent")       # 工具执行后回到 agent
    workflow.add_edge("human_review", END)
    return workflow.compile(checkpointer=MemorySaver())


import json

@traceable
def tool_node_with_state_update(state: AgentState):
    """执行工具并直接更新 booking_info，无需单独的 process_results 节点"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls  # 此时 last_msg 一定是 AIMessage

    new_booking_info = state.get("booking_info", {}).copy()
    results = []

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_func = tool_map[tool_name]

        raw_result = tool_func.invoke(tool_args)
        result_data = json.loads(raw_result)
        results.append(ToolMessage(content=raw_result, tool_call_id=tc["id"]))

        # 直接更新状态
        if tool_name == "submit_booking" and result_data.get("success"):
            new_booking_info["booking_no"] = result_data["booking_no"]
            new_booking_info["so_no"] = result_data["so_no"]
            new_booking_info.update(tool_args)  # 保存原始参数

    return {"messages": results, "booking_info": new_booking_info}