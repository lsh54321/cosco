import json
import re
from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable

from cosco_rag import config
from cosco_rag.tools import tools
from cosco_rag.knowledge.milvus_client import search_sensitive_goods
from cosco_rag.utils.logger import get_logger

logger = get_logger("multi_agent_graph")

# ==================== 工具映射 ====================
tool_map = {t.name: t for t in tools}


# ==================== 状态定义（增加多 Agent 字段） ====================
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    booking_info: dict
    human_approval_needed: bool
    human_feedback: str
    sensitive_check: dict
    # 多 Agent 新增字段
    intent: str  # 识别出的意图
    next_agent: str  # Supervisor 选择的下一个 Agent
    active_agent: str  # 当前正在执行的 Agent 名称


# ==================== 原有辅助函数（解析工具调用） ====================
def parse_tool_calls_from_ai_content(content: str) -> List[Dict[str, Any]]:
    """（原函数，不变）"""
    pattern = r'(\w+)\(([^)]+)\)'
    matches = re.findall(pattern, content, re.DOTALL)
    tool_calls = []
    invalid_placeholders = ["全称", "名称", "编码", "公斤", "负责", "待定", "unknown", "请提供", "输入", "?"]
    for tool_name, args_str in matches:
        args = {}
        arg_pattern = r'(\w+)=("([^"]*)"|\'([^\']*)\'|([^,)]+))'
        for match in re.finditer(arg_pattern, args_str):
            key = match.group(1)
            value = match.group(3) or match.group(4) or match.group(5)
            if value is not None:
                value = value.strip()
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                args[key] = value
        # 占位符检测
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
            return []
        tool_calls.append({"name": tool_name, "args": args})
    return tool_calls


# ==================== 工厂函数：创建受限子 Agent ====================
def create_sub_agent_node(agent_name: str, allowed_tools: List[str], system_extra: str = ""):
    """
    返回一个子 Agent 节点函数，该节点只允许使用指定的工具列表。
    """
    # 构建允许的工具描述
    allowed_tools_set = set(allowed_tools)
    filtered_tools = [t for t in tools if t.name in allowed_tools_set]
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in filtered_tools])

    sys_prompt = f"""你是中远海运订舱助手的【{agent_name}】模块。
你可以使用以下工具：
{tool_descriptions}
{system_extra}
**严格指令**：
- **禁止输出任何解释、推理或过渡语句**（例如“我需要查询”、“为了获取信息”等）。
- 如果用户请求查询信息，直接调用相应工具并输出工具返回的原始结果，不要添加任何额外文字。
- 如果工具已经返回结果，将结果原样输出，不要加前缀或说明。
**核心指令**：
1. 当用户提供了某个工具所需的所有必需参数时，你必须**立即输出该工具调用**，格式为：工具名(参数1="值1", 参数2="值2", ...)
2. 如果用户提供的参数不完整，用自然语言追问缺少的参数，不要输出工具调用。
3. 一次只能调用一个工具。

现在，用户请求：{{user_query}}
请直接输出。
"""

    @traceable
    def sub_agent_node(state: AgentState):
        user_query = state['messages'][-1].content
        messages = [SystemMessage(content=sys_prompt.format(user_query=user_query))] + state["messages"]
        # 绑定允许的工具
        try:
            llm_with_tools = config.base_llm.bind_tools(filtered_tools)
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"子Agent {agent_name} 执行失败:\n{error_trace}")
            # 尝试从 e 中提取有用信息
            error_type = type(e).__name__
            error_msg_text = str(e)
            user_friendly = f"❌ {agent_name} 服务暂时不可用（{error_type}）。请检查 Ollama/Milvus 是否运行。详细信息：{error_msg_text[:200]}"
            return {"messages": [AIMessage(content=user_friendly)], "active_agent": agent_name}
        return {"messages": [AIMessage(content=response.content, tool_calls=response.tool_calls)],
                "active_agent": agent_name}

    return sub_agent_node


# 定义 5 个子 Agent
query_agent_node = create_sub_agent_node(
    "query_agent",
    ["query_space", "get_so_status", "track_container", "get_vgm_deadline"],
    system_extra="注意：你负责舱位、状态、集装箱、VGM 截止时间等查询类任务。"
)
booking_agent_node = create_sub_agent_node(
    "booking_agent",
    ["query_space", "submit_booking", "submit_bl_draft", "get_so_status"],
    system_extra="你负责提交订舱和提单确认件。如果缺少参数，请追问。"
)
compliance_agent_node = create_sub_agent_node(
    "compliance_agent",
    [],  # 合规 Agent 不直接调用工具，而是触发 sensitive_check_node
    system_extra="你负责检查敏感品名和合规要求。如果用户提到敏感品名，请告知需要提供的文件。无需调用工具，直接用自然语言回答。"
)
notify_agent_node = create_sub_agent_node(
    "notify_agent",
    [],
    system_extra="你负责发送通知、创建待办、提醒用户。直接给出友好的回复。"
)
document_agent_node = create_sub_agent_node(
    "document_agent",
    [],
    system_extra="你负责解释文档处理流程。实际上文档解析在系统后台自动完成，回答用户关于托书、图片上传等问题。"
)


# ==================== Supervisor 节点 ====================
def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """根据用户最新消息判断意图，选择对应的子 Agent"""
    last_msg = state["messages"][-1].content.lower()
    # 简单的关键词规则（可扩展为 LLM 分类）
    if any(kw in last_msg for kw in ["舱位", "查询", "查一下", "s/o", "so", "集装箱", "追踪", "vgm", "截止"]):
        intent = "query"
        next_agent = "query_agent"
    elif any(kw in last_msg for kw in ["订舱", "订一个", "订柜子", "提交订舱", "下单", "提单"]):
        intent = "booking"
        next_agent = "booking_agent"
    elif any(kw in last_msg for kw in ["敏感", "合规", "危险品", "msds", "un38.3", "文件"]):
        intent = "compliance"
        next_agent = "compliance_agent"
    elif any(kw in last_msg for kw in ["通知", "提醒", "待办", "邮件"]):
        intent = "notify"
        next_agent = "notify_agent"
    elif any(kw in last_msg for kw in ["图片", "托书", "上传", "解析"]):
        intent = "document"
        next_agent = "document_agent"
    else:
        # 默认交给订舱 Agent
        intent = "booking"
        next_agent = "booking_agent"

    logger.info(f"Supervisor -> intent: {intent}, next_agent: {next_agent}")
    return {"intent": intent, "next_agent": next_agent, "active_agent": next_agent}


# ==================== 原有节点（敏感检测、人工审核、工具执行） ====================
@traceable
def sensitive_check_node(state: AgentState):
    """RAG 敏感品名检测（原函数，不变）"""
    goods = state.get("booking_info", {}).get("goods_name", "")
    if not goods and len(state["messages"]) > 1:
        # 尝试从上一个 AIMessage 的 tool_calls 中提取
        prev_msg = state["messages"][-2] if len(state["messages"]) >= 2 else None
        if prev_msg and hasattr(prev_msg, "tool_calls") and prev_msg.tool_calls:
            for tc in prev_msg.tool_calls:
                if "goods_name" in tc.get("args", {}):
                    goods = tc["args"]["goods_name"]
                    break
    if not goods:
        return {"sensitive_check": {"is_sensitive": False}}
    res = search_sensitive_goods(goods)
    if res.get("is_sensitive"):
        warn = AIMessage(content=f"⚠️ {goods} 属于【{res['risk_level']}】风险品名，需提供 {res['required_docs']}")
        return {"messages": [warn], "sensitive_check": res}
    return {"sensitive_check": res}


@traceable
def human_review_node(state: AgentState):
    """人工审核节点（原函数，不变）"""
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
def tool_node_with_state_update(state: AgentState):
    """执行工具并更新 booking_info（原函数，仅微调）"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    new_booking_info = state.get("booking_info", {}).copy()
    results = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        # 特殊处理 submit_bl_draft 参数补全
        if tool_name == "submit_bl_draft":
            if "bl_draft_json" not in tool_args or not tool_args["bl_draft_json"]:
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
        tool_func = tool_map[tool_name]
        raw_result = tool_func.invoke(tool_args)
        result_data = json.loads(raw_result)
        results.append(ToolMessage(content=raw_result, tool_call_id=tc["id"]))
        if tool_name == "submit_booking" and result_data.get("success"):
            new_booking_info["booking_no"] = result_data["booking_no"]
            new_booking_info["so_no"] = result_data["so_no"]
            new_booking_info.update(tool_args)
    return {"messages": results, "booking_info": new_booking_info}


# ==================== 路由函数 ====================
def route_after_agent(state: AgentState) -> Literal["tools", "human_review", "supervisor", "end"]:
    """子 Agent 执行后，判断是否需要调用工具、人工审核，或返回 Supervisor"""
    messages = state.get("messages", [])
    if not messages:
        return "end"
    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            if tc.get("name") == "submit_bl_draft":
                return "human_review"
        return "tools"
    # 没有工具调用表示任务完成，等待下一轮用户消息
    return "end"


def route_after_tools(state: AgentState) -> str:
    """工具执行后，返回当前活跃的子 Agent 继续处理"""
    return state.get("active_agent", "supervisor")


def route_after_human_review(state: AgentState) -> str:
    """人工审核后返回 Supervisor"""
    return "supervisor"


# ==================== 构建多 Agent 图 ====================
def create_agent():
    """主函数，返回编译后的多 Agent 图（兼容原有调用）"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("query_agent", query_agent_node)
    workflow.add_node("booking_agent", booking_agent_node)
    workflow.add_node("compliance_agent", compliance_agent_node)
    workflow.add_node("notify_agent", notify_agent_node)
    workflow.add_node("document_agent", document_agent_node)
    workflow.add_node("sensitive_check", sensitive_check_node)
    workflow.add_node("tools", tool_node_with_state_update)
    workflow.add_node("human_review", human_review_node)

    # 入口为 Supervisor
    workflow.set_entry_point("supervisor")

    # Supervisor 动态选择子 Agent
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {
            "query_agent": "query_agent",
            "booking_agent": "booking_agent",
            "compliance_agent": "compliance_agent",
            "notify_agent": "notify_agent",
            "document_agent": "document_agent",
        }
    )

    # 每个子 Agent 后都执行敏感品检测
    for agent in ["query_agent", "booking_agent", "compliance_agent", "notify_agent", "document_agent"]:
        workflow.add_edge(agent, "sensitive_check")

    # 敏感检测后路由
    workflow.add_conditional_edges("sensitive_check", route_after_agent, {
        "tools": "tools",
        "human_review": "human_review",
        "supervisor": "supervisor",
        "end": END
    })

    # 工具执行后回到当前子 Agent
    workflow.add_conditional_edges("tools", route_after_tools, {
        "query_agent": "query_agent",
        "booking_agent": "booking_agent",
        "compliance_agent": "compliance_agent",
        "notify_agent": "notify_agent",
        "document_agent": "document_agent",
        "supervisor": "supervisor",
        "end": END
    })

    # 人工审核后返回 Supervisor
    workflow.add_edge("human_review", "supervisor")

    return workflow.compile(checkpointer=MemorySaver())