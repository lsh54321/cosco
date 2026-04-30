import json
from langchain.tools import tool
from langsmith import traceable

from cosco_rag.utils.mock_api import mock_cosco_api
from cosco_rag.knowledge.milvus_client import search_port_code

@tool
@traceable
def query_space(port_of_loading: str, destination: str, container_type: str = "40HQ") -> str:
    """
    查询中远海运舱位可用性。
    参数：
        port_of_loading: 起运港（中文或简称）
        destination: 目的港
        container_type: 箱型，如20GP,40HQ
    返回：舱位状态JSON
    """
    pol = search_port_code(port_of_loading)
    pod = search_port_code(destination)
    res = mock_cosco_api("query_space", {"pol": pol, "pod": pod, "ct": container_type})
    return json.dumps(res)