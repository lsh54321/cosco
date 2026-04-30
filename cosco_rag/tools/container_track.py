import json
from langchain.tools import tool
from langsmith import traceable

from cosco_rag.utils.mock_api import mock_cosco_api

@tool
@traceable
def track_container(container_no: str) -> str:
    """追踪集装箱动态
    追踪集箱号
    """
    res = mock_cosco_api("track_container", {"container_no": container_no})
    return json.dumps(res)