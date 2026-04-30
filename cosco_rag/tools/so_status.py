import json
from langchain.tools import tool
from langsmith import traceable

from cosco_rag.utils.mock_api import mock_cosco_api

@tool
@traceable
def get_so_status(so_no: str) -> str:
    """查询S/O状态"""
    res = mock_cosco_api("get_so_status", {"so_no": so_no})
    return json.dumps(res)