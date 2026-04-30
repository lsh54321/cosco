import json
from langchain.tools import tool
from langsmith import traceable

from cosco_rag.utils.mock_api import mock_cosco_api

@tool
@traceable
def get_vgm_deadline(booking_no: str, vessel_eta: str = None) -> str:
    """根据订舱号查询VGM截止时间"""
    res = mock_cosco_api("get_vgm_deadline", {"booking_no": booking_no})
    return json.dumps(res)