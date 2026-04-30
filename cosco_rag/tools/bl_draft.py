import json
from langchain.tools import tool
from langsmith import traceable

from cosco_rag.utils.mock_api import mock_cosco_api

@tool
@traceable
def submit_bl_draft(booking_no: str, bl_draft_json: str, reviewer: str = "ops_team") -> str:
    """提交提单确认件，推送给人工审核"""
    res = mock_cosco_api("submit_bl_draft", {"booking_no": booking_no, "draft": bl_draft_json})
    return json.dumps(res)