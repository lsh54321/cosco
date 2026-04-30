import json
from langchain.tools import tool
from langsmith import traceable

from cosco_rag.utils.mock_api import mock_cosco_api
from cosco_rag.knowledge.milvus_client import search_port_code, search_hs_code

@tool
@traceable
def submit_booking(
    shipper: str,
    consignee: str,
    goods_name: str,
    hs_code: str,
    weight_kg: float,
    container_type: str,
    port_of_loading: str,
    destination: str
) -> str:
    """
    提交订舱申请到中远海运系统。

    Args:
        shipper: 发货人全称
        consignee: 收货人全称
        goods_name: 货物名称（品名）
        hs_code: 10位HS编码
        weight_kg: 毛重，单位公斤
        container_type: 集装箱类型，如20GP、40HQ
        port_of_loading: 起运港（中文或UN/LOCODE）
        destination: 目的港（中文或UN/LOCODE）

    Returns:
        包含订舱结果的JSON字符串，包括booking_no、so_no等。
    """
    hs_desc = search_hs_code(hs_code)   # 可选验证
    pol = search_port_code(port_of_loading)
    pod = search_port_code(destination)
    res = mock_cosco_api("submit_booking", locals())
    return json.dumps(res)